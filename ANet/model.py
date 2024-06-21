import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class GV_CNN(nn.Module):
    def __init__(self):
        super(GV_CNN, self).__init__()
        self.vgg16 = models.vgg16(weights='VGG16_Weights.IMAGENET1K_V1')
        self.features = nn.Sequential(*list(self.vgg16.features.children())[:-3])
        self.fc = nn.Linear(14 * 14 * 512, 784)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        feature_maps: list = []  
        for layer in self.features:
            if isinstance(layer, nn.MaxPool2d):
                feature_maps.append(x)
            x = layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.sigmoid(x)
        x = x.view(1, -1, 28, 28)
        feature_maps.append(x)
        return feature_maps
    
class RCL(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, num_time_steps: int = 3, alpha: float = 0.001, beta: float = 0.75, n_size: int = 7):
        super(RCL, self).__init__()
        
        self.num_time_steps = num_time_steps
        self.alpha = alpha
        self.beta = beta
        self.n_size = n_size
        
        self.ff_conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.recurrent_conv = nn.Conv2d(out_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.bias = nn.Parameter(torch.zeros(out_channels))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u: torch.Tensor = self.ff_conv(x)
        
        batch_size, out_channels, height, width = u.size()
        x_recurrent: torch.Tensor = torch.zeros_like(u) 
        
        for time_step in range(self.num_time_steps):
            z: torch.Tensor = u + self.recurrent_conv(x_recurrent) + self.bias.view(1, -1, 1, 1)
            f: torch.Tensor = F.relu(z)
            
            squared_sum: torch.Tensor = torch.zeros_like(f)
            for i in range(out_channels):
                start: int = max(0, i - self.n_size // 2)
                end: int = min(out_channels, i + self.n_size // 2 + 1)
                
                squared_sum[:, i, :, :] = torch.sum(f[:, start:end, :, :] ** 2, dim=1)
                
            squared_sum = squared_sum / self.n_size
            x_recurrent = f / (1 + self.alpha * squared_sum) ** self.beta
        
        return x_recurrent  
    
class HRCNN(nn.Module):
    def __init__(self, feature_maps: list[torch.Tensor]):
        super(HRCNN, self).__init__()
        self.feature_maps = feature_maps
        
    def forward(self, sm_G: torch.Tensor) -> list[torch.Tensor]:
        saliency_maps: list = []
        saliency_maps.append(sm_G)
        
        concat_map1: torch.Tensor = torch.cat((sm_G, self.feature_maps[3]), dim=1)
        rcl4: RCL = RCL(sm_G.shape[1] + self.feature_maps[3].shape[1], 1, kernel_size=3)
        sm_RCL4: torch.Tensor = rcl4(concat_map1)
        saliency_maps.append(sm_RCL4)
        
        sm_RCL4 = F.interpolate(sm_RCL4, scale_factor=2, mode='bilinear', align_corners=False)
        concat_map2: torch.Tensor = torch.cat((sm_RCL4, self.feature_maps[2]), dim=1)
        rcl3: RCL = RCL(sm_RCL4.shape[1] + self.feature_maps[2].shape[1], 1, kernel_size=3)
        sm_RCL3: torch.Tensor = rcl3(concat_map2)
        saliency_maps.append(sm_RCL3)
        
        sm_RCL3 = F.interpolate(sm_RCL3, scale_factor=2, mode='bilinear', align_corners=False)
        concat_map3: torch.Tensor = torch.cat((sm_RCL3, self.feature_maps[1]), dim=1)
        rcl2: RCL = RCL(sm_RCL3.shape[1] + self.feature_maps[1].shape[1], 1, kernel_size=3)
        sm_RCL2: torch.Tensor = rcl2(concat_map3)
        saliency_maps.append(sm_RCL2)
        
        sm_RCL2 = F.interpolate(sm_RCL2, scale_factor=2, mode='bilinear', align_corners=False)
        concat_map4: torch.Tensor = torch.cat((sm_RCL2, self.feature_maps[0]), dim=1)
        rcl1: RCL = RCL(sm_RCL2.shape[1] + self.feature_maps[0].shape[1], 1, kernel_size=3)
        sm_RCL1: torch.Tensor = rcl1(concat_map4)
        saliency_maps.append(sm_RCL1)
        
        return saliency_maps
             
            
class Classification_Stream(nn.Module):
    def __init__(self, input_dim: int):
        super(Classification_Stream, self).__init__()
        self.fc1 = nn.Linear(input_dim, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 2)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = F.softmax(x, dim=1)
        return x

    
if __name__ == "__main__":
    
    """
    # Sample input tensor
    sample_input = torch.randn(1, 3, 224, 224)  # Batch size 1, 3 channels, 224x224 image

    # Initialize the GV_CNN model
    gv_cnn = GV_CNN()

    # Pass the sample input through the model
    outputs = gv_cnn(sample_input)

    for i, output in enumerate(outputs):
        print(f"Feature map {i+1} shape: {output.shape}")
    
    Feature map 1 shape: torch.Size([1, 64, 224, 224])
    Feature map 2 shape: torch.Size([1, 128, 112, 112])
    Feature map 3 shape: torch.Size([1, 256, 56, 56])
    Feature map 4 shape: torch.Size([1, 512, 28, 28])
    Feature map 5 shape: torch.Size([1, 1, 28, 28])
    """
    
    """
    # Create a dummy input image
    input_image = torch.randn(1, 3, 224, 224)

    # Create an instance of the RCL module
    rcl = RCL(in_channels=3, out_channels=1, kernel_size=3)

    # Pass the input image through the RCL module
    output_features = rcl(input_image)

    print("Input Image Shape:", input_image.shape)
    print("Output Features Shape:", output_features.shape)

    Input Image Shape: torch.Size([1, 3, 224, 224])
    Output Features Shape: torch.Size([1, 1, 224, 224])
    """
    
    """
    sample_input = torch.randn(1, 3, 224, 224)  # Batch size of 1, 3 color channels, 224x224 image

    # Instantiate GV_CNN model and get feature maps
    gv_cnn = GV_CNN()
    feature_maps = gv_cnn(sample_input)

    # Sample saliency map for HRCNN model (matching dimensions of the output of GV_CNN)
    sample_sm_G = feature_maps[-1]  # Use the final output from GV_CNN as the initial saliency map
    
    # Instantiate HRCNN model with the obtained feature maps
    hrcnn = HRCNN(feature_maps)

    # Pass the sample saliency map through HRCNN
    saliency_maps = hrcnn(sample_sm_G)

    # Print the shapes of the resulting saliency maps
    for i, saliency_map in enumerate(saliency_maps):
        print(f"Saliency map {i+1} shape: {saliency_map.shape}")
        
    Saliency map 1 shape: torch.Size([1, 1, 28, 28])
    Saliency map 2 shape: torch.Size([1, 1, 28, 28])
    Saliency map 3 shape: torch.Size([1, 1, 56, 56])
    Saliency map 4 shape: torch.Size([1, 1, 112, 112])
    Saliency map 5 shape: torch.Size([1, 1, 224, 224])
    """