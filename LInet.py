import torch
import torchvision
import torch.nn as nn 
from torchvision import transforms
from torch.utils.data import DataLoader
from random import randint
from torch.utils.data import Dataset
from PIL import Image
import os

from PIL import Image,ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

# import splitfolders

# input_folder = "C:\Zorage\ML\Research\Camoouflage paper\TestDataset\COD10K"
# output_folder = './Train'
# test_ratio = 0.2  
# splitfolders.ratio(input_folder, output=output_folder, seed=42, ratio=(1 - test_ratio, test_ratio), group_prefix=None)  # Default values

# print("Dataset split into train and test sets successfully!")


class SegmentationDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(img_dir)
        self.masks = os.listdir(mask_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx].replace('jpg', 'png'))
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transform is not None:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask

transform = transforms.Compose([
    transforms.Resize((352,352)),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
])

train_path = "./Train/train/Imgs"
test_path = "./Train/val/Imgs"
val_path = "./Train/val/Imgs"

train_data = SegmentationDataset(img_dir='Train/train/Imgs', mask_dir='Train/train/GT', transform=transform)
val_data = SegmentationDataset(img_dir='Train/train/Imgs', mask_dir='Train/train/GT', transform=transform)
test_data = SegmentationDataset(img_dir='Train/train/Imgs', mask_dir='Train/train/GT', transform=transform)

batch_size = 1

train_loader = DataLoader(dataset=train_data,batch_size=batch_size)
test_loader = DataLoader(dataset=test_data,batch_size=batch_size)
val_loader = DataLoader(dataset=val_data,batch_size=batch_size)

if torch.cuda.is_available:
    device = torch.device('cuda')
    torch.cuda.empty_cache()
else:
    device = torch.device('cpu')

device = 'cpu'

class Conv2d(nn.Module):

    def __init__(self,in_channels, out_channels, kernel_size=(1,1), stride=1, padding=0, dilation=1, bias=False,padding_mode='zeros',relu=True, bn=True):

        super(Conv2d,self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=stride,padding=padding,dilation=dilation,bias=bias,padding_mode=padding_mode)
        self.bn = nn.BatchNorm2d(num_features=out_channels) 
        self.relu = nn.ReLU(inplace=True)

    def forward(self,x):
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        if self.relu:
            x = self.relu(x)

        return x

class RFB(nn.Module):
    def __init__(self, in_channels, out_channels=32):
        super(RFB, self).__init__()

        self.branch1 = nn.Sequential(
            Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1))
        )
        self.branch2 = nn.Sequential(
            Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1))
        )
        self.branch3 = nn.Sequential(
            Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1)),
            Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(1, 7), padding=(0, 3)),
            Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(7, 1), padding=(3, 0)),
            Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3), dilation=7, padding=7, bias=False)
        )
        self.branch4 = nn.Sequential(
            Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1)),
            Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(1, 5), padding=(0, 2)),
            Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(5, 1), padding=(2, 0)),
            Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3), dilation=5, padding=5, bias=False)
        )
        self.branch5 = nn.Sequential(
            Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1)),
            Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(1, 3), padding=(0, 1)),
            Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 1), padding=(1, 0)),
            Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3), dilation=3, padding=3, bias=False)
        )

        self.conv11 = nn.Conv2d(in_channels=out_channels * 4, out_channels=out_channels, kernel_size=(1, 1))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        b5 = self.branch5(x)
        b_concat = torch.cat([b2, b3, b4, b5], dim=1)
        b_concat = self.conv11(b_concat)
        
        b = torch.add(b1, b_concat)
        b = self.relu(b)
        return b

class DC(nn.Module):

    def __init__(self,in_channels,out_channels) -> None:
        
        super(DC,self).__init__()
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.conv_up1 = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), padding=1)
        self.conv_up2 = Conv2d(in_channels=2*in_channels, out_channels=2*out_channels, kernel_size=(3, 3), padding=1)

        self.conv_concat1 = Conv2d(in_channels=2*in_channels, out_channels=2*out_channels, kernel_size=(3, 3), padding=1)
        self.conv_concat2 = Conv2d(in_channels=4*in_channels, out_channels=4*out_channels, kernel_size=(3, 3), padding=1)
        self.conv_11 = Conv2d(in_channels=4*in_channels, out_channels=1, kernel_size=(1, 1))

    def forward(self,x1,x2,x3,x4):
        x1 = x1
        x2 = x2

        x3_1 = x3
        x3_2 = self.conv_up1(self.up2(x3))

        x4_1 = self.conv_up1(self.up2(x4))
        x4_2 = self.conv_up1(self.up2(x4))
        x4_3 = self.conv_up1(self.up4(x4))

        x4_1 = torch.multiply(x4_1,x3_1)
        x4_1 = torch.cat([x4_1,x4_2],dim=1)

        x4_3 = torch.multiply(torch.multiply(x2,x3_2),x4_3)

        x4_1 = self.conv_concat1(x4_1)
        x4_1 = self.up2(x4_1)
        x4_1 = self.conv_concat1(x4_1)

        print(x1.shape)
        print(x4_1.shape)
        print(x4_3.shape)

        x4_3 = torch.cat([x1,x4_1,x4_3],dim=1)
        x4_3 = self.conv_concat2(x4_3)
        x4_3 = self.conv_concat2(x4_3)
        x4_3 = self.conv_11(x4_3)

        return x4_3


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self):
        self.inplanes = 64
        super(ResNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer2 = self._make_layer(Bottleneck, 64, 3)
        self.layer3 = self._make_layer(Bottleneck, 128, 4, stride=2)
        self.layer4 = self._make_layer(Bottleneck, 256, 6, stride=2)
        self.layer5 = self._make_layer(Bottleneck, 512, 3, stride=2)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = x
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)


        return x1, x2, x3, x4, x5

from turtle import shape


class LINet(nn.Module):

    def __init__(self) -> None:
        super(LINet,self).__init__()

        self.resnet = ResNet()

        self.up_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up_8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.down_2 = nn.MaxPool2d(kernel_size=(2,2),stride=2)
        
        self.rf1 = RFB(in_channels=320)
        self.rf2 = RFB(in_channels=3584)
        self.rf3 = RFB(in_channels=3072)
        self.rf4 = RFB(in_channels=2048)

        self.dc = DC(in_channels=32,out_channels=32)

    
    def forward(self,x):
        x1,x2,x3,x4,x5 = self.resnet(x)
        rf1_in = self.down_2(torch.cat([x1,x2],dim=1))
        rf1_out = self.rf1(rf1_in)

        rf2_in = torch.cat([x3,self.up_2(x4),self.up_4(x5)],dim=1)
        rf2_out = self.rf2(rf2_in)

        rf3_in = torch.cat([x4,self.up_2(x5)],dim=1)
        rf3_out = self.rf3(rf3_in)

        rf4_in = x5
        rf4_out = self.rf4(rf4_in)

        x = self.dc(rf1_out,rf2_out,rf3_out,rf4_out)
        print(x.shape)
        x = self.up_8(x)

        return x

model = LINet()

def eval_mae(y_pred, y):
    return torch.abs(y_pred - y).mean()

def train(model, optimizer, loss_fn, train_loader, val_loader, epochs=2,device=device):
    for epoch in range(1, epochs+1):
        training_loss = 0.0
        valid_loss = 0.0
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            output = model(inputs)
            loss = loss_fn(output, targets)
            loss.backward()
            optimizer.step()
            training_loss += loss.data.item() * inputs.size(0)
        training_loss /= len(train_loader.dataset)
        
        model.eval()
        for batch in val_loader:
            inputs, targets = batch
            inputs = inputs.to(device)
            output = model(inputs)
            targets = targets.to(device)
            loss = loss_fn(output, targets) 
            valid_loss += loss.data.item() * inputs.size(0)
            valid_mae += eval_mae(output,targets) * inputs.size(0)
        valid_loss /= len(val_loader.dataset)
        valid_mae /= len(val_loader.dataset)

        print('Epoch: {}, Training Loss: {:.2f}, Validation Loss: {:.2f}, accuracy = {:.2f}'.format(epoch, training_loss,
        valid_loss,valid_mae))

optimizer = torch.optim.Adam(params=model.parameters(),lr=1e-4)
loss = nn.CrossEntropyLoss()
# lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True)

train(model=model,optimizer=optimizer,loss_fn=loss,train_loader=train_loader,val_loader=val_loader,epochs=1,device=device)




