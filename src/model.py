import torch
import torch.nn as nn
import torchvision.models as models

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.GELU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class BottleneckAttention(nn.Module):
    def __init__(self, in_channels):
        super(BottleneckAttention, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.tensor(0.01))
        self.fusion_conv = nn.Conv2d(2 * in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        proj_query = self.query(x).view(B, -1, H * W).permute(0, 2, 1)
        proj_key = self.key(x).view(B, -1, H * W)
        proj_value = self.value(x).view(B, -1, H * W)

        d_k = proj_query.shape[-1]
        attention = torch.bmm(proj_query, proj_key) / (d_k ** 0.5)
        attention = torch.softmax(attention, dim=-1)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(B, C, H, W)

        out = self.gamma * out
        out = torch.cat([out, x], dim=1)
        out = self.fusion_conv(out)
        return out


# class SelfAttention(nn.Module):
#     def __init__(self, in_channels):
#         super(SelfAttention, self).__init__()
#         self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
#         self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
#         self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
#         self.gamma = nn.Parameter(torch.zeros(1))

#     def forward(self, x):
#         B, C, H, W = x.shape
#         proj_query = self.query(x).view(B, -1, H * W).permute(0, 2, 1)  # B, HW, C'
#         proj_key = self.key(x).view(B, -1, H * W)                       # B, C', HW
#         attention = torch.bmm(proj_query, proj_key)                    # B, HW, HW
#         attention = torch.softmax(attention, dim=-1)
#         proj_value = self.value(x).view(B, -1, H * W)                  # B, C, HW

#         out = torch.bmm(proj_value, attention.permute(0, 2, 1))        # B, C, HW
#         out = out.view(B, C, H, W)
#         return self.gamma * out + x
    
class ConvolutionalAutoEncoder(nn.Module):
    def __init__(self):
        super(ConvolutionalAutoEncoder,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,stride=2,padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=2,padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=2,padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.SE1 = SEBlock(32,16)
        self.SE2 = SEBlock(64,16)
        self.SE3 = SEBlock(128,16)
        self.SE4 = SEBlock(64,16)
        self.SE5 = SEBlock(32,16)
        self.attn = BottleneckAttention(128)
        #self.maxPool = nn.MaxPool2d(kernel_size=2,stride=1,return_indices=True)
        
        
        self.conv1Trans = nn.ConvTranspose2d(in_channels=128,out_channels=64,kernel_size=3,stride=2,padding=1,output_padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv2Trans = nn.ConvTranspose2d(in_channels=64,out_channels=32,kernel_size=3,stride=2,padding=1,output_padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.conv3Trans = nn.ConvTranspose2d(in_channels=32,out_channels=3,kernel_size=3,stride=2,padding=1,output_padding=1)
        self.bn6 = nn.BatchNorm2d(3)
        self.convoutput = nn.Conv2d(in_channels=3,out_channels=3,kernel_size=3,stride=1,padding=1)
        #self.maxUnpool = nn.MaxUnpool2d(kernel_size=2,stride=1)
        
        self.LReLU = nn.LeakyReLU(0.1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self,x):
        x1 = self.LReLU(self.bn1(self.conv1(x)))
        
        x1 = self.SE1(x1)
        
        x2 = self.LReLU(self.bn2(self.conv2(x1)))
        
        x2 = self.SE2(x2)
        
        x3 = self.LReLU(self.bn3(self.conv3(x2)))
        
        x3 = self.SE3(x3)
        
        x3 = self.attn(x3)
        
        x4 = self.LReLU(self.bn4(self.conv1Trans(x3)))
        
        x4 = x4 + x2
        
        x4 = self.SE4(x4)
        
        x5 = self.LReLU(self.bn5(self.conv2Trans(x4)))
        
        x5 = x5 + x1
        
        x5 = self.SE5(x5)
        
        x6 = self.LReLU(self.bn6(self.conv3Trans(x5)))
        
        x6 = self.sigmoid(self.convoutput(x6))

        return x6

class VGGFeatureExtractor(nn.Module):
    def __init__(self, layers=['relu1_2','relu2_2']):
        super().__init__()
        vgg = models.vgg19(pretrained=True).features.eval()

        self.selected_layers = {
            'relu1_2': 3,
            'relu2_2': 8
        }
        
        self.layers = layers
        self.vgg = nn.Sequential(*[vgg[i] for i in range(max(self.selected_layers.values()) + 1)])
        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, x):
        features = {}
        for name, layer in enumerate(self.vgg):
            x = layer(x)
            for k, v in self.selected_layers.items():
                if name == v:
                    features[k] = x
        return features