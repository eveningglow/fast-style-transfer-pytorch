import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import numpy as np

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel=3, stride=2, pad=1, norm_type='batch', act_type='relu'):
        super(ConvBlock, self).__init__()
        layers = []
        layers += [nn.ReflectionPad2d(pad), 
                   nn.Conv2d(in_c, out_c, kernel_size=kernel, stride=stride, padding=0)]
        
        if norm_type is 'batch':
            layers += [nn.BatchNorm2d(out_c, affine=True)]
        elif norm_type is 'instance':
            layers += [nn.InstanceNorm2d(out_c, affine=True)]
        elif norm_type is None:
            pass
        
        if act_type is 'relu': 
            layers += [nn.ReLU()]
        elif act_type is 'tanh':
            layers += [nn.Tanh()]
        elif act_type is None:
            pass
        
        self.block = nn.Sequential(*layers)
        
    def forward(self, x):
        out = self.block(x)
        return out
    
class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(ConvBlock(channels, channels, kernel=3, stride=1, pad=1, norm_type='instance',
                                             act_type='relu'),
                                   ConvBlock(channels, channels, kernel=3, stride=1, pad=1, norm_type='instance',
                                             act_type=None))
    
    def forward(self, x):
        out = self.block(x) + x
        return out

class ConvTransBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel=3, stride=2, pad=1, out_pad=1, norm_type='batch'):
        super(ConvTransBlock, self).__init__()
        layers = []
        
        # Conv transpose layer
        layers += [nn.ConvTranspose2d(in_c, out_c, kernel_size=kernel, stride=stride, padding=pad,
                                      output_padding=out_pad)]
        
        # Normalization layer
        if norm_type is 'batch':
            layers += [nn.BatchNorm2d(out_c, affine=True)]
        elif norm_type is 'instance':
            layers += [nn.InstanceNorm2d(out_c, affine=True)]

        # Activiation layer
        layers += [nn.ReLU()]
        
        self.conv_trans_block = nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv_trans_block(x)
        return out
    
class ImageTransformNet(nn.Module):
    def __init__(self, res_block_num=5):
        super(ImageTransformNet, self).__init__()
        # Downsampling blocks
        self.downsamples = nn.Sequential(ConvBlock(3, 32, kernel=9, stride=1, pad=4, norm_type='instance',
                                                   act_type='relu'),
                                         ConvBlock(32, 64, kernel=3, stride=2, pad=1, norm_type='instance',
                                                   act_type='relu'),
                                         ConvBlock(64, 128, kernel=3, stride=2, pad=1, norm_type='instance',
                                                   act_type='relu'))
        
        # Residual blocks
        res = []
        for _ in range(res_block_num): 
            res += [ResBlock(128)]
        self.residuals = nn.Sequential(*res)    

        # Upsampling blocks
        self.upsamples = nn.Sequential(ConvTransBlock(128, 64, kernel=3, stride=2, pad=1, out_pad=1,
                                                      norm_type='instance'),
                                       ConvTransBlock(64, 32, kernel=3, stride=2, pad=1, out_pad=1,
                                                      norm_type='instance'),
                                       ConvBlock(32, 3, kernel=9, stride=1, pad=4, norm_type=None,
                                                 act_type='tanh'))

    def forward(self, x):
        out = self.downsamples(x)
        out = self.residuals(out)
        out = self.upsamples(out)
        out = (out + 1) / 2
        return out



class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        print('Preparing pretrained VGG 16 ...')
        self.vgg_16 = torchvision.models.vgg16(pretrained=True).features
        
        self.relu_1_2 = nn.Sequential(*list(self.vgg_16.children())[0:4])
        self.relu_2_2 = nn.Sequential(*list(self.vgg_16.children())[4:9])
        self.relu_3_3 = nn.Sequential(*list(self.vgg_16.children())[9:16])
        self.relu_4_3 = nn.Sequential(*list(self.vgg_16.children())[16:23])
    
    def forward(self, x):
        out_1_2 = self.relu_1_2(x)
        out_2_2 = self.relu_2_2(out_1_2)
        out_3_3 = self.relu_3_3(out_2_2)
        out_4_3 = self.relu_4_3(out_3_3)

        return [out_1_2, out_2_2, out_3_3, out_4_3]