import torch
from torch.autograd import Variable
import torchvision
import torchvision.transforms as Transforms
from PIL import Image
import numpy as np

# Load image with size of parameter size
def load_img(path, size=None):
    img = Image.open(path).convert('RGB')

    transform_list = []
    if size is not None:
        transform_list += [Transforms.Scale(size)]

    transform_list += [Transforms.ToTensor()]
    transform = Transforms.Compose(transform_list)
    
    img = transform(img)
    img = img.unsqueeze(dim=0)
    
    return img

# Make image with shape of [content | result | style] and save it
def save_img(img_name, content, style, result):
    _, H, W = content.size()
    size = (H, W)

    img = torch.stack([content, result, style], dim=0)
    torchvision.utils.save_image(img, img_name, nrow=3)

# Load pretrained weight
def load_weight(model, path):
    model.load_state_dict(torch.load(path))
    return model

# Normalization with mean and std of VGG
def vgg_norm(var):
    if torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor
    
    mean = Variable(torch.zeros(var.size()).type(dtype))
    std = Variable(torch.zeros(var.size()).type(dtype))
    
    mean[:, 0, :, :] = 0.485
    mean[:, 1, :, :] = 0.456
    mean[:, 2, :, :] = 0.406
    
    std[:, 0, :, :] = 0.229
    std[:, 1, :, :] = 0.224
    std[:, 2, :, :] = 0.225

    normed = var.sub(mean).div(std)
    return normed

# Denormalization with mean and std of VGG
def vgg_denorm(var):
    if torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    mean = Variable(torch.zeros(var.size()).type(dtype))
    std = Variable(torch.zeros(var.size()).type(dtype))
    
    mean[:, 0, :, :] = 0.485
    mean[:, 1, :, :] = 0.456
    mean[:, 2, :, :] = 0.406
    
    std[:, 0, :, :] = 0.229
    std[:, 1, :, :] = 0.224
    std[:, 2, :, :] = 0.225

    normed = var.mul(std).add(mean)
    return normed

# Get gram matrix
def gram(var_list):
    gram_list = []
    
    for i in range(len(var_list)):
        var = var_list[i]
        N, C, H, W = var.size()
        var = var.view(N, C, H*W)
        g = torch.bmm(var, var.transpose(2, 1)) / (C * H * W)
        gram_list.append(g)
        
    return gram_list