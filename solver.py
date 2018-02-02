import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torchvision
import os

import model
import dataloader
import util

class Solver():
    def __init__(self, trn_dir, style_path, result_dir, weight_dir, num_epoch=2, batch_size=4,
                 content_loss_pos=2, lr=1e-3, lambda_c=1, lambda_s=5e+5, show_every=1000, save_every=5000):
        
        if torch.cuda.is_available():
            self.dtype = torch.cuda.FloatTensor
        else:
            self.dtype = torch.FloatTensor
            
        self.style_path = style_path
        self.result_dir = result_dir
        self.weight_dir = weight_dir
        
        # Models
        self.trans_net = model.ImageTransformNet().type(self.dtype)
        self.vgg16 = model.VGG16().type(self.dtype)
        
        # Dataloader
        self.dloader, total_num = dataloader.data_loader(root=trn_dir, batch_size = batch_size)
        self.total_iter = int(total_num / batch_size) + 1
        
        # Loss function and optimizer
        self.mse_loss = nn.MSELoss()
        self.optimizer = optim.Adam(self.trans_net.parameters(), lr=lr)
        
        # Hyperparameters
        self.content_loss_pos = content_loss_pos
        self.lambda_c = lambda_c
        self.lambda_s = lambda_s
        self.show_every = show_every
        self.save_every = save_every
        self.num_epoch = num_epoch
        
    def train(self):
        # Process on style image. Only need to be done once.
        style_img = util.load_img(self.style_path, size=(256, 256)).type(self.dtype)
        _style_img = style_img.clone()
        style_img = Variable(style_img)
        style_img = util.vgg_norm(style_img)

        style_relu = self.vgg16(style_img)
        gram_target = util.gram(style_relu)

        for epoch in range(self.num_epoch):
            for iters, (trn_img, _) in enumerate(self.dloader):
                # Forward training images to ImageTransformNet
                trn_img = Variable(trn_img.type(self.dtype))
                trn_img = util.vgg_norm(trn_img)
                out_img = self.trans_net(trn_img)
                
                content_img = Variable(trn_img.data.clone())
                out_img = util.vgg_norm(out_img)

                # Forward training img and content img to VGG16
                relu_target = self.vgg16(content_img)
                relu_out = self.vgg16(out_img)

                # Get 4 activations from VGG16
                feature_y = relu_out[self.content_loss_pos]
                feature_t = Variable(relu_target[self.content_loss_pos].data, requires_grad=False)
                
                # Content loss
                content_loss = self.lambda_c * self.mse_loss(feature_y, feature_t)

                # Gram matrix
                gram_out = util.gram(relu_out)

                # Style matrix
                style_loss = 0
                for i in range(len(gram_target)):
                    gram_y = gram_out[i]
                    gram_t = Variable(gram_target[i].expand_as(gram_out[i]).data, requires_grad=False)
                    style_loss += self.lambda_s * self.mse_loss(gram_y, gram_t)

                loss = content_loss + style_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if iters % self.show_every == 0:
                    print('[Epoch : (%d / %d), Iters : (%d / %d)] => Content : %f, Style : %f' \
                          %(epoch + 1, self.num_epoch, iters, self.total_iter, content_loss.data[0], style_loss.data[0]))
                    
                    _, style_name = os.path.split(self.style_path)
                    style_name, _ = os.path.splitext(style_name)
                    result_dir = os.path.join(self.result_dir, style_name)
                    
                    if os.path.exists(result_dir) is not True:
                        os.makedirs(result_dir)
                        
                    file_name = str(epoch) + '_' + str(iters) + '.png'
                    file_name = os.path.join(result_dir, file_name)
                    
                    # Denorm the img to get correct img
                    content_img = util.vgg_denorm(content_img)
                    out_img = util.vgg_denorm(out_img)
                    
                    util.save_img(file_name, content_img.data[0], _style_img[0], out_img.data[0])
                    
                if iters % self.save_every == 0:
                    if os.path.exists(self.weight_dir) is not True:
                        os.mkdir(self.weight_dir)
                    
        weight_name = style_name + '.weight'
        weight_path = os.path.join(self.weight_dir, weight_name)
        torch.save(self.trans_net.state_dict(), weight_path)