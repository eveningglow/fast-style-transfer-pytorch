import model
import dataloader
import util

import torch
from torch.autograd import Variable
import torchvision

import argparse
import os
import datetime

def main(args):
    if torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    print('Loading the model...')
    trans_net = model.ImageTransformNet().type(dtype)
    trans_net = util.load_weight(model=trans_net, path=args.weight_path)

    print('Loading the model is done!')
    # content_img = (1, 3, 256, 256)
    content_img = util.load_img(path=args.content_path)
    content_img = Variable(content_img.type(dtype))
    content_img = util.vgg_norm(content_img)

    # result_img = (1, 3, 256, 256)
    result_img = trans_net(content_img)        
    
    # content_img = (1, 3, 256, 256)
    content_img = util.vgg_denorm(content_img)
    
    out_dir, _ = os.path.split(args.output_path)
    if os.path.exists(out_dir) is not True:
        os.mkdir(out_dir)

    torchvision.utils.save_image(result_img.data, args.output_path, nrow=1)
    print('Saved image : ' + args.output_path)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight_path', type=str, default='weight/picasso.weight',
                        help='Model weight path')
    parser.add_argument('--content_path', type=str, default='content/korea_univ.png',
                        help='Content img path')
    parser.add_argument('--output_path', type=str, default='example/korea_univ_picasso.png',
                        help='Output img path')
    args = parser.parse_args()
    main(args)
