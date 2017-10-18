import argparse
import os
import sys
import time

import numpy as np
import torch
from torch.autograd import Variable
from PIL import Image
#from torch.optim import Adam
#from torch.utils.data import DataLoader
#from torchvision import datasets
#from torchvision import transforms

import utils
from transformer_net import TransformerNet
from vgg16 import Vgg16

def stylize(args):
    content_image = utils.tensor_load_rgbimage(args.content_image, size = 256, scale=args.content_scale)
    content_image = content_image.unsqueeze(0)

    if args.cuda:
        content_image = content_image.cuda()
    content_image = Variable(utils.preprocess_batch(content_image), volatile=True)
    style_model = TransformerNet()
    style_model.load_state_dict(torch.load(args.model))

    if args.cuda:
        style_model.cuda()

    output = style_model(content_image)
    #return output
    #return utils.tensor_save_bgrimage(output.data[0], args.output_image, args.cuda)
    return utils.tensor_save_bgrimage(output.data[0], args.cuda)


def main(args):
    if args.cuda and not torch.cuda.is_available():
        print("ERROR: cuda is not available, try running on CPU")
        sys.exit(1)
    return stylize(args)


if __name__ == "__main__":
    eval_arg_parser = argparse.ArgumentParser(description="parser for fast-neural-style")
    
    eval_arg_parser.add_argument("--content-image", type=str, required=True,
                                 help="path to content image you want to stylize")
    eval_arg_parser.add_argument("--content-scale", type=float, default=None,
                                 help="factor for scaling down the content image")
    eval_arg_parser.add_argument("--output-image", type=str, required=True,
                                 help="path for saving the output image")
    eval_arg_parser.add_argument("--model", type=str, required=True,
                                 help="saved model to be used for stylizing the image")
    eval_arg_parser.add_argument("--cuda", type=int, required=True,
                                 help="set it to 1 for running on GPU, 0 for CPU")

    args = eval_arg_parser.parse_args()

    output_image = main(args)
    
    output_image.save(args.output_image)