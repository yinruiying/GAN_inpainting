# -*- coding: utf-8 -*-
"""
Create on 2019/2/28 15:40
Create by ring
Function Description:
"""
import argparse
def arg_parse():
    parser = argparse.ArgumentParser(description="Training parameters for GANs")
    parser.add_argument("--nz", default=100, type=int, help="Length of input vector")
    parser.add_argument("--ngf", default=64, type=int, help="Feature channel of a convolution layer in G")
    parser.add_argument("--input_dim", default=3, type=int, help="input channels")
    parser.add_argument("--nc", default=3, type=int, help="Output channel of G")
    parser.add_argument("--ndf", default=64, type=int, help="Feature channel of a convolution layer in D")
    parser.add_argument("--gpu_mode", default=False, action="store_true", help="Using GPU or not")
    parser.add_argument("--batch_size", default=64, type=int, help="Batch size")
    parser.add_argument("--epochs", default=1000, type=int, help="Training epochs")
    parser.add_argument("--lrG", default=0.0002, type=float, help="Learning rate of G")
    parser.add_argument("--lrD", default=0.0002, type=float, help="Learning rate of D")
    parser.add_argument("--beta1", default=0.5, type=float, help="beta for adam")
    parser.add_argument("--beta2", default=0.999, type=float, help="beta for adam")
    parser.add_argument("--image_size", default=128, type=int, help="Image size, if changed, rebuild the net")
    parser.add_argument("--dataroot", default='', required=True, type=str, help="root dir of images, root/c1/*.png")
    parser.add_argument("--output_dir", default='', required=True, type=str, help="output dir")
    parser.add_argument("--dataset", default='face', type=str, help="dataset name")
    parser.add_argument("--model_name", default="DCGAN", type=str, help="model name")
    parser.add_argument("--complete", default=False, action="store_true", help="whether to complete")
    parser.add_argument("--test_image_dir", default='', required=True, type=str, help="test image dir")
    parser.add_argument("--test_mask_dir", default='', required=True, type=str, help="test mask dir")
    parser.add_argument("--lamd", default=0.1, type=float, help="lamd")
    parser.add_argument("--lr", default=0.01, type=float, help="learning rate")
    parser.add_argument("--num_iters", default=40000, type=int, help="num iters")

    args = parser.parse_args()
    return args

def main():
    args = arg_parse()
    print(args)
    if args.model_name == "DCGAN":
        from libs.models.DCGAN import DCGAN
        gan = DCGAN(args)
    elif args.model_name == "WGAN":
        from libs.models.WGAN import WGAN
        gan = WGAN(args)
    elif args.model_name == "DCGAN_dilation":
        from libs.models.DCGAN_dilation import DCGAN
        gan = DCGAN(args)
    elif args.model_name == "WGAN_GP":
        from libs.models.WGAN_GP import WGAN_GP
        gan = WGAN_GP(args)
    else:
        raise RuntimeError("Model name not supported.")
    if args.complete:
        gan.complete(args)
    else:
        gan.train()
if __name__ == '__main__':
    main()