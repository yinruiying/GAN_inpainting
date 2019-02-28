# -*- coding: utf-8 -*-
"""
Create on 2019/2/28 15:40
Create by ring
Function Description:
"""

from __future__ import print_function
import os

import cv2
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn

from libs.image_utils import save_tensor_images
from libs.utils import loss_plot, pil_loader, read_mask, loss_plot_multi_pkls

torch.manual_seed(2019)
torch.cuda.manual_seed_all(2019)
import torch.optim as optim
import torch.utils.data
import pickle as pkl
import torchvision.datasets as dset
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
# custom weights initialization called on netG and netD
from libs import utils


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

# Generator Code
class Generator(nn.Module):
    def __init__(self, nz, ngf, nc):
        """
        :param nz: 输入向量长度
        :param ngf: 中间特征图通道数
        :param nc: 生成图片通道数
        :return:
        """
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False, dilation=1),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False, dilation=3),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False, dilation=3),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False, dilation=3),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, ngf//2, 4, 2, 1, bias=False, dilation=3),
            nn.BatchNorm2d(ngf//2),
            nn.ReLU(True),
            # state size. (ngf//2) x 64 x 64
            nn.ConvTranspose2d(ngf//2, nc, 4, 2, 1, bias=False, dilation=3),
            nn.Tanh()
            # state size. (nc) x 128 x 128
        )

    def forward(self, x):
        # [batch, nz, 1, 1]
        return self.main(x)

# Discriminator Code
class Discriminator(nn.Module):
    def __init__(self, nc, ndf):
        """
        :param nc:图片通道数
        :param ndf: 中间特征图通道数
        :return:
        """
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 128 x 128
            nn.Conv2d(nc, ndf, kernel_size=4, stride=2, padding=4, bias=False, dilation=3),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 64 x 64
            nn.Conv2d(ndf, ndf, kernel_size=4, stride=2, padding=4, bias=False, dilation=3),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=4, bias=False, dilation=3),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=4, bias=False, dilation=3),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=4, bias=False, dilation=3),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False, dilation=1),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

class DCGAN(object):
    def __init__(self, args):
        self.nz = args.nz
        self.ngf = args.ngf
        self.nc = args.nc
        self.ndf = args.ndf
        self.G = Generator(self.nz, self.ngf, self.nc)
        self.D = Discriminator(self.nc, self.ndf)
        self.gpu_mode = args.gpu_mode
        if torch.cuda.is_available() and self.gpu_mode:
            self.device = torch.device("cuda:0")
            torch.backends.cudnn.benchmark = True
        else:
            self.device = torch.device("cpu")
        self.batch_size = args.batch_size
        self.epochs = args.epochs

        self.lrG = args.lrG
        self.lrD = args.lrD
        self.beta1 = args.beta1
        self.beta2 = args.beta2

        self.image_size = args.image_size
        self.dataroot = args.dataroot
        self.output_dir = args.output_dir
        self.dataset = args.dataset
        self.model_name = args.model_name
        self.model_dir = os.path.join(self.output_dir, self.dataset, self.model_name)
        self.result_dir = os.path.join(self.output_dir, self.dataset, self.model_name, "results")
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)
        self.resume = os.path.join(self.model_dir, "model.pth")
        self.history = os.path.join(self.model_dir, "history.pkl")

    def load(self):
        checkpoint = torch.load(self.resume)
        self.start_epoch = checkpoint['epoch']
        self.iters = checkpoint['iters']
        self.G.load_state_dict(checkpoint['G'])
        self.D.load_state_dict(checkpoint['D'])

        datas = pkl.load(open(self.history, 'rb'))
        self.G_losses = datas["G_loss"]
        self.D_losses = datas["D_loss"]
        print("* {} loaded.\n* {} loaded.".format(self.resume, self.history))

    def save(self, epoch):
        checkpoint = {}
        checkpoint['iters'] = self.iters
        checkpoint['epoch'] = epoch
        checkpoint['G'] = self.G.state_dict()
        checkpoint['D'] = self.D.state_dict()
        datas = {}
        datas["G_loss"] = self.G_losses
        datas["D_loss"] = self.D_losses
        resume = os.path.join(self.model_dir, "model_{}.pth".format(str(epoch).zfill(4)))
        torch.save(checkpoint, resume)
        # torch.save(checkpoint, self.resume)
        history = os.path.join(self.model_dir, "history_{}.pkl".format(str(epoch).zfill(4)))
        pkl.dump(datas, open(history, 'wb'))
        # pkl.dump(datas, open(self.history, 'wb'))
        self.G_losses = []
        self.D_losses = []  # 确保没一个history文件只保存当前epoch的loss
        print("* {} saved.\n* {} saved.".format(resume, history))

    def train(self):
        # 加载存在的权重和loss
        if os.path.exists(self.resume) and os.path.exists(self.history):
            self.load()
        else:
            self.G.apply(weights_init)
            self.D.apply(weights_init)
            self.start_epoch = 0
            self.iters = 0
            self.G_losses = []
            self.D_losses = []
        self.G.to(self.device)
        self.D.to(self.device)
        # Setup Adam optimizers for both G and D
        optimizerD = optim.Adam(self.D.parameters(), lr=self.lrG, betas=(self.beta1, self.beta2))
        optimizerG = optim.Adam(self.G.parameters(), lr=self.lrD, betas=(self.beta1, self.beta2))

        # Initialize BCELoss function
        criterion = nn.BCELoss()

        # Create batch of latent vectors that we will use to visualize
        #  the progression of the generator
        self.sample_z_ = torch.randn(self.batch_size, self.nz, 1, 1, device=self.device)

        # Establish convention for real and fake labels during training
        real_label = 1
        fake_label = 0

        dataloader = torch.utils.data.DataLoader(
            dataset=dset.ImageFolder(root=self.dataroot,
                                     transform=transforms.Compose([
                                         transforms.Resize(self.image_size),
                                         transforms.CenterCrop(self.image_size),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                     ])),
            batch_size=self.batch_size, shuffle=True)

        print("Starting Training Loop...")
        # For each epoch
        for epoch in range(self.start_epoch, self.epochs):
            # For each batch in the dataloader
            for i, data in enumerate(dataloader, 0):
                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                ## Train with all-real batch
                self.D.zero_grad()
                # Format batch
                real_cpu = data[0].to(self.device)
                b_size = real_cpu.size(0)
                label = torch.full((b_size,), real_label, device=self.device)
                # Forward pass real batch through D
                # print(self.D(real_cpu).size())
                output = self.D(real_cpu).view(-1)
                # Calculate loss on all-real batch
                # print(output.size(), label.size())
                errD_real = criterion(output, label)
                # Calculate gradients for D in backward pass
                errD_real.backward()
                D_x = output.mean().item()

                ## Train with all-fake batch
                # Generate batch of latent vectors
                noise = torch.randn(b_size, self.nz, 1, 1, device=self.device)
                # Generate fake image batch with G
                fake = self.G(noise)
                label.fill_(fake_label)
                # Classify all fake batch with D
                output = self.D(fake.detach()).view(-1)
                # Calculate D's loss on the all-fake batch
                errD_fake = criterion(output, label)
                # Calculate the gradients for this batch
                errD_fake.backward()
                D_G_z1 = output.mean().item()
                # Add the gradients from the all-real and all-fake batches
                errD = errD_real + errD_fake
                # Update D
                optimizerD.step()

                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                self.G.zero_grad()
                label.fill_(real_label)  # fake labels are real for generator cost
                # Since we just updated D, perform another forward pass of all-fake batch through D
                output = self.D(fake).view(-1)
                # Calculate G's loss based on this output
                errG = criterion(output, label)
                ####################################################################
                # D_fake = self.D(fake)
                # errG = -torch.mean(D_fake)
                ####################################################################
                # Calculate gradients for G
                errG.backward()
                D_G_z2 = output.mean().item()
                # Update G
                optimizerG.step()

                # Output training stats
                if i % 50 == 0:
                    print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                          % (epoch, self.epochs, i, len(dataloader),
                             errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

                # Save Losses for plotting later
                self.G_losses.append(errG.item())
                self.D_losses.append(errD.item())
                self.iters += 1
            self.save(epoch)
            self.visualize_results(epoch, fix=True)  # 利用相同的向量生成图片，利于比较不同epoch的结果
            # loss_plot(self.history)
            try:
                loss_plot_multi_pkls(self.model_dir)
            except Exception:
                pass

    def visualize_results(self, epoch, fix=True):
        self.G.eval()
        tot_num_samples = min(64, self.batch_size)
        image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))

        if fix:
            """ fixed noise """
            samples = self.G(self.sample_z_)
        else:
            """ random noise """
            sample_z_ = torch.rand((self.batch_size, self.nz, 1, 1))
            if self.gpu_mode:
                sample_z_ = sample_z_.cuda()

            samples = self.G(sample_z_)

        if self.gpu_mode:
            samples = samples.cpu().data.numpy().transpose(0, 2, 3, 1)
        else:
            samples = samples.data.numpy().transpose(0, 2, 3, 1)

        samples = (samples + 1) / 2
        utils.save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                          os.path.join(self.result_dir, 'epoch{}.png'.format(str(epoch).zfill(4))))

    def save_image(self, tensor, save_path):
        self.G.eval()
        if self.gpu_mode:
            samples = tensor.cpu().data.numpy().transpose(0, 2, 3, 1)
        else:
            samples = tensor.data.numpy().transpose(0, 2, 3, 1)

        samples = (samples + 1) / 2
        utils.save_images(samples[:1, :, :, :], [1, 1], save_path)

    # Completion
    def complete(self, cfgs=None):
        # 加载存在的权重
        if os.path.exists(self.resume):
            self.load()
            print('model loaded successfully.')
        else:
            raise RuntimeError('no model to load')
        source_imagedir = os.path.join(self.output_dir, self.dataset, self.model_name, 'results',  "source_images")
        masked_imagedir = os.path.join(self.output_dir, self.dataset, self.model_name, 'results', "masked_images")
        impainted_imagedir = os.path.join(self.output_dir, self.dataset, self.model_name, 'results', "impainted_images")
        os.makedirs(source_imagedir, exist_ok=True)
        os.makedirs(masked_imagedir, exist_ok=True)
        os.makedirs(impainted_imagedir, exist_ok=True)
        transform_img = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        transform_mask = transforms.Compose([
            transforms.ToTensor()
        ])
        criteria = nn.BCELoss()
        image_dir = cfgs.test_image_dir
        mask_dir = cfgs.test_mask_dir
        # 需要修补的图片与对应的mask
        images = os.listdir(image_dir)
        aligned_images = [os.path.join(image_dir, image) for image in images]  # 源图片的路径
        mask_images = [os.path.join(mask_dir, os.path.splitext(image)[0] + '.png') for image in images]  # mask的路径
        real_label = 1
        fake_label = 0
        label = torch.full((1,), real_label, device=self.device)
        for i, image in enumerate(images):
            print(aligned_images[i])
            batch_images = [transform_img(pil_loader(aligned_images[i]))]
            batch_images = torch.stack(batch_images).to(self.device)
            batch_masks = [transform_mask(read_mask(mask_images[i], self.image_size, self.image_size))]
            batch_masks = torch.stack(batch_masks).to(self.device)
            z_hat = torch.rand(size=[1, self.nz, 1, 1], dtype=torch.float32, requires_grad=True,
                               device=self.device)
            masked_batch_images = torch.mul(batch_images, batch_masks).to(self.device)
            # z_hat.data.mul_(2.0).sub_(1.0)
            opt = optim.Adam([z_hat], lr=cfgs.lr)
            v = torch.tensor(0, dtype=torch.float32, device=self.device)
            m = torch.tensor(0, dtype=torch.float32, device=self.device)
            for iteration in range(cfgs.num_iters):
                # 对每一个batch的图像分别迭代impainting
                if z_hat.grad is not None:
                    z_hat.grad.data.zero_()
                self.G.zero_grad()
                self.D.zero_grad()
                self.G.to(self.device)
                self.D.to(self.device)
                batch_images_g = self.G(z_hat)
                batch_images_g_masked = torch.mul(batch_images_g, batch_masks)
                impainting_images = torch.mul(batch_images_g, (1 - batch_masks)) + masked_batch_images
                if iteration % 5000 == 0:
                    # 保存impainting 图片结果
                    print("\nsaving impainted images for iteration:{}".format(iteration))
                    # save_tensor_images(batch_images_g.detach(),
                    #                    os.path.join(impainted_imagedir, os.path.splitext(image)[0]+"_iteration_{}.png".format(iteration)),
                    #                    nrow=1)
                    save_path = os.path.join(impainted_imagedir, os.path.splitext(image)[0]+"_iteration_{}.png".format(iteration))
                    self.save_image(impainting_images, save_path)


                    # output = self.D(impainting_images.detach()).view(-1)
                    # label.fill_(fake_label)
                    # errD_fake = nn.BCELoss()(output, label)
                    # errD_fake.backward()
                    #
                    # label.fill_(real_label)
                    # output = self.D(impainting_images).view(-1)
                    # # Calculate G's loss based on this output
                    # errG = nn.BCELoss()(output, label)
                    # errG.backward()


                    loss_context = torch.norm(
                        (masked_batch_images - batch_images_g_masked), p=1)
                    dis_output = self.D(impainting_images)
                    batch_labels = torch.full((1,), 1, device=self.device)
                    loss_perceptual = criteria(dis_output, batch_labels)

                    total_loss = loss_context + cfgs.lamd * loss_perceptual
                    print("iteration : {:4} , context_loss:{:.4f},percptual_loss:{:4f}".format(iteration,
                                                                                                 loss_context,
                                                                                                 loss_perceptual))
                    total_loss.backward()
                    opt.step()
                    g = z_hat.grad
                    if g is None:
                        print("g is None")
                        continue
                    vpre = v.clone()
                    mpre = m.clone()
                    m = 0.99*mpre+(1-0.99)*g
                    v = 0.999*vpre+(1-0.999)*(g*g)
                    m_hat = m/(1-0.99**(iteration+1))
                    v_hat = v/(1-0.999**(iteration+1))
                    z_hat.data.sub_(m_hat/(torch.sqrt(v_hat)+1e-8))
                    z_hat.data = torch.clamp(z_hat.data, min=-1.0,max=1.0).to(self.device)

if __name__ == '__main__':
    pass
