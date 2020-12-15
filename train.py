import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from models import *
from dataset import prepare_data, Dataset
from utils import *
from matplotlib import pyplot as plt
from torchvision.transforms import ToPILImage
import torchvision.transforms as transforms

from imageio import imwrite
import PIL
from PIL import Image
from PIL import ImageFilter
from datetime import datetime

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="FBDN")
parser.add_argument("--preprocess", type=bool, default=False, help='run prepare_data or not')
parser.add_argument("--batchSize", type=int, default=16, help="Training batch size")
parser.add_argument("--num_of_layers", type=int, default=16, help="Number of total layers")
parser.add_argument("--epochs", type=int, default=60, help="Number of training epochs")
parser.add_argument("--milestone", type=int, default=30, help="When to decay learning rate; should be less than epochs")
parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
parser.add_argument("--outf", type=str, default="logs", help='path of log files')
parser.add_argument("--mode", type=str, default="S", help='with known noise level (S) or blind training (B)')
parser.add_argument("--noiseL", type=float, default=25, help='noise level; ignored when mode=B')
parser.add_argument("--val_noiseL", type=float, default=25, help='noise level used on validation set')
parser.add_argument("--patchsize", type=int, default=96, help='patch size of image')
opt = parser.parse_args()


def main():
    # Load dataset
    print('Loading dataset ...\n')
    dataset_train = Dataset(train=True)
    dataset_val = Dataset(train=False)
    loader_train = DataLoader(dataset=dataset_train, num_workers=4, batch_size=opt.batchSize, shuffle=True) # if debug, num_workers=0 not 4
    print("# of training samples: %d\n" % int(len(dataset_train)))

    # toPILimg = transforms.Compose([transforms.ToPILImage()])
    toTensor = transforms.Compose([transforms.ToTensor()])
    toPILImg = transforms.ToPILImage()

    # Build model
    net = FeaturesBlockDualNet(channels=1)
    net.apply(weights_init_kaiming)
    criterion = nn.L1Loss()
    model = net.cuda()
    criterion.cuda()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    # training
    writer = SummaryWriter(opt.outf)
    step = 0
    noiseL_B=[0,55] # ingnored when opt.mode=='S '

    start_time = datetime.now()
    print('Training Start!!')
    print(start_time)

    for epoch in range(opt.epochs):
        if epoch < opt.milestone:
            current_lr = opt.lr # 1e-3
        else:
            current_lr = opt.lr / 10. # 1e-4

        # set learning rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
        print('learning rate %f' % current_lr)

        # train
        for i, data in enumerate(loader_train, 0):
            # training step
            model.train()
            model.zero_grad()
            optimizer.zero_grad()
            img_train = data

            if opt.mode == 'S':
                noise = torch.FloatTensor(img_train.size()).normal_(mean=0, std=opt.noiseL/255.)
            if opt.mode == 'B':
                noise = torch.zeros(img_train.size())
                stdN = np.random.uniform(noiseL_B[0], noiseL_B[1], size=noise.size()[0])
                for n in range(noise.size()[0]):
                    sizeN = noise[0,:,:,:].size()
                    noise[n,:,:,:] = torch.FloatTensor(sizeN).normal_(mean=0, std=stdN[n]/255.)
            imgn_train = img_train + noise

            img_train, imgn_train = Variable(img_train.cuda()), Variable(imgn_train.cuda())
            noise = Variable(noise.cuda())

            out_train = model(imgn_train)

            loss = criterion(out_train, img_train)

            loss.backward()
            optimizer.step()

            # results
            model.eval()
            out_train, out_noise = model(imgn_train)
            out_train = torch.clamp(out_train, 0., 1.)
            psnr_train = batch_PSNR(out_train, img_train, 1.)
            # i%100 == 0 -> each 100 epochs, print loss and psnr.
            if i % 500 == 0 :
                print("[epoch %d][%d/%d] loss: %.4f PSNR_train: %.4f" %
                    (epoch+1, i+1, len(loader_train), loss.item(), psnr_train))
            # if you are using older version of PyTorch, you may need to change loss.item() to loss.data[0]
            if step % 10 == 0:
                # Log the scalar values
                writer.add_scalar('loss', loss.item(), step)
                writer.add_scalar('PSNR on training data', psnr_train, step)
            step += 1

        # the end of each epoch
        model.eval()

        # validate
        psnr_val = 0
        for k in range(len(dataset_val)):
            img_val = torch.unsqueeze(dataset_val[k], 0)
            noise = torch.FloatTensor(img_val.size()).normal_(mean=0, std=opt.val_noiseL/255.)
            imgn_val = img_val + noise

            with torch.no_grad():
                img_val, imgn_val = Variable(img_val.cuda()), Variable(imgn_val.cuda())
                out_val = model(imgn_val)
                out_val = torch.clamp(out_val, 0., 1.)
            psnr_val += batch_PSNR(out_val, img_val, 1.)

        psnr_val /= len(dataset_val)
        print("[epoch %d] PSNR_val: %.4f\n" % (epoch+1, psnr_val))
        midtime = datetime.now() - start_time
        print(midtime)
        writer.add_scalar('PSNR on validation data', psnr_val, epoch)

        # log the images
        # out_train = torch.clamp(imgn_train-model(imgn_train), 0., 1.)
        Img = utils.make_grid(img_val.data, nrow=8, normalize=True, scale_each=True)
        Imgn = utils.make_grid(imgn_val.data, nrow=8, normalize=True, scale_each=True)
        Irecon = utils.make_grid(out_val.data, nrow=8, normalize=True, scale_each=True)
        # writer.add_image('clean image', Img, epoch)
        # writer.add_image('noisy image', Imgn, epoch)
        # writer.add_image('reconstructed image', Irecon, epoch)

        # Compare clean, noisy, denoising image
        fig = plt.figure()
        fig.suptitle(epoch + 1)
        rows = 1
        cols = 3

        ax1 = fig.add_subplot(rows, cols, 1)
        # tensor는 cuda에서 처리하지 못하기 때문에 .cpu()로 보내줌.
        ax1.imshow(np.transpose(Img.cpu(), (1,2,0)), cmap="gray")
        ax1.set_title('clean image')

        ax2 = fig.add_subplot(rows, cols, 2)
        ax2.imshow(np.transpose(Imgn.cpu(), (1,2,0)), cmap="gray")
        ax2.set_title('noisy image')

        ax3 = fig.add_subplot(rows, cols, 4)
        ax3.imshow(np.transpose(Irecon.cpu(), (1, 2, 0)), cmap="gray")
        ax3.set_title('denoising image')

        # plt.savefig('./fig_result/epoch_{:d}.png'.format(epoch + 1))
        plt.show()


        # save model
        # torch.save(model.state_dict(), os.path.join(opt.outf, 'UsingIQRnNoiseblock_Dualnet_25.pth'))
        # nl(noise level)25 => 30.6000 nl15 => 32.8000 nl50 => 27.3000
        if psnr_val >= 30.6000:
            torch.save(model.state_dict(), os.path.join(opt.outf, 'FBDNet_' + str(round(psnr_val, 4)) + '.pth'))

    end_time = datetime.now()
    print('Training Finished!!')
    print(end_time)

if __name__ == "__main__":
    if opt.preprocess:
        # prepare_data에서 data를 patch_size대로 나누어주고 다 준비해서 .h5 파일로 만들어주서 main()의 dataset에 집어넣는다.
        if opt.mode == 'S':
            prepare_data(data_path='data', patch_size=opt.patchsize, stride=96, aug_times=1)
        if opt.mode == 'B':
            prepare_data(data_path='data', patch_size=50, stride=10, aug_times=2)
    main()
