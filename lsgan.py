import argparse
import os

import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from torch.utils import data
from torchvision import datasets, transforms
from torchvision.utils import save_image

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', type=str, default='data')
parser.add_argument('--image-dir', '-d', type=str, default='lsgan')
parser.add_argument('--batch-size', '-bs', type=int, default=128)
parser.add_argument('--learning-rate', '-lr', type=float, default=2e-4)
parser.add_argument('--channels', type=int, default=8)
parser.add_argument('--num-workers', type=int, default=0)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--log-interval', '-li', type=int, default=10)
parser.add_argument('--no-cuda', action='store_true')
args = parser.parse_args()
print(args)

os.makedirs(args.image_dir, exist_ok=True)

ch = args.channels

use_cuda = torch.cuda.is_available() and not args.no_cuda
if use_cuda:
    print("Use CUDA.")

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.conv = nn.Sequential(
            nn.ConvTranspose2d(100, ch * 8, 4, 1, 0, 0, bias=False),
            nn.BatchNorm2d(ch * 8),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(ch * 8, ch * 4, 5, 2, 2, 1, bias=False),
            nn.BatchNorm2d(ch * 4),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(ch * 4, ch * 2, 5, 2, 2, 1, bias=False),
            nn.BatchNorm2d(ch * 2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(ch * 2, ch, 5, 2, 2, 1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(ch, 1, 5, 2, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input_):
        return self.conv(input_.view(-1, 100, 1, 1))


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, ch, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ch, ch * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ch * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ch * 2, ch * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ch * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ch * 4, ch * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ch * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ch * 8, 1, 4, 1, 0, bias=False)
        )

    def forward(self, input_):
        return self.conv(input_).view(-1, 1)


transform = transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor(),
    # transforms.Normalize((0.13066047740239478,), (0.3081078087569972,))
])

train_dataloader = data.DataLoader(datasets.MNIST(args.data_dir,
                                                  train=True,
                                                  transform=transform,
                                                  download=True),
                                   batch_size=args.batch_size,
                                   shuffle=True,
                                   num_workers=args.num_workers)


d = Discriminator()
g = Generator()

if use_cuda:
    d.cuda()
    g.cuda()

optimizer_d = torch.optim.Adam(d.parameters(),
                               lr=args.learning_rate,
                               betas=(0.5, 0.999))
optimizer_g = torch.optim.Adam(g.parameters(),
                               lr=args.learning_rate,
                               betas=(0.5, 0.999))

losses_d = []
losses_g = []

def train(epoch):

    for batch_index, (x, _) in enumerate(train_dataloader):
        # train discriminator
        x = Variable(x)
        z = Variable(torch.randn(len(x), 100))

        if use_cuda:
            x = x.cuda()
            z = z.cuda()

        loss_d = 0.5 * (d(x) - 1).pow(2).mean() + 0.5 * d(g(z).detach()).pow(2).mean()

        optimizer_d.zero_grad()
        loss_d.backward()
        optimizer_d.step()

        # train generator
        z = Variable(torch.randn(len(x), 100))

        if use_cuda:
            z = z.cuda()

        fake_x = g(z)
        loss_g = 0.5 * (d(g(z)) - 1).pow(2).mean()

        optimizer_g.zero_grad()
        loss_g.backward()
        optimizer_g.step()

        # log
        losses_d.append(float(loss_d.data))
        losses_g.append(float(loss_g.data))

        if batch_index % args.log_interval == 0:
            print('{}: {}, {}: {}, {}: {:.4f}, {}: {:.4f}.'.format(
                'Train epoch', epoch,
                'batch index', batch_index,
                'discriminator loss:', float(loss_d.data),
                'generator loss', float(loss_g.data)))
            plot_losses()

    # plot samples
    plot_samples(epoch)

def plot_losses():
    fig, ax = plt.subplots(nrows=2)
    ax[0].plot(losses_d, c='r', label='discriminator loss')
    ax[0].legend()
    ax[1].plot(losses_g, c='b', label='generator loss')
    ax[1].legend()
    fig.savefig(os.path.join(args.image_dir, 'losses.png'))
    plt.close(fig)

def plot_samples(epoch):
    g.eval()

    rand_z = Variable(torch.randn(16 * 16, 100), volatile=True)
    if use_cuda:
        rand_z = rand_z.cuda()

    filename = os.path.join(args.image_dir,
                           'samples_epoch_{}.jpg'.format(epoch,))
    save_image(g(rand_z).data, filename, normalize=True, nrow=16)

    g.train()


for epoch in range(args.epochs):
    train(epoch)