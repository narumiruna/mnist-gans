import argparse
import os

import torch
from torch import nn
from torch.autograd import Variable
from torch.utils import data
from torchvision import datasets, transforms
from torchvision.utils import save_image

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--data-dir', type=str, default='data')
parser.add_argument('--image-dir', type=str, default='images')
parser.add_argument('--batch-size', '-bs', type=int, default=128)
parser.add_argument('--learning-rate', '-lr', type=float, default=1e-5)
parser.add_argument('--no-cuda', action='store_true')
parser.add_argument('--log-interval', type=int, default=100)
args = parser.parse_args()

use_cuda = torch.cuda.is_available() and not args.no_cuda

os.makedirs(args.image_dir, exist_ok=True)

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 1000, bias=False),
            nn.Linear(1000, 500, bias=False),
            nn.Linear(500, 250, bias=False),
            nn.Linear(250, 30, bias=False),
        )

        self.decoder = nn.Sequential(
            nn.Linear(30, 250, bias=False),
            nn.Linear(250, 500, bias=False),
            nn.Linear(500, 1000, bias=False),
            nn.Linear(1000, 784, bias=False),
        )

    def forward(self, input_):
        return self.decode(self.encode(input_))

    def encode(self, input_):
        return self.encoder(input_.view(-1, 784))

    def decode(self, code):
        return self.decoder(code).view(-1, 1, 28, 28)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.13066047740239478,), (0.3081078087569972,))
])

train_dataloader = data.DataLoader(datasets.MNIST(args.data_dir,
                                                  train=True,
                                                  transform=transform,
                                                  download=True),
                                   batch_size=args.batch_size,
                                   shuffle=True)

autoencoder = Autoencoder()

if use_cuda:
    autoencoder.cuda()

optimizer = torch.optim.Adam(autoencoder.parameters(),
                             lr=args.learning_rate)

losses = []

def train(epoch):
    for batch_index, (train_x, _) in enumerate(train_dataloader):
        optimizer.zero_grad()

        train_x = Variable(train_x)

        if use_cuda:
            train_x = train_x.cuda()

        loss = (train_x - autoencoder(train_x)).pow(2).mean()

        loss.backward()
        optimizer.step()

        losses.append(float(loss.data))

        if batch_index % args.log_interval == 0:
            print('{}: {}, {}: {}, {}: {}, {}: {}.'.format('Train epoch', epoch,
                                                           'batch index', batch_index,
                                                           'step', len(losses),
                                                           'loss', float(loss.data)))
    plot(epoch, train_x)

def plot(epoch, train_x):
    train_x.volatile = True
    if use_cuda:
        train_x = train_x.cuda()
    filename = os.path.join(args.image_dir,
                            'autoencoder_epoch_{}.jpg'.format(epoch))
    save_image(autoencoder(train_x).data, filename, normalize=True)

for epoch in range(args.epochs):
    train(epoch)
