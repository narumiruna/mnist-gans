
import os

import torch
from torch import nn
from torch.autograd import Variable, grad
from torch.utils import data
from torchvision import datasets, transforms
from torchvision.utils import save_image


from model import Discriminator, Generator


class Solver:
    def __init__(self, args):

        self.type = args.type
        self.cuda = torch.cuda.is_available()

        # optimizer args
        self.g_lr = args.g_lr
        self.d_lr = args.d_lr
        self.g_beta1 = args.g_beta1
        self.g_beta2 = args.g_beta2
        self.d_beta1 = args.d_beta1
        self.d_beta2 = args.d_beta2

        # dataset
        self.root = args.root
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers

        # training
        self.epochs = args.epochs
        self.d_steps = args.d_steps
        self.log_interval = args.log_interval
        self.clip_param = args.clip_param
        self.penalty = args.penalty

        # build
        self.build_model()
        self.build_dataloader()

        # logging
        self.g_loss = []
        self.d_loss = []

        # make sample
        self.z = Variable(torch.randn(16 * 16, 100), volatile=True)
        if self.cuda:
            self.z = self.z.cuda()

        # makedir
        os.makedirs(self.type, exist_ok=True)

        self.picklefile = os.path.join(self.type, 'model.pickle')
        if os.path.exists(self.picklefile):
            self.load()

    def build_dataloader(self):
        self.transform = transforms.Compose([
            transforms.Resize(64),
            transforms.ToTensor()
        ])

        self.dataset = datasets.MNIST(self.root,
                                      transform=self.transform,
                                      download=True)

        self.dataloader = data.DataLoader(self.dataset,
                                          batch_size=self.batch_size,
                                          shuffle=True,
                                          num_workers=self.num_workers)

    def solve(self):
        for epoch in range(self.epochs):
            self.train(epoch)
            self.make_sample(epoch)
            self.save()

    def print_log(self, epoch, batch_index):
        num_batches = len(self.dataset) // self.batch_size

        print('Epoch: {}/{},'.format(epoch + 1, self.epochs),
              'iter: {}/{},'.format(batch_index + 1, num_batches),
              'g_loss: {:.4f}, d_loss: {:.4f}'.format(self.g_loss[-1], self.d_loss[-1]))

    def make_sample(self, epoch):
        self.g.eval()

        filename = os.path.join(self.type, 'sample_{}.jpg'.format(epoch + 1))
        save_image(self.g(self.z).data, filename, nrow=16, normalize=True)

        self.g.train()

    def save(self):

        d = {
            'g_dict': self.g.state_dict(),
            'd_dict': self.d.state_dict(),
            'g_optim_dict': self.g_optim.state_dict(),
            'd_optim_dict': self.d_optim.state_dict()
        }

        torch.save(d, self.picklefile)

        print('save')

    def load(self):
        d = torch.load(self.picklefile)

        self.g.load_state_dict(d['g_dict'])
        self.d.load_state_dict(d['d_dict'])
        self.g_optim.load_state_dict(d['g_optim_dict'])
        self.d_optim.load_state_dict(d['d_optim_dict'])

        print('load')


class GAN(Solver):
    def __init__(self, args):
        super(GAN, self).__init__(args)

    def build_model(self):
        self.g = Generator()
        self.d = nn.Sequential(
            Discriminator(),
            nn.Sigmoid()
        )

        if self.cuda:
            self.g.cuda()
            self.d.cuda()

        self.g_optim = torch.optim.Adam(self.g.parameters(),
                                        lr=self.g_lr,
                                        betas=(self.g_beta1, self.g_beta2))
        self.d_optim = torch.optim.Adam(self.d.parameters(),
                                        lr=self.d_lr,
                                        betas=(self.d_beta1, self.d_beta2))

    def train(self, epoch):
        bce = nn.BCELoss()

        for batch_index, (x, _) in enumerate(self.dataloader):
            x = Variable(x)
            z = Variable(torch.randn(len(x), 100))
            ones = Variable(torch.ones(len(x)))
            zeros = Variable(torch.zeros(len(x)))

            if self.cuda:
                x = x.cuda()
                z = z.cuda()
                ones = ones.cuda()
                zeros = zeros.cuda()

            x_fake = self.g(z).detach()
            d_loss = bce(self.d(x), ones) + bce(self.d(x_fake), zeros)

            self.d_optim.zero_grad()
            d_loss.backward()
            self.d_optim.step()

            # log
            self.d_loss.append(float(d_loss.data))

            if (batch_index + 1) % self.d_steps == 0:
                z = Variable(torch.randn(len(x), 100))

                if self.cuda:
                    z = z.cuda()

                x_fake = self.g(z)
                g_loss = bce(self.d(x_fake), ones)

                self.g_optim.zero_grad()
                g_loss.backward()
                self.g_optim.step()

                # log
                self.g_loss.append(float(g_loss.data))

            if (batch_index + 1) % self.log_interval == 0:
                self.print_log(epoch, batch_index)


class WGAN(Solver):
    def __init__(self, args):
        super(WGAN, self).__init__(args)

    def build_model(self):
        self.g = Generator()
        self.d = Discriminator()

        if self.cuda:
            self.g.cuda()
            self.d.cuda()

        self.g_optim = torch.optim.RMSprop(self.g.parameters(),
                                           lr=self.g_lr)
        self.d_optim = torch.optim.RMSprop(self.d.parameters(),
                                           lr=self.d_lr)

    def train(self, epoch):
        for batch_index, (x, _) in enumerate(self.dataloader):
            x = Variable(x)
            z = Variable(torch.randn(len(x), 100))

            if self.cuda:
                x = x.cuda()
                z = z.cuda()

            x_fake = self.g(z).detach()
            d_loss = -self.d(x).mean() + self.d(x_fake).mean()

            self.d_optim.zero_grad()
            d_loss.backward()
            self.d_optim.step()

            for p in self.d.parameters():
                p.data.clamp_(-self.clip_param, self.clip_param)

            # log
            self.d_loss.append(float(d_loss.data))

            if (batch_index + 1) % self.d_steps == 0:
                z = Variable(torch.randn(len(x), 100))

                if self.cuda:
                    z = z.cuda()

                x_fake = self.g(z)
                g_loss = -self.d(x_fake).mean()

                self.g_optim.zero_grad()
                g_loss.backward()
                self.g_optim.step()

                # log
                self.g_loss.append(float(g_loss.data))

            if (batch_index + 1) % self.log_interval == 0:
                self.print_log(epoch, batch_index)


class LSGAN(Solver):
    def __init__(self, args):
        super(LSGAN, self).__init__(args)

    def build_model(self):
        self.g = Generator()
        self.d = Discriminator()

        if self.cuda:
            self.g.cuda()
            self.d.cuda()

        self.g_optim = torch.optim.Adam(self.g.parameters(),
                                        lr=self.g_lr,
                                        betas=(self.g_beta1, self.g_beta2))
        self.d_optim = torch.optim.Adam(self.d.parameters(),
                                        lr=self.d_lr,
                                        betas=(self.d_beta1, self.d_beta2))

    def train(self, epoch):
        for batch_index, (x, _) in enumerate(self.dataloader):
            x = Variable(x)
            z = Variable(torch.randn(len(x), 100))

            if self.cuda:
                x = x.cuda()
                z = z.cuda()

            x_fake = self.g(z).detach()
            d_loss = 0.5 * (self.d(x) - 1).pow(2).mean() + \
                0.5 * self.d(x_fake).pow(2).mean()

            self.d_optim.zero_grad()
            d_loss.backward()
            self.d_optim.step()

            # log
            self.d_loss.append(float(d_loss.data))

            if (batch_index + 1) % self.d_steps == 0:
                z = Variable(torch.randn(len(x), 100))

                if self.cuda:
                    z = z.cuda()

                x_fake = self.g(z)
                g_loss = 0.5 * (self.d(x_fake) - 1).pow(2).mean()

                self.g_optim.zero_grad()
                g_loss.backward()
                self.g_optim.step()

                # log
                self.g_loss.append(float(g_loss.data))

            if (batch_index + 1) % self.log_interval == 0:
                self.print_log(epoch, batch_index)


class WGANGP(Solver):
    def __init__(self, args):
        super(WGANGP, self).__init__(args)

    def build_model(self):
        self.g = Generator()
        self.d = Discriminator()

        if self.cuda:
            self.g.cuda()
            self.d.cuda()

        self.g_optim = torch.optim.Adam(self.g.parameters(),
                                        lr=self.g_lr,
                                        betas=(self.g_beta1, self.g_beta2))
        self.d_optim = torch.optim.Adam(self.d.parameters(),
                                        lr=self.d_lr,
                                        betas=(self.d_beta1, self.d_beta2))

    def gradient_penalty(self, x, x_fake):
        epsilon = Variable(torch.rand(len(x), 1, 1, 1))
        grad_outputs = Variable(torch.ones(len(x)))

        if self.cuda:
            epsilon = epsilon.cuda()
            grad_outputs = grad_outputs.cuda()

        interpolates = epsilon * x + (1 - epsilon) * x_fake
        interpolates.requires_grad = True

        gradients = grad(self.d(interpolates),
                         interpolates,
                         grad_outputs=grad_outputs,
                         create_graph=True)[0]

        gradient_penalty = (gradients.view(
            len(x), -1).norm(2, dim=1) - 1).pow(2)

        return gradient_penalty

    def train(self, epoch):
        for batch_index, (x, _) in enumerate(self.dataloader):
            x = Variable(x)
            z = Variable(torch.randn(len(x), 100))

            if self.cuda:
                x = x.cuda()
                z = z.cuda()

            x_fake = self.g(z).detach()
            d_loss = self.d(x_fake).mean() - self.d(x).mean() + \
                self.penalty * self.gradient_penalty(x, x_fake).mean()

            self.d_optim.zero_grad()
            d_loss.backward()
            self.d_optim.step()

            # log
            self.d_loss.append(float(d_loss.data))

            if (batch_index + 1) % self.d_steps == 0:
                z = Variable(torch.randn(len(x), 100))

                if self.cuda:
                    z = z.cuda()

                x_fake = self.g(z)
                g_loss = - self.d(x_fake).mean()

                self.g_optim.zero_grad()
                g_loss.backward()
                self.g_optim.step()

                # log
                self.g_loss.append(float(g_loss.data))

            if (batch_index + 1) % self.log_interval == 0:
                self.print_log(epoch, batch_index)
