from torch import nn

CH = 8

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.conv = nn.Sequential(
            nn.ConvTranspose2d(100, CH * 8, 4, 1, 0, 0, bias=False),
            nn.BatchNorm2d(CH * 8),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(CH * 8, CH * 4, 5, 2, 2, 1, bias=False),
            nn.BatchNorm2d(CH * 4),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(CH * 4, CH * 2, 5, 2, 2, 1, bias=False),
            nn.BatchNorm2d(CH * 2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(CH * 2, CH, 5, 2, 2, 1, bias=False),
            nn.BatchNorm2d(CH),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(CH, 1, 5, 2, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input_):
        return self.conv(input_.view(-1, 100, 1, 1))


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, CH, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(CH, CH * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(CH * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(CH * 2, CH * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(CH * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(CH * 4, CH * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(CH * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(CH * 8, 1, 4, 1, 0, bias=False)
        )

    def forward(self, input_):
        return self.conv(input_).view(-1)
