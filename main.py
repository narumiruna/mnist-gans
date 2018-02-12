import argparse
from solver import GAN, WGAN, WGANGP, LSGAN

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('type', type=str, help='gan, wgan, wgangp, lsgan.')

    # optimizer
    parser.add_argument('--g-lr', type=float, default=2e-4)
    parser.add_argument('--d-lr', type=float, default=2e-4)
    parser.add_argument('--g-beta1', type=float, default=0.5)
    parser.add_argument('--g-beta2', type=float, default=0.999)
    parser.add_argument('--d-beta1', type=float, default=0.5)
    parser.add_argument('--d-beta2', type=float, default=0.999)

    # dataset
    parser.add_argument('--root', type=str, default='data')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-workers', type=int, default=0)

    # training
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--d-steps', type=int, default=1)
    parser.add_argument('--log-interval', type=int, default=10)
    parser.add_argument('--clip-param', type=float, default=0.1, help='clipping parameter for wgan')
    parser.add_argument('--penalty', type=int, default=10, help='penalty coefficient for wgan gp')
    args = parser.parse_args()
    print(args)

    if args.type == 'gan':
        GAN(args).solve()
    elif args.type == 'wgan':
        WGAN(args).solve()
    elif args.type == 'wgangp':
        WGANGP(args).solve()
    elif args.type == 'lsgan':
        LSGAN(args).solve()
    else:
        raise Exception('No such type.')

if __name__ == '__main__':
    main()