import imageio
import glob
import argparse
from os.path import join

parser = argparse.ArgumentParser()
parser.add_argument('dir', type=str)
args = parser.parse_args()

paths = sorted(glob.glob(join(args.dir, 'sample_*.jpg')))

images = []
for path in paths:
    images.append(imageio.imread(path))

imageio.mimsave(join(args.dir, 'sample.gif'), images, fps=1)

