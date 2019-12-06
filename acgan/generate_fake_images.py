import argparse
import os
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from utils import weights_init, compute_acc
from network import _netG, _netD, _netD_CIFAR10, _netG_CIFAR10
from folder import ImageFolder
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--nz', type=int, default=110, help='size of the latent z vector')
parser.add_argument('--outf', default='/home/ubuntu/xray_gan/data/generated_images/', help='folder to output images and model checkpoints')
parser.add_argument('--nsamples', type=int, default=1, help='number of images to sample')
parser.add_argument('--batchSize', type=int, default=30000, help='input batch size')
parser.add_argument('--gpu_id', type=int, default=0, help='The ID of the specified GPU')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--num_classes', type=int, default=2, help='Number of classes for AC-GAN')

PERCENT_YOUNG = 0.4
PERCENT_OLD = 0.4
PERCENT_MIDDLE = 0.2
assert PERCENT_YOUNG + PERCENT_OLD + PERCENT_MIDDLE == 1
opt = parser.parse_args()
assert opt.batchSize % 10 ==0

MAX_AGE = 120.0
if opt.ngpu == 1:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu_id)

    
if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")
nz = int(opt.nz)
ngf = int(opt.ngf)
num_classes = int(opt.num_classes)
ngpu = int(opt.ngpu)    
    
netG = _netG_CIFAR10(ngpu, nz)
epoch = 0

netG.load_state_dict(torch.load('./netG_epoch_%d.pth' % (epoch)))
netG.eval()

eval_noise = torch.FloatTensor(1, nz, 1, 1).normal_(0, 1)
if opt.cuda:
    netG.cuda()
    eval_noise = eval_noise.cuda()
    
# noise = Variable(noise)
eval_noise = Variable(eval_noise)

image_idx = 0
fake_images_to_labels = {}
ages = np.zeros(opt.batchSize)


ages[:int(PERCENT_YOUNG*opt.batchSize)] = np.random.randint(0, 20,int(PERCENT_YOUNG*opt.batchSize))


ages[int(PERCENT_YOUNG*opt.batchSize):int(PERCENT_YOUNG*opt.batchSize)+int(PERCENT_MIDDLE*opt.batchSize)] = np.random.randint(20, 70, int(PERCENT_MIDDLE*opt.batchSize))

ages[int(PERCENT_YOUNG*opt.batchSize)+int(PERCENT_MIDDLE*opt.batchSize):] = np.random.randint(70, 90, int(PERCENT_OLD*opt.batchSize))



for i in range(len(ages)):
    if i % 1000 == 0: print(i)
    eval_noise_ = np.random.normal(0, 1, (1, nz))
    eval_label = np.random.randint(0, num_classes)
#     print('eval_label = {}'.format(eval_label))
    eval_onehot = np.zeros((1, num_classes + 1))
    eval_onehot[np.arange(1),0] = eval_label
    eval_age = ages[i]  #np.random.randint(0, MAX_AGE)
    eval_onehot[np.arange(1), 1] = eval_age / MAX_AGE
    eval_gender = np.random.randint(0, 2)
    eval_onehot[np.arange(1),2] = eval_gender
    #[0,1]
    # [healthy/unhealthy, age / 120, male/female]
    #[0,0.8,1]
    eval_noise_[np.arange(1), :num_classes + 1] = eval_onehot[np.arange(1)]
    eval_noise_ = (torch.from_numpy(eval_noise_))
    eval_noise.data.copy_(eval_noise_.view(1, nz, 1, 1))



    fake = netG(eval_noise)
#     print('age {} gender {} label {}'.format(eval_age, eval_gender, eval_label))
    fake_images_to_labels['fake_generated_image%03d.png' % (i)] = (eval_age, eval_gender, eval_label)
#     print(type(fake.data))
#     print(fake.data.shape)
#     print(fake.data[0])
    vutils.save_image(
        fake.data[0],
        '%s/fake_generated_image%03d.png' % ((opt.outf + ('healthy' if eval_label == 0 else 'unhealthy')), i)
        )

# print(fake_image_to_labels)
with open('/home/ubuntu/xray_gan/data/fake_images_to_labels.data', 'wb') as p_file:
    pickle.dump(fake_images_to_labels, p_file)
    
print(len(fake_images_to_labels.keys()))


