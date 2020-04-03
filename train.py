import torch
import torch.nn as nn
import torch.optim as optim

from hlw_dataset import HLWDataset
from model import Model

import torchvision.utils as vutils
from skimage.io import imsave
import skimage.io as io

import time
import warnings
import argparse

from ngdsac import NGDSAC
from loss import Loss

parser = argparse.ArgumentParser(description='Train a horizon line estimation network on the HLW dataset using (NG-)DSAC.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--session', '-sid', default='', 
	help='custom session name appended to output files, useful to separate different runs of the program')

parser.add_argument('--capacity', '-c', type=int, default=4, 
	help='controls the model capactiy of the network by scaling the number of channels in each layer')

parser.add_argument('--imagesize', '-is', type=int, default=256, 
	help='rescale images to this max side length')

parser.add_argument('--inlierthreshold', '-it', type=float, default=0.05, 
	help='threshold used in the soft inlier count, relative to image size (1 = image width)')

parser.add_argument('--inlieralpha', '-ia', type=float, default=0.1, 
	help='scaling factor for the soft inlier scores (controls the peakiness of the hypothesis distribution)')

parser.add_argument('--inlierbeta', '-ib', type=float, default=100.0, 
	help='scaling factor within the sigmoid of the soft inlier count')

parser.add_argument('--storeinterval', '-si', type=int, default=1000, 
	help='store network weights and a prediction vizualisation every x training iterations')

parser.add_argument('--hypotheses', '-hyps', type=int, default=16, 
	help='number of line hypotheses sampled for each image')

parser.add_argument('--batchsize', '-bs', type=int, default=32, 
	help='training batch size')

parser.add_argument('--learningrate', '-lr', type=float, default=0.0001, 
	help='learning rate')

parser.add_argument('--iterations', '-i', type=int, default=250000, 
	help='number of training iterations (parameter updates)')

parser.add_argument('--scheduleoffset', '-soff', type=int, default=150000, 
	help='start learning rate schedule ofter this many iterations')

parser.add_argument('--schedulestep', '-sstep', type=int, default=25000, 
	help='half learning rate ofter this many iterations')

parser.add_argument('--samplesize', '-ss', type=int, default=2, 
	help='number of ng-dsac runs for each training image to approximate the expectation when learning neural guidance')

parser.add_argument('--invalidloss', '-il', type=float, default=1, 
	help='penalty for sampling invalid hypotheses')

parser.add_argument('--uniform', '-u', action='store_true', 
	help='disable neural-guidance and sample data points uniformely; corresponds to a DSAC model')

opt = parser.parse_args()

# setup training set
trainset = HLWDataset('hlw/split/train.txt', opt.imagesize, training=True)
trainset_loader = torch.utils.data.DataLoader(trainset, shuffle=True, num_workers=6, batch_size=opt.batchsize)

# setup ng dsac estimator
loss = Loss(opt.imagesize) 
ngdsac = NGDSAC(opt.hypotheses, opt.inlierthreshold, opt.inlierbeta, opt.inlieralpha, loss, opt.invalidloss)

# setup network
nn = Model(opt.capacity)
nn.train()
nn = nn.cuda()

# optimizer and lr schedule (schedule offset handled further below)
optimizer = optim.Adam(nn.parameters(), lr=opt.learningrate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=opt.schedulestep, gamma=0.5)

# keep track of training progress
train_log = open('log_'+opt.session+'.txt', 'w', 1)

iteration = 0
epoch = 0

while True:

	print('=== Epoch: ', epoch, '========================================')

	for inputs, labels, xStart, xEnd, imh, idx in trainset_loader:

		start_time = time.time()

		# predict points and neural guidance
		inputs = inputs.cuda()
		points, log_probs = nn(inputs)

		if opt.uniform:
			# overwrite neural guidance with uniform sampling probabilities
			log_probs.fill_(1/log_probs.size(1))
			log_probs = torch.log(log_probs)

		g_log_probs = torch.zeros(log_probs.size()).cuda() # gradients for neural guidance
		g_points = torch.zeros(points.size()).cuda() # gradients for point positions

		# approximate neural guidance expectation by sampling

		exp_loss = 0 # mean loss over samples
		losses = [] # losses per sample, we will substract the mean loss later as baseline
		sample_grads = [] # gradients per sample

		for s in range(opt.samplesize):

			# fit lines with ngdsac (also calculates expected loss for DSAC)
			cur_loss = ngdsac(points, log_probs, labels, xStart, xEnd, imh)

			# calculate gradients for 2D point predictions by PyTorch (autograd of expected loss)
			g_points += torch.autograd.grad(cur_loss, points)[0]
			# gradients for neural guidance have been calculated by NGDSAC
			sample_grads.append(ngdsac.g_log_probs.cuda() / opt.batchsize)

			exp_loss += cur_loss
			losses.append(cur_loss)
		 
		g_points /= opt.samplesize
		exp_loss /= opt.samplesize

		# subtract baseline (mean over samples) for neural guidance gradients to reduce variance
		for i,l in enumerate(losses):
			g_log_probs += sample_grads[i] * (float(l) - float(exp_loss))
		g_log_probs /= opt.samplesize * 10
		g_log_probs = g_log_probs.cuda()

		if opt.uniform:
			# DSAC, no neural guidance
			torch.autograd.backward((points), (g_points))
		else:
			# full NG-DSAC
			torch.autograd.backward((points, log_probs), (g_points, g_log_probs))
		
		optimizer.step() 
		# apply learning rate schedule
		if iteration >= opt.scheduleoffset:
			scheduler.step()
		optimizer.zero_grad()

		# wrap up
		end_time = time.time()-start_time
		print('Iteration: %6d, Exp. Loss: %2.2f, Time: %.2fs' % (iteration, exp_loss, end_time), flush=True)

		train_log.write('%d %f\n' % (iteration, exp_loss))

		if iteration % opt.storeinterval == 0:
			torch.save(nn.state_dict(), './weights_' + opt.session + '.net')

		del exp_loss, points, log_probs, g_log_probs, g_points, losses, sample_grads

		iteration += 1

		if iteration > opt.iterations:
			break

	epoch += 1

	if iteration > opt.iterations:
		break

print('Done without errors.')
train_log.close()
