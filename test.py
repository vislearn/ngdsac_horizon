import torch
import numpy as np

from hlw_dataset import HLWDataset
from model import Model

import time
import argparse

from ngdsac import NGDSAC
from loss import Loss

parser = argparse.ArgumentParser(description='Test a trained horizon line estimation network (DSAC or NG-DSAC).', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('model', type=str,
	help='a trained network')

parser.add_argument('--capacity', '-c', type=int, default=4, 
	help='controls the model capactiy of the network, must match the model to load (multiplicative factor for number of channels)')

parser.add_argument('--imagesize', '-is', type=int, default=256, 
	help='size of input images to the network, must match the model to load')

parser.add_argument('--inlierthreshold', '-it', type=float, default=0.05, 
	help='threshold used in the soft inlier count, relative to image size')

parser.add_argument('--inlieralpha', '-ia', type=float, default=0.1, 
	help='scaling factor for the soft inlier scores (controls the peakiness of the hypothesis distribution)')

parser.add_argument('--inlierbeta', '-ib', type=float, default=100.0, 
	help='scaling factor within the sigmoid of the soft inlier count')

parser.add_argument('--hypotheses', '-hyps', type=int, default=16, 
	help='number of line hypotheses sampled for each image')

parser.add_argument('--session', '-sid', default='', 
	help='custom session name appended to output files; useful to separate different runs of the program')

parser.add_argument('--invalidloss', '-il', type=int, default=1, 
	help='penalty for sampling invalid hypotheses')

parser.add_argument('--uniform', '-u', action='store_true', 
	help='disable neural-guidance and sample data points uniformely; corresponds to a DSAC model')

opt = parser.parse_args()

# setup test set
testset = HLWDataset('hlw/split/test.txt', opt.imagesize, training=False)
testset_loader = torch.utils.data.DataLoader(testset, shuffle=False, num_workers=6, batch_size=1)

# setup ng dsac estimator
loss = Loss(opt.imagesize, cut_off = 100) 
ngdsac = NGDSAC(opt.hypotheses, opt.inlierthreshold, opt.inlierbeta, opt.inlieralpha, loss, opt.invalidloss)

# load network
nn = Model(opt.capacity)
nn.load_state_dict(torch.load(opt.model))
nn.eval()
nn = nn.cuda()

# write test results
test_log = open('test_'+opt.session+'.txt', 'w', 1)

def AUC(losses, thresholds, binsize):
	"""Compute the AUC up to a set of error thresholds.
	Return mutliple AUC corresponding to multiple threshold provided.
	Keyword arguments:
	losses -- list of losses which the AUC should be calculated for
	thresholds -- list of threshold values up to which the AUC should be calculated
	binsize -- bin size to be used fo the cumulative histogram when calculating the AUC, the finer the more accurate
	"""

	bin_num = int(max(thresholds) / binsize)
	bins = np.arange(bin_num + 1) * binsize  

	hist, _ = np.histogram(losses, bins) # histogram up to the max threshold
	hist = hist.astype(np.float32) / len(losses) # normalized histogram
	hist = np.cumsum(hist) # cumulative normalized histogram
	 
	# calculate AUC for each threshold
	return [np.mean(hist[:int(t / binsize)]) for t in thresholds]

losses = []

for inputs, labels, xStart, xEnd, imh, idx in testset_loader:

	start_time = time.time()

	with torch.no_grad():
		# forward pass of neural network
		points, log_probs = nn(inputs.cuda())

		if opt.uniform:
			# overwrite neural guidance with uniform sampling probabilities
			log_probs.fill_(1/log_probs.size(1))
			log_probs = torch.log(log_probs)

		# fit line with NG-DSAC
		ngdsac(points, log_probs, labels, xStart, xEnd, imh) 

	# evaluate (assumes a batch size of 1)
	cur_loss = loss(ngdsac.est_parameters[0], labels[0], xStart[0], xEnd[0], imh[0])

	# wrap up
	end_time = time.time()-start_time
	print('Image: %s, Loss: %2.2f, Time: %.2fs' 
		% (testset.images[idx[0]], cur_loss, end_time), flush=True)

	test_log.write('%s %f\n' % (testset.images[idx[0]], cur_loss))
	losses.append(cur_loss)

auc = AUC(losses, thresholds=[0.25], binsize=0.0001)

print("\n==========================================")
print("AUC@0.25: %.1f%%" % (auc[0]*100))
print("==========================================\n")
print('Done without errors.')

test_log.close()
