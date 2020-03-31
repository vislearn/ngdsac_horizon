import torch
import torch.nn.functional as F

import random

class NGDSAC:
	'''
	Neural-Guided DSAC to robustly fit lines.
	'''

	def __init__(self, hyps, inlier_thresh, inlier_beta, inlier_alpha, loss_function, invalid_loss):
		'''
		Constructor.

		hyps -- number of line hypotheses sampled for each image
		inlier_thresh -- threshold used in the soft inlier count, its measured in relative image size
		inlier_beta -- scaling factor within the sigmoid of the soft inlier count
		inlier_alpha -- scaling factor for the soft inlier scores (controls the peakiness of the hypothesis distribution)
		loss_function -- function to compute the quality of estimated line parameters wrt ground truth
		invalid_loss -- punishment when sampling invalid hypothesis
		'''

		self.hyps = hyps
		self.inlier_thresh = inlier_thresh
		self.inlier_beta = inlier_beta
		self.inlier_alpha = inlier_alpha
		self.loss_function = loss_function
		self.invalid_loss = invalid_loss

	def __sample_hyp(self, x, y, p, pool):
		'''
		Calculate a line hypothesis (slope, intercept) from two random points.

		x -- vector of x values
		y -- vector of y values
		p -- sampling probabilities for selecting points
		pool -- indicator vector updated with which points have been selected
		'''

		# select points
		idx = torch.multinomial(p, 2, replacement = True)
		idx1 = int(idx[0])
		idx2 = int(idx[1])

		# set indicators which points have been selected
		pool[idx1] += 1
		pool[idx2] += 1
	
		# validity check, do not choose too close together
		if torch.abs(x[idx1] - x[idx2]) < 0.05:
			return 0, 0, False # no valid hypothesis found, indicated by False

		# calculate line parameters
		slope = (y[idx1] - y[idx2]) / (x[idx1] - x[idx2])
		intercept = y[idx1] - slope * x[idx1]

		return slope, intercept, True # True indicates a valid hypothesos

		

	def __soft_inlier_count(self, slope, intercept, x, y):
		'''
		Soft inlier count for a given line and a given set of points.

		slope -- slope of the line
		intercept -- intercept of the line
		x -- vector of x values
		y -- vector of y values
		'''

		# point line distances
		dists = torch.abs(slope * x - y + intercept)
		dists = dists / torch.sqrt(slope * slope + 1)

		# soft inliers
		dists = 1 - torch.sigmoid(self.inlier_beta * (dists - self.inlier_thresh)) 
		score = torch.sum(dists)

		return score, dists	

	def __call__(self, prediction, log_probs, labels,  xStart, xEnd, imh):
		'''
		Perform robust, differentiable line fitting according to NG-DSAC.

		Returns the expected loss and hypothesis distribution entropy.
		Expected loss can be used for backprob, entropy for monitoring / debugging.

		prediction -- predicted 2D points for a batch of images, array of shape (Bx2xN) where
			B is the number of images in the batch
			2 is the number of point dimensions (y, x)
			N is the number of predicted points
		log_probs -- log of selection probabilities, array of shape (BxN)
		labels -- ground truth labels for the batch, array of shape (Bx2) where
			2 is the number of parameters (intercept, slope)
		xStart -- x-values where each ground truth line starts (for calculating the loss), array of shape (B)
		xEnd -- x-values where each ground truth line ends (for calculating the loss), array of shape (B)
		imh -- relative height of the image (for calculating the loss), <= 1, array of shape (B)
		'''

		# faster on CPU because of many, small matrices
		prediction = prediction.cpu()
		batch_size = prediction.size(0)

		avg_exp_loss = 0 # expected loss
		avg_entropy = 0 # hypothesis distribution entropy 

		self.est_parameters = torch.zeros(batch_size, 2) # estimated lines (w/ max inliers)
		self.batch_inliers = torch.zeros(batch_size, prediction.size(2)) # (soft) inliers for estimated lines
		self.g_log_probs = torch.zeros(batch_size, prediction.size(2)) # gradient tensor for neural guidance

		for b in range(0, batch_size):

			hyp_losses = torch.zeros([self.hyps, 1]) # loss of each hypothesis
			hyp_scores = torch.zeros([self.hyps, 1]) # score of each hypothesis

			max_score = 0 	# score of best hypothesis

			y = prediction[b, 0] # all y-values of the prediction
			x = prediction[b, 1] # all x.values of the prediction

			p = torch.exp(log_probs[b]) # selection probabilities for points

			for h in range(0, self.hyps):	

				# === step 1: sample hypothesis ===========================
				slope, intercept, valid = self.__sample_hyp(x, y, p, self.g_log_probs[b])
				if not valid: 
					hyp_losses[h] = self.invalid_loss
					hyp_scores[h] = 0.0001
					continue # skip other steps for invalid hyps

				# === step 2: score hypothesis using soft inlier count ====
				score, inliers = self.__soft_inlier_count(slope, intercept, x, y)

				hyp = torch.zeros([2])
				hyp[1] = slope
				hyp[0] = intercept

				# === step 3: calculate loss of hypothesis ================
				loss = self.loss_function(hyp, labels[b],  xStart[b], xEnd[b], imh[b]) 

				# store results
				hyp_losses[h] = loss
				hyp_scores[h] = score

				# keep track of best hypothesis so far
				if score > max_score:
					max_score = score
					self.est_parameters[b] = hyp.detach()
					self.batch_inliers[b] = inliers.detach()

			# === step 4: calculate the expectation ===========================

			#softmax distribution from hypotheses scores			
			hyp_scores = F.softmax(self.inlier_alpha * hyp_scores, 0)

			# expectation of loss
			avg_exp_loss += torch.sum(hyp_losses * hyp_scores)

		return avg_exp_loss / batch_size
