import torch
import torch.nn as nn
import torch.nn.functional as F

import random


class Model(nn.Module):
	'''
	Network predicting a set of points for an input image.

	'''

	def __init__(self, net_capacity):
		'''
		Constructor.

		net_capacity -- scaling factor applied to the number of channels in each layer
		'''
		super(Model, self).__init__()

		c = net_capacity

		strides = [1, 1, 2, 2, 2, 2, 2]

		self.output_dim = 2 # dimensionality of the output points

		# build network
		self.conv1 = nn.Conv2d(3, 8*c, 3, strides[0], 1)
		self.bn1 = nn.BatchNorm2d(8*c)	
		self.conv2 = nn.Conv2d(8*c, 16*c, 3, strides[1], 1)
		self.bn2 = nn.BatchNorm2d(16*c)
		self.conv3 = nn.Conv2d(16*c, 32*c, 3, strides[2], 1)
		self.bn3 = nn.BatchNorm2d(32*c)
		self.conv4 = nn.Conv2d(32*c, 64*c, 3, strides[3], 1)
		self.bn4 = nn.BatchNorm2d(64*c)

		self.conv5 = nn.Conv2d(64*c, 64*c, 3, strides[4], 1)
		self.bn5 = nn.BatchNorm2d(64*c)	
		self.conv6 = nn.Conv2d(64*c, 64*c, 3, strides[5], 1)
		self.bn6 = nn.BatchNorm2d(64*c)
		self.conv7 = nn.Conv2d(64*c, 64*c, 3, strides[6], 1)
		self.bn7 = nn.BatchNorm2d(64*c)

		self.conv8 = nn.Conv2d(64*c, 64*c, 3, 1, 1)
		self.bn8 = nn.BatchNorm2d(64*c)	
		self.conv9 = nn.Conv2d(64*c, 64*c, 3, 1, 1)
		self.bn9 = nn.BatchNorm2d(64*c)
		self.conv10 = nn.Conv2d(64*c, 64*c, 3, 1, 1)
		self.bn10 = nn.BatchNorm2d(64*c)		
				
		# output branch 1 for predicting points
		self.fc1 = nn.Conv2d(64*c, 128*c, 1, 1, 0)
		self.bn_fc1 = nn.BatchNorm2d(128*c)
		self.fc2 = nn.Conv2d(128*c, 128*c, 1, 1, 0)
		self.bn_fc2 = nn.BatchNorm2d(128*c)
		self.fc3 = nn.Conv2d(128*c, self.output_dim, 1, 1, 0)

		# output branch 2 for predicting neural guidance
		self.fc1_1 = nn.Conv2d(64*c, 128*c, 1, 1, 0)
		self.bn_fc1_1 = nn.BatchNorm2d(128*c)
		self.fc2_1 = nn.Conv2d(128*c, 128*c, 1, 1, 0)
		self.bn_fc2_1 = nn.BatchNorm2d(128*c)
		self.fc3_1 = nn.Conv2d(128*c, 1, 1, 1, 0)

	def forward(self, inputs):
		'''
		Forward pass.

		inputs -- 4D data tensor (BxCxHxW)
		'''

		batch_size = inputs.size(0)

		x = F.relu(self.bn1(self.conv1(inputs)))
		x = F.relu(self.bn2(self.conv2(x)))
		x = F.relu(self.bn3(self.conv3(x)))
		x = F.relu(self.bn4(self.conv4(x)))

		res = x

		x = F.relu(self.bn5(self.conv5(res)))
		x = F.relu(self.bn6(self.conv6(x)))
		x = F.relu(self.bn7(self.conv7(x)))

		res = x

		x = F.relu(self.bn8(self.conv8(res)))
		x = F.relu(self.bn9(self.conv9(x)))
		x = F.relu(self.bn10(self.conv10(x)))

		res = res + x
		
		# === output branch 1, predict 2D points ====================
		x1 = F.relu(self.bn_fc1(self.fc1(res)))
		x1 = F.relu(self.bn_fc2(self.fc2(x1)))
		points = self.fc3(x1)
		points = torch.sigmoid(points) # normalize to 0,1

		# map local (patch-centric) point predictions to global image coordinates
		# i.e. distribute the points over the image
		patch_offset = 1 / points.size(2)
		patch_size = 3

		points = points * patch_size - patch_size / 2 + patch_offset / 2

		for col in range(0, points.size(3)):
			points[:,1,:,col] = points[:,1,:,col] + col * patch_offset
			
		for row in range(0, points.size(2)):
			points[:,0,row,:] = points[:,0,row,:] + row * patch_offset

		points = points.view(batch_size, 2, -1)
		
		# === output branch 2, predict neural guidance ============== 
		x2 = F.relu(self.bn_fc1_1(self.fc1_1(res.detach())))
		x2 = F.relu(self.bn_fc2_1(self.fc2_1(x2)))
		log_probs = self.fc3_1(x2)
		log_probs = log_probs.view(batch_size, -1)
		log_probs = F.logsigmoid(log_probs) # normalize output to 0,1

		# normalize probs to sum to 1
		normalizer = torch.logsumexp(log_probs, dim=1)
		normalizer = normalizer.unsqueeze(1).expand(-1, log_probs.size(1))
		norm_log_probs = log_probs - normalizer

		return points, norm_log_probs
