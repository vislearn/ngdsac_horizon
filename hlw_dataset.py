import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from skimage import io
from skimage import color
from skimage.draw import line, set_color, circle

import csv
import random
import math

class HLWDataset(Dataset):
	"""
	Access to the HLW dataset.

	Horizon Lines in the Wild, version 1
	http://www.cs.uky.edu/~jacobs/datasets/hlw/

	Assumed to be in the ./hlw/ folder in the working directory.

	"""

	def __init__(self, data_file, imsize, training):
		"""
		Read image list and meta data.

		data_file -- training/test split file
		imsize -- rescale images to this max side length
		training -- flag that toggles data augmentation
		"""

		self.imsize = imsize
		self.training = training

		# read image list
		img_db = open(data_file, 'r')
		self.images = ['hlw/images/' + f[:-1] for f in img_db.readlines()]
		img_db.close()

		# read ground truth labels
		metadata_file = open('hlw/metadata.csv')
		metadata = csv.reader(metadata_file)

		self.gt = [None] * len(self.images)

		for row in metadata:
			cur_image = 'hlw/images/' + row[0]
			if cur_image in self.images:
				idx = self.images.index(cur_image)

				cur_labels = torch.zeros((4))
				cur_labels[0] = float(row[1])
				cur_labels[1] = float(row[2])
				cur_labels[2] = float(row[3])
				cur_labels[3] = float(row[4])
				
				self.gt[idx] = cur_labels

		metadata_file.close()

	def __len__(self):
		return len(self.images)

	def __getitem__(self, idx):

		image = io.imread(self.images[idx])

		label_scale = max(image.shape[0], image.shape[1]) # longer image side
		image_scale = self.imsize / label_scale # normalize to longest side = 1

		#convert ground truth coordinate system (from zero center to zero corner)
		yOffset = image.shape[0] / label_scale / 2 # mind zero padding to make image square
		xOffset = image.shape[1] / label_scale / 2

		gt = self.gt[idx].clone()
		gt /= label_scale
		gt[0] += xOffset
		gt[1] *= -1
		gt[1] += yOffset
		gt[2] += xOffset
		gt[3] *= -1
		gt[3] += yOffset

		# convert image to RGB
		if len(image.shape) < 3:
			image = color.gray2rgb(image)

		# original image dimensions		
		src_h = int(image.shape[0] * image_scale)
		src_w = int(image.shape[1] * image_scale)

		# resize and convert to gray scale
		image = transforms.functional.to_pil_image(image)
		image = transforms.functional.resize(image, (src_h, src_w))
		image = transforms.functional.adjust_saturation(image, 0)

		if self.training:
			# data augmenation
			random_shift = 8
			random_shift_x = random.randint(-random_shift,random_shift)
			random_shift_y = random.randint(-random_shift,random_shift)
			random_angle = random.uniform(-5,5)
			random_scale = random.uniform(0.8,1.2)

			image = transforms.functional.adjust_contrast(image, random.uniform(0.8, 1.2))
			image = transforms.functional.adjust_brightness(image, random.uniform(0.8, 1.2))
			image = transforms.functional.affine(image, random_angle, (random_shift_x, random_shift_y), random_scale, 0, resample=2)

		image = transforms.functional.to_tensor(image)

		# image dimensions after augmentation
		imh = image.size(1)
		imw = image.size(2)

		# zero pad image to make it square
		padding_left = int((self.imsize - image.size(2)) / 2)
		padding_right = self.imsize - image.size(2) - padding_left
		padding_top = int((self.imsize - image.size(1)) / 2)
		padding_bottom = self.imsize - image.size(1) - padding_top

		padding = torch.nn.ZeroPad2d((padding_left, padding_right, padding_top, padding_bottom))
		image = padding(image)
	
		# normalization of color values (mean and stddev calculated offline over HLW training set)
		img_mask = image.sum(0) > 0

		image[:,img_mask] -= 0.45
		image[:,img_mask] /= 0.25

		# add padding offset due to augmentation to ground truth
		gt[0] += padding_left / self.imsize
		gt[1] += padding_top / self.imsize
		gt[2] += padding_left / self.imsize
		gt[3] += padding_top / self.imsize

		if self.training:
			# rotate and scale ground truth according to augmentation
			a = random_angle * math.pi / 180
			cos_a = math.cos(a)
			sin_a = math.sin(a)
			rot_off = 0.5

			gt -= rot_off

			r_x1 = cos_a * gt[0] - sin_a * gt[1]
			r_y1 = sin_a * gt[0] + cos_a * gt[1]
			r_x2 = cos_a * gt[2] - sin_a * gt[3]
			r_y2 = sin_a * gt[2] + cos_a * gt[3]		

			gt[0] = r_x1
			gt[1] = r_y1
			gt[2] = r_x2 
			gt[3] = r_y2

			gt *= random_scale
			gt += rot_off

			gt[0] += random_shift_x / self.imsize 
			gt[2] += random_shift_x / self.imsize 
			gt[1] += random_shift_y / self.imsize 
			gt[3] += random_shift_y	/ self.imsize 	

		#calculate slope and intercept
		labels = torch.zeros((2))
		labels[1] = (gt[3] - gt[1]) / (gt[2] - gt[0])
		labels[0] = gt[1] - labels[1] * gt[0]
		
		# pre-compute start and end coordinate of line (used in loss)
		xStart = padding_left / self.imsize
		xEnd = xStart + imw / self.imsize

		# meta data for calculating the HLW loss correctly:
		# xStart -- at which x position does the GT line enter the image
		# xEnd -- at which x position does the GT line leave the image
		# imh -- what is the image height (without zero padding)
		return image, labels, xStart, xEnd, imh, idx