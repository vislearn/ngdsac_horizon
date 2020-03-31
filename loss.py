import torch

class Loss:
	'''
	Compares two lines by calculating the distance between their ends in the image.
	'''

	def __init__(self, image_size, cut_off = 0.25):
		'''
		Constructor.

		image_size -- size of the input images, used to normalize the loss
		cut_off -- soft clamping of loss after this value
		'''
		self.image_size = image_size
		self.cut_off = cut_off
	
	def __get_max_points(self, slope, intercept, xStart, xEnd):
		'''
		Calculates the 2D points where a line intersects with the image borders.

		slope -- slope of the line
		intercept -- intercept of the line
		'''
		pts = torch.zeros([2, 2])

		x0 = float(xStart)
		x1 = float(xEnd)
		y0 = intercept + x0 * slope
		y1 = intercept + x1 * slope
		
		pts[0, 0] = x0
		pts[0, 1] = y0
		pts[1, 0] = x1
		pts[1, 1] = y1

		return pts

	def __call__(self, est, gt,  xStart, xEnd, imh):
		'''
		Calculate the line loss.

		est -- estimated line, form: [intercept, slope]
		gt -- ground truth line, form: [intercept, slope]
		'''

		pts_est = self.__get_max_points(est[1], est[0], xStart, xEnd,)
		pts_gt = self.__get_max_points(gt[1], gt[0], xStart, xEnd,)

		# not clear which ends of the lines should be compared (there are ambigious cases), compute both and take min
		loss = pts_est - pts_gt
		loss = loss.norm(2, 1).max()

		loss = loss * self.image_size / float(imh)

		# soft clamping
		if loss < self.cut_off:
			return loss
		else:
			return torch.sqrt(self.cut_off * loss)
