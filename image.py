class Image:
	def __init__(self, data_array, lbl):
		self.image = data_array
		self.label = lbl

	def size(self):
		return self.image.size #returns number of pixels in image
