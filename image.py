class Image:
	def __init__(self, data_array, lbl):
		self.image = data_array
		self.label = lbl
	def size(self):
		return length(data_array) #returns number of pixels in image, not sure if this code is correct
