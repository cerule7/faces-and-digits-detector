<<<<<<< HEAD
class Image:
	def __init__(self, data_array, lbl):
		self.image = data_array
		self.label = lbl
	def size(self):
		return length(data_array) #returns number of pixels in image
=======
class Image:
	def __init__(self, data_array, lbl):
		self.image = data_array
		self.label = lbl

	def size(self):
		return self.image.size #returns number of pixels in image
>>>>>>> 635ec2c653d928de6cd650d37d58419af2b512bd
