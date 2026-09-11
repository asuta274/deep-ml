
def calculate_brightness(img):
	# Write your code here
	#validate
	l = len(img)
	if l == 0: 
		return -1
	w = len(img[0])
	for r in img:
		if len(r) != w:
			return -1
		for v in r:
			if v < 0 or v > 255:
				return -1

	import numpy as np
	img = np.array(img)
	if len(img.shape) != 2:
		return -1
	
	w, h = img.shape
	if (w * h) == 0:
		return -1
	return np.sum(img)/(w * h)
