
from collections import Counter

def confusion_matrix(data):
	# Implement the function here
	tp, fn, fp, tn = 0, 0, 0 ,0
	for r in data:
		y_true, y_pred = r[0], r[1]
		if (y_true == y_pred == 1):
			tp += 1
		elif (y_true == y_pred == 0):
			tn += 1
		elif (y_true == 0 and y_pred == 1):
			fp += 1
		else:
			fn += 1
	return [
		[tp, fn],
		[fp, tn],
	]
