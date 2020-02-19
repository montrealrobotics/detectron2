import numpy as np

out = np.load('kitti_tracking_car_detections.npy', allow_pickle=True)[()]

## Sequence name
seq_name = '0018' ## could be '0000', '0002', '0003', '0004', '0007', '0008', '0018'

## getting detections for a particular sequence
detections = out[seq_name]	## array


## detections have detections of all images of a particular sequence
## iterating through list of dictionary
for detection in detections:

	## a particular element in the array is dictionary
	for key in detection.keys():

		## key of the dictionary is name of the image
		image_name = key

		## getting all the car detections of a particular image
		per_image_detection = detection[key]

		## iterating through all the detections of a particular image
		for det in per_image_detection:
			## (x,y) --> top left corner
			## (w,h) --> height and width of the box
			x, y, h, w = det 
			print(f"Car detection of image {key} : {x,y,h,w}")

