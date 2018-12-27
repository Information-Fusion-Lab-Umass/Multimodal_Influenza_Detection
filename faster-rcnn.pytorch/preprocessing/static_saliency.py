# USAGE
# python static_saliency.py --image images/neymar.jpg

# import the necessary packages
import argparse
import cv2
import glob

# construct the argument parser and parse the arguments
#ap = argparse.ArgumentParser()
#ap.add_argument("-i", "--image", required=True,
#	help="path to input image")
#args = vars(ap.parse_args())

# load the input image
images = [cv2.imread(file) for file in sorted(glob.glob("/home/debasmita/Documents/combined_test/set11_V001_*.jpg"))]
print(file)
print(len(images))

for i in range(len(images)):
	# initialize OpenCV's static fine grained saliency detector and
	# compute the saliency map
	#print(i)
	saliency = cv2.saliency.StaticSaliencyFineGrained_create()
	(success, saliencyMap) = saliency.computeSaliency(images[i])

	# if we would like a *binary* map that we could process for contours,
	# compute convex hull's, extract bounding boxes, etc., we can
	# additionally threshold the saliency map
	threshMap = cv2.threshold(saliencyMap.astype("uint8"), 0, 255,
		cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

	# show the images
	#cv2.write("Image", image)
	cv2.imwrite('/home/debasmita/Documents/salient_combined_test/set11_V001_I'+'%05d'%i+'.jpg', saliencyMap)
	#cv2.imwrite("I01255_thresh.jpg", threshMap)
	cv2.waitKey(0)
'''
# initialize OpenCV's static saliency spectral residual detector and
# compute the saliency map
#saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
#cv2.imshow("Image", image)
#(success, saliencyMap) = saliency.computeSaliency(image)
#saliencyMap = (saliencyMap * 255).astype("uint8")
#cv2.imshow("Image", image)
#cv2.imshow("Output", saliencyMap)
#cv2.waitKey(0)

# initialize OpenCV's static fine grained saliency detector and
# compute the saliency map
saliency = cv2.saliency.StaticSaliencyFineGrained_create()
(success, saliencyMap) = saliency.computeSaliency(image)

# if we would like a *binary* map that we could process for contours,
# compute convex hull's, extract bounding boxes, etc., we can
# additionally threshold the saliency map
threshMap = cv2.threshold(saliencyMap.astype("uint8"), 0, 255,
	cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

# show the images
#cv2.write("Image", image)
cv2.imwrite("I01255_sal.jpg", saliencyMap)
cv2.imwrite("I01255_thresh.jpg", threshMap)
cv2.waitKey(0)
'''
