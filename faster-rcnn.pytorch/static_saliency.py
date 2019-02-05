# USAGE
# python static_saliency.py --image images/neymar.jpg

# import the necessary packages
import argparse
import cv2
import glob

# construct the argument parser and parse the arguments
#ap = argparse.ArgumentParser()
#ap.add_argument("-i", "--image", required=True,
#       help="path to input image")
#args = vars(ap.parse_args())
image = cv2.imread("/home/dghose/Project/Influenza_Detection/Data/KAIST/Train/combined_train/set00_V000_I00100.jpg")
print(image)

# load the input image
images = [cv2.imread(file) for file in glob.glob("/home/dghose/Project/Influenza_Detection/Data/KAIST/Train/combined_train/*.jpg")]
image = cv2.imread("/home/dghose/Project/Influenza_Detection/Data/KAIST/Train/combined_train/I00100.jpg")
#print(image.shape)
print(file)

print(len(images))

for i in range(len(images)):
    # initialize OpenCV's static fine grained saliency detector and
    # compute the saliency map
    saliency = cv2.saliency.StaticSaliencyFineGrained_create()
    (success, saliencyMap) = saliency.computeSaliency(images[i])

    # if we would like a *binary* map that we could process for contours,
    # compute convex hull's, extract bounding boxes, etc., we can
    # additionally threshold the saliency map
    threshMap = cv2.threshold(saliencyMap.astype("uint8"), 0, 255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # show the images
    #cv2.write("Image", image)
    cv2.imwrite('/home/dghose/Project/Influenza_Detection/Data/KAIST/Train/combined_train/saliency_maps/I'+'%05d'%i+'.jpg', saliencyMap)
    #cv2.imwrite("I01255_thresh.jpg", threshMap)
    cv2.waitKey(0)
