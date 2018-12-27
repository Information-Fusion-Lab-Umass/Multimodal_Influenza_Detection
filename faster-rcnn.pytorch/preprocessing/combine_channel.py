import cv2
import numpy as np 
from matplotlib import pyplot as plt 
import glob

original_images = [cv2.imread(file) for file in sorted(glob.glob("/home/debasmita/Documents/combined_test/set11_V001_*.jpg"))]
print("read original images")
saliency_maps = [cv2.imread(file) for file in sorted(glob.glob("/home/debasmita/Documents/salient_combined_test/set11_V001_*.jpg"))]
print("read saliency maps")

print(len(saliency_maps))


for i in range(len(original_images)):
	print(i)
	combined_image = np.zeros_like(original_images[i])
	#print(saliency_maps[i].shape)
	combined_image[:,:,0] = original_images[i][:,:,0]
	combined_image[:,:,1] = original_images[i][:,:,1]
	combined_image[:,:,2] = saliency_maps[i][:,:,2]
	cv2.imwrite("/home/debasmita/Documents/test_combined_salient_ir/set11_V001_I"+"%05d"%i+".jpg", combined_image)









#original_image = cv2.imread('I01255.jpg')
#saliency_map = cv2.imread('I01255_sal.jpg')
#plt.imshow(saliency_map[:,:,2])
#plt.show()

'''
combined_image = np.zeros_like(original_image)

combined_image[:,:,0] = original_image[:,:,0]
combined_image[:,:,1] = original_image[:,:,1]
combined_image[:,:,2] = saliency_map[:,:,2]
print(np.max(combined_image))

#plt.imshow(combined_image)
#plt.show()
'''
