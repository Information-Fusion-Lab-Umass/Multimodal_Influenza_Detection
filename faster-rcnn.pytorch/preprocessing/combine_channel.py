import cv2
import numpy as np 
from matplotlib import pyplot as plt 

original_image = cv2.imread('I01465.jpg')
saliency_map = cv2.imread('I01465_sal.jpg')
#plt.imshow(saliency_map[:,:,2])
#plt.show()

combined_image = np.zeros_like(original_image)

combined_image[:,:,0] = original_image[:,:,0]
combined_image[:,:,1] = original_image[:,:,1]
combined_image[:,:,2] = saliency_map[:,:,2]

plt.imshow(combined_image)
plt.show()
