# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 01:20:27 2020

@author: XXX
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

#Read the image
image = cv2.imread('blue_cat.jpg')
#Print the shape of image for further analysis
print(type(image),image.shape)

#Best habit is to make a copy of image 
image_copy = np.copy(image)
#Change the iage to RGB format as Opencv have BGR format as default
image_copy = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
plt.imshow(image_copy)


# Define the threshold
lower_blue = np.array([0,150,200])
upper_blue = np.array([255,255,255])

#Create a mask
mask = cv2.inRange(image_copy,lower_blue,upper_blue)
plt.imshow(mask, cmap='gray')

# Mask the image to let the cat show through
masked_image = np.copy(image_copy)

masked_image[mask == 0] = [0, 0, 0]

# Display it!
plt.imshow(masked_image,cmap='gray')

#load in background image
background_image = cv2.imread('back.jpg')
background_image = cv2.cvtColor(background_image, cv2.COLOR_BGR2RGB)


height = image.shape[0]
width= image.shape[1]
dim = (width,height)
background_image= cv2.resize(background_image,dim,interpolation= cv2.INTER_AREA)
print(background_image.shape)

cropped = background_image[0:1080,0:1920]
cropped[mask!=0] = [0,0,0]
plt.imshow(cropped)

complete_image = cropped+ masked_image

plt.imshow(complete_image)

cv2.imwrite("done.jpg", complete_image)
 
# Lets play with an image having green background

import cv2
import matplotlib.pyplot as plt
import numpy as np

# Load an image 
new_image = cv2.imread('green_cat.jpg')
new_image = cv2.cvtColor(new_image,cv2.COLOR_BGR2RGB)
print(new_image.shape)  ## (480, 852, 3)

new_image_copy = np.copy(new_image)
plt.imshow(new_image_copy)

# Lets defne threshold for image to get green intensity pixels

lower_green = np.array([0,185,0])
upper_green = np.array([255,255,255])

#Lets create a mask now
mask = cv2.inRange(new_image_copy, lower_green, upper_green)
plt.imshow(mask,cmap='gray')

masked_image_new =  np.copy(new_image)
masked_image_new[mask!=0] = [0,0,0]
plt.imshow(masked_image_new)

background_image_new = cv2.imread('back.jpg')
background_image_new = np.copy(background_image_new)

cropped_image = background_image_new[0:480,0:852]
cropped_image[mask==0] = [0,0,0]

plt.imshow(cropped_image)
compl = cropped_image+masked_image_new
plt.imshow(compl)
cv2.imwrite('green.jpg',compl)









