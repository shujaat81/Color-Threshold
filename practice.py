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

#############################################################
# Color spaces
import cv2
import matplotlib.pyplot as plt
import numpy as np

#load the image
image = cv2.imread('water_balloons.jpg')
image_copy = np.copy(image)
image_copy = cv2.cvtColor(image_copy,cv2.COLOR_BGR2RGB)

# RGB channels
r = image_copy[:,:,0]
g = image_copy[:,:,1]
b = image_copy[:,:,2]

f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20,10))

ax1.set_title('Red')
ax1.imshow(r, cmap='gray')

ax2.set_title('Green')
ax2.imshow(g, cmap='gray')

ax3.set_title('Blue')
ax3.imshow(b, cmap='gray')

hsv = cv2.cvtColor(image_copy,cv2.COLOR_RGB2HSV)
#HSV Channels
h = hsv[:,:,0]
s = hsv[:,:,1]
v = hsv[:,:,2]

f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20,10))

ax1.set_title('Hue')
ax1.imshow(h, cmap='gray')

ax2.set_title('Saturation')
ax2.imshow(s, cmap='gray')

ax3.set_title('Value')
ax3.imshow(v, cmap='gray')



# Define our color selection criteria in HSV values
lower_hue = np.array([160,0,0]) 
upper_hue = np.array([180,255,255])

# Define our color selection criteria in RGB values
lower_pink = np.array([180,0,100]) 
upper_pink = np.array([255,255,230])

# Define the masked area in RGB space
mask_rgb = cv2.inRange(image, lower_pink, upper_pink)

# mask the image
masked_image = np.copy(image)
masked_image[mask_rgb==0] = [0,0,0]

# Vizualize the mask
plt.imshow(masked_image)


# Now try HSV!

# Define the masked area in HSV space
mask_hsv = cv2.inRange(hsv, lower_hue, upper_hue)

# mask the image
masked_image = np.copy(image)
masked_image[mask_hsv==0] = [0,0,0]

# Vizualize the mask
plt.imshow(masked_image)

#########################################################################

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import numpy as np
import cv2


# Read in the image
image = mpimg.imread('car_green_screen2.jpg')

plt.imshow(image)

lower_green = np.array([0,180,0]) 
upper_green = np.array([100,255,100])

# Define the masked area
mask = cv2.inRange(image, lower_green, upper_green)

# Mask the image to let the car show through
masked_image = np.copy(image)

masked_image[mask != 0] = [0, 0, 0]

# Display it!
plt.imshow(masked_image)

# Convert to HSV
hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

# HSV channels
h = hsv[:,:,0]
s = hsv[:,:,1]
v = hsv[:,:,2]

# Visualize the individual color channels
f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20,10))
ax1.set_title('H channel')
ax1.imshow(h, cmap='gray')
ax2.set_title('S channel')
ax2.imshow(s, cmap='gray')
ax3.set_title('V channel')
ax3.imshow(v, cmap='gray')

# Define our color selection boundaries in HSV values

## TODO: Change these thresholds
# This initial threshold allows a certain low range for Hue (H)
lower_hue = np.array([10,0,0]) 
upper_hue = np.array([180,255,255])

# Define the masked area
mask = cv2.inRange(image, lower_hue, upper_hue)

# Mask the image to let the car show through
masked_image = np.copy(image)

masked_image[mask != 0] = [0, 0, 0]

# Display it!
plt.imshow(masked_image)




















