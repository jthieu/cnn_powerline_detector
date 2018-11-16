import os
import cv2
import numpy as np
from tqdm import tqdm

from skimage import io

dir_name = "/Users/jamesthieu23/Documents/Classes/CS221/cs221_project/Power_Line_Database/Visible_Light_VL/"
directory = os.fsencode("/Users/jamesthieu23/Documents/Classes/CS221/cs221_project/Power_Line_Database/Visible_Light_VL")

labels = []

f = open("powerline_labels.txt","w+")

stacked_images = None

for file in tqdm(os.listdir(directory)):
    filename = os.fsdecode(file)
    if filename.endswith(".jpg"):
        image = io.imread(dir_name + filename, as_grey=True)
        if stacked_images is None:
            stacked_images = image
        else:
            stacked_images = np.vstack((stacked_images, image))
        # f.write(filename) # Based on this, it goes in a semi-random order based on the 
        if filename[:2] == "TV": # There is actually a powerline in this image
            labels.append(1)
        elif filename[:2] == "TY": # There is no powerline in this image
            labels.append(0)

np.save("powerlines", stacked_images)

f.write(str(labels))
f.close()