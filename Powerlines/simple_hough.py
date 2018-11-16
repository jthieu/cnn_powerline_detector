import os
import cv2
import numpy as np

dir_name = "/Users/jamesthieu23/Desktop/Power_Line_Database/Visible_Light_VL/"
directory = os.fsencode("/Users/jamesthieu23/Desktop/Power_Line_Database/Visible_Light_VL")
ret_dir = "/Users/jamesthieu23/Desktop/Powerlines/HoughImages/"

file_count = 0

positive = 0
negative = 0

all_positive = 0
all_negative = 0

false_positive = 0
false_negative = 0

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    rem_filename = filename[:-4]
    if filename.endswith(".jpg"):
        file_count += 1
        img = cv2.imread(dir_name + filename)
        new_filename = str(rem_filename) + "_hough.jpg"
        # print (new_filename)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray,20,150,apertureSize = 3)
        
        if filename[:2] == "TV":
            all_positive += 1
        elif filename[:2] == "TY":
            all_negative += 1

        lines = cv2.HoughLinesP(edges,rho = 1, theta = np.pi/180, threshold = 100,minLineLength = 100,maxLineGap = 50)
        if lines is not None:
            if filename[:2] == "TV":
                positive += 1
            elif filename[:2] == "TY":
                false_positive += 1
            for line in lines:   
                for x1,y1,x2,y2 in line:
                    cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)
        else:
            if filename[:2] == "TV":
                false_negative += 1
            elif filename[:2] == "TY":
                negative += 1
        cv2.imwrite(ret_dir + new_filename,img)

print ("Positives:", (positive+false_positive)/file_count)
print ("Positives Successfully Detected:", positive/all_positive)
print ("True Positives:", positive/(positive+false_positive))

print ("Negatives:", (negative+false_negative)/file_count)
print ("Negatives Successfully Detected:", negative/all_negative)
print ("True Negatives:", negative/(negative+false_negative))

# import cv2
# import numpy as np

# img = cv2.imread('pl_6.jpg')
# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# edges = cv2.Canny(gray,50,150,apertureSize = 5)

# lines = cv2.HoughLines(edges,1,np.pi/180,200)
# for rho,theta in lines[0]:
#     a = np.cos(theta)
#     b = np.sin(theta)
#     x0 = a*rho
#     y0 = b*rho
#     x1 = int(x0 + 1000*(-b))
#     y1 = int(y0 + 1000*(a))
#     x2 = int(x0 - 1000*(-b))
#     y2 = int(y0 - 1000*(a))

#     cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

# cv2.imwrite('pl_6_hough.jpg',img)