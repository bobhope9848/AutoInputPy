#https://stackoverflow.com/questions/25125670/best-value-for-threshold-in-canny
import cv2
import numpy as np
import matplotlib.pyplot as plt


def callback(x):
    print(x)


for file in range(1, 45):
    img = cv2.imread(f'C:\\Users\\bobhope\\Documents\\youtube-dl\\New frames\\video{file}.png ', 0) #read image as grayscale
    img = cv2.GaussianBlur(img, (7,7), 0)

    canny = cv2.Canny(img, 85, 255)

    cv2.namedWindow('image') # make a window with name 'image'
    cv2.createTrackbar('L', 'image', 0, 255, callback) #lower threshold trackbar for window 'image
    cv2.createTrackbar('U', 'image', 0, 255, callback) #upper threshold trackbar for window 'image

    while(1):
        numpy_horizontal_concat = np.concatenate((img, canny), axis=1) # to display image side by side
        cv2.imshow('image', numpy_horizontal_concat)
        k = cv2.waitKey(1) & 0xFF
        if k == 27: #escape key
            break
        l = cv2.getTrackbarPos('L', 'image')
        u = cv2.getTrackbarPos('U', 'image')

        canny = cv2.Canny(img, l, u)


    cv2.destroyAllWindows()
