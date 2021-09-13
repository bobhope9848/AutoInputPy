import time

import cv2
path = "C:\\Users\\bobhope\\Documents\\youtube-dl\\New frames\\"
for n in range(37, 45):
    img = cv2.imread(path + f"video{n}.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img,(7, 7), 0)
    img = cv2.Canny(img, 100, 200)
    cv2.imshow("image", img)
    print(f"video{n}.png")
    cv2.waitKey(0)


