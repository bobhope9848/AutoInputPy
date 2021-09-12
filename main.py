import cv2
img = cv2.imread("video001.bmp")
edges = cv2.Canny(img, 100, 200)
cv2.imshow("image", edges)
cv2.waitKey(0)
