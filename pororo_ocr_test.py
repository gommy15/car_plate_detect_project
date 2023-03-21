import cv2

img = cv2.imread("./detections/crop/cars/carPlate_.png")
shape = img.shape
print(img.shape)
print(img.size)