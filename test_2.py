import easyocr
import pytesseract
import cv2
import os
import numpy as np
import time

start = time.time()

# If you don't have tesseract executable in your PATH, include the following:
# pytesseract.pytesseract.tesseract_cmd = r'<full_path_to_your_tesseract_executable>'
# Example tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract'

# point to license plate image (works well with custom crop function)
gray = cv2.imread("./detections/crop/car/test5.png", 0)
gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
gray = cv2.medianBlur(gray, 3)
# perform otsu thresh (using binary inverse since opencv contours work better with white text)
ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
#cv2.imshow("Otsu", thresh)
#cv2.waitKey(0)
rect_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

# apply dilation
dilation = cv2.dilate(thresh, rect_kern, iterations=1)
# cv2.imshow("dilation", dilation)
# cv2.waitKey(0)
# find contours
try:
    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
except:
    ret_img, contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
sorted_contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])

# create copy of image
im2 = gray.copy()

plate_num = ""
# loop through contours and find letters in license plate
reader = easyocr.Reader(['ko'])
total_box = []
for cnt in sorted_contours:
    box = []
    x, y, w, h = cv2.boundingRect(cnt)
    height, width = im2.shape

    # if height of box is not a quarter of total height then skip
    if height / float(h) > 10: continue
    ratio = h / float(w)
    # if height to width ratio is less than 1.5 skip
    if ratio < 1.0: continue
    area = h * w
    # if width is not more than 25 pixels skip
    if width / float(w) > 20: continue
    # if area is less than 100 pixels skip
    if area < 50: continue
    box.append(x)
    box.append(y)
    box.append(w)
    box.append(h)
    total_box.append(box)

print(total_box)
x = total_box[0][0]
y = total_box[0][1]
w = total_box[2][0] + total_box[2][2]
h = total_box[2][1] + total_box[2][3]

rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2)
#cv2.imshow("Character's Segmented", im2)
#cv2.waitKey(0)
roi = thresh[y - 1:y + h + 1, x - 1:x + w + 1]
roi = cv2.bitwise_not(roi)
roi = cv2.medianBlur(roi, 5)
easy_text = reader.readtext(roi)
print(easy_text)
cv2.imshow("ROI", roi)
cv2.waitKey(0)

#print(easy_text)

plate_num += easy_text[0][1]

x = total_box[3][0]
y = total_box[3][1]
w = total_box[-1][0] + total_box[-1][2]
h = total_box[-1][1] + total_box[-1][3]

rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2)
cv2.imshow("Character's Segmented", im2)
#cv2.waitKey(0)
roi = thresh[y - 1:y + h + 1, x - 1:x + w + 1]
roi = cv2.bitwise_not(roi)
roi = cv2.medianBlur(roi, 5)
easy_text = reader.readtext(roi)
cv2.imshow("ROI", roi)
cv2.waitKey(0)

print(easy_text)

plate_num += easy_text[0][1]

print(plate_num)

print(time.time() - start)

'''
    # draw the rectangle
    rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow("ROI", rect)
    cv2.waitKey(0)
    roi = thresh[y - 10:y + h + 10, x - 10:x + w + 10]
    roi = cv2.bitwise_not(roi)
    roi = cv2.medianBlur(roi, 5)

    #text = pytesseract.image_to_string(roi, config='-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 8 --oem 3')
    #reader = easyocr.Reader(['ko'])
    easy_text = reader.readtext(roi)
    #cv2.imshow("ROI", roi)
    #cv2.waitKey(0)
    #easy_text = reader.readtext(roi)

    print(easy_text)

    try:
        plate_num += easy_text[0][1]
    except:
        pass

    #plate_num += text
print(plate_num)
print("time : ", time.time() - start)
cv2.imshow("Character's Segmented", im2)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''