# test file if you want to quickly try tesseract on a license plate image
import cv2
import easyocr
import numpy as np
import re
import matplotlib.pyplot as plt
from PIL import ImageFont, ImageDraw, Image
import time

start_time = time.time()

reader = easyocr.Reader(['ko'])
result = reader.readtext('./detections/crop/car/test2.png')

img = cv2.imread("./detections/crop/car/test2.png")
blur = cv2.GaussianBlur(img, (5,5), 0)
img = cv2.medianBlur(img, 3)

img = Image.fromarray(img)
font = ImageFont.truetype("malgun.ttf", 40)
draw = ImageDraw.Draw(img)

for i in result:
  x = i[0][0][0]
  y = i[0][0][1]
  w = i[0][1][0] - i[0][0][0]
  h = i[0][2][1] - i[0][1][1]

  draw.rectangle(((x, y), (x+w, y+h)), outline="blue", width=2)
  draw.text((int((x+x+w)/2), y-40), str(i[1]), font=font, fill="blue")
  #if(len(str(i[1])) > 4):
  if (len(str(i[1])) > 0):
    new_str = re.sub(r"[^\uAC00-\uD7A30-9a-zA-Z\s]", "", (i[1]))
    print(str(i[1]))

print("time : ", time.time() - start_time)
plt.imshow(img)
plt.show()
#print("time : ", time.time() - start_time)
#cv2.waitKey(0)
#cv2.destroyAllWindows()