import numpy as np
import platform
from PIL import ImageFont, ImageDraw, Image
from matplotlib import pyplot as plt

import uuid
import json
import time
import cv2
import requests
import re
import access_key

def plt_imshow(title='image', img=None, figsize=(8, 5)):
    plt.figure(figsize=figsize)

    if type(img) == list:
        if type(title) == list:
            titles = title
        else:
            titles = []

            for i in range(len(img)):
                titles.append(title)

        for i in range(len(img)):
            if len(img[i].shape) <= 2:
                rgbImg = cv2.cvtColor(img[i], cv2.COLOR_GRAY2RGB)
            else:
                rgbImg = cv2.cvtColor(img[i], cv2.COLOR_BGR2RGB)

            plt.subplot(1, len(img), i + 1), plt.imshow(rgbImg)
            plt.title(titles[i])
            plt.xticks([]), plt.yticks([])

        plt.show()
    else:
        if len(img.shape) < 3:
            rgbImg = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            rgbImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        plt.imshow(rgbImg)
        plt.title(title)
        plt.xticks([]), plt.yticks([])
        plt.show()

def put_text(image, text, x, y, color=(0, 255, 0), font_size=22):
    if type(image) == np.ndarray:
        color_coverted = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(color_coverted)

    if platform.system() == 'Darwin':
        font = 'AppleGothic.ttf'
    elif platform.system() == 'Windows':
        font = 'malgun.ttf'

    image_font = ImageFont.truetype(font, font_size)
    font = ImageFont.load_default()
    draw = ImageDraw.Draw(image)

    draw.text((x, y), text, font=image_font, fill=color)

    numpy_image = np.array(image)
    opencv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)

    return opencv_image

path = './detections/crop/cars/carPlate_0.png'
files = [('file', open(path,'rb'))]

api_url = access_key.api_url
secret_key = access_key.secret_key

request_json = {'images': [{'format': 'png',
                            'name': 'demo'
                            }],
                'requestId': str(uuid.uuid4()),
                'version': 'V2',
                'lang': 'ko',
                'timestamp': int(round(time.time() * 1000))
                }

payload = {'message': json.dumps(request_json).encode('UTF-8')}

headers = {
    'X-OCR-SECRET': secret_key,
}

response = requests.request("POST", api_url, headers=headers, data=payload, files=files)
result = response.json()

img = cv2.imread(path)
roi_img = img.copy()
image_h, image_w, _ = img.shape

#fields = result['images'][0]['fields']
fields = [{'valueType': 'ALL', 'boundingPoly': {'vertices': [{'x': 46.0, 'y': 18.0}, {'x': 171.0, 'y': 31.0}, {'x': 164.0, 'y': 98.0}, {'x': 39.0, 'y': 85.0}]}, 'inferText': '35주', 'inferConfidence': 0.9999, 'type': 'NORMAL', 'lineBreak': False}, {'valueType': 'ALL', 'boundingPoly': {'vertices': [{'x': 184.0, 'y': 35.0}, {'x': 358.0, 'y': 51.0}, {'x': 352.0, 'y': 119.0}, {'x': 178.0, 'y': 103.0}]}, 'inferText': '6552', 'inferConfidence': 0.9997, 'type': 'NORMAL', 'lineBreak': True}, {'valueType': 'ALL', 'boundingPoly': {'vertices': [{'x': 143.0, 'y': 104.0}, {'x': 170.0, 'y': 107.0}, {'x': 170.0, 'y': 113.0}, {'x': 142.0, 'y': 111.0}]}, 'inferText': 'HAULT', 'inferConfidence': 0.9189, 'type': 'NORMAL', 'lineBreak': False}, {'valueType': 'ALL', 'boundingPoly': {'vertices': [{'x': 173.0, 'y': 107.0}, {'x': 219.0, 'y': 112.0}, {'x': 218.0, 'y': 119.0}, {'x': 172.0, 'y': 114.0}]}, 'inferText': 'SAMSUNG', 'inferConfidence': 0.9927, 'type': 'NORMAL', 'lineBreak': False}, {'valueType': 'ALL', 'boundingPoly': {'vertices': [{'x': 223.0, 'y': 113.0}, {'x': 262.0, 'y': 118.0}, {'x': 261.0, 'y': 125.0}, {'x': 222.0, 'y': 121.0}]}, 'inferText': 'MOTORS', 'inferConfidence': 0.9802, 'type': 'NORMAL', 'lineBreak': True}]

plate_num = ""
for field in fields:
    text = field['inferText']
    vertices_list = field['boundingPoly']['vertices']
    pts = [tuple(vertice.values()) for vertice in vertices_list]
    topLeft = [int(_) for _ in pts[0]]
    topRight = [int(_) for _ in pts[1]]
    bottomRight = [int(_) for _ in pts[2]]
    bottomLeft = [int(_) for _ in pts[3]]

    #cv2.rectangle(roi_img, tuple(topLeft), tuple(bottomRight), (50, 0, 255), 2)

    #roi_img = put_text(roi_img, text, topLeft[0], topLeft[1] - 10, font_size=30)

    #print(text)
    plate_num += text

plate_num = re.sub('[^0-9가-힣]', '', plate_num)
print(plate_num)
roi_img = put_text(roi_img, plate_num, 0, image_h-50, font_size=30)
cv2.imwrite('./detections/crop/cars/'+plate_num+'.png', roi_img)
cv2.imshow("ROI", roi_img)
cv2.waitKey(0)
#plt_imshow(["Original", "ROI"], [img, roi_img], figsize=(16, 10))