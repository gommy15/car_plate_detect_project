import os
# comment out below line to enable tensorflow outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app
import core.utils as utils
from core.yolov4 import filter_boxes
from core.functions import *
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np
import time
from threading import Thread
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

class VideoStream:
    """Camera object that controls video streaming from the WebCamera"""

    def __init__(self, resolution=(640, 480), framerate=30):
        # Initialize the PiCamera and the camera image stream
        self.stream = cv2.VideoCapture(0)
        #ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3, resolution[0])
        ret = self.stream.set(4, resolution[1])

        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()

        # Variable to control when the camera is stopped
        self.stopped = False

    def start(self):
        # Start the thread that reads frames from the video stream
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        # Return the most recent frame
        return self.frame

    def stop(self):
        # Indicate that the camera and thread should be stopped
        self.stopped = True


input_size = (416, 416)
images = './data/images/car.jpg'
output = './detections/'
source = './detections/stream.jpg'
stream_path = './stream.jpg'
plate_list = []

def main(_argv):
    start = time.time()
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)

    weights = './checkpoints/custom-416.tflite'
    videostream = VideoStream(framerate=30).start()
    time.sleep(1)

    interpreter = tf.lite.Interpreter(model_path=weights)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print(input_details)
    print(output_details)

    img_num = 0
    #plate_list = []

    plate_file = open("./plate_file.txt", 'w', encoding='UTF-8')
    plate_file.close()

    while True:
        frame1 = videostream.read()
        frame = frame1.copy()
        cv2.imwrite(source, frame)

        image_data = cv2.resize(frame, input_size)
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        start_time = time.time()

        end = time.time()
        print(f"-----------------first : {end - start:.5f} sec--------------------")
        start = end

        interpreter.set_tensor(input_details[0]['index'], image_data)
        interpreter.invoke()
        pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]

        try:
            boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                            input_shape=tf.constant([input_size, input_size]))

            boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
                boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
                scores=tf.reshape(
                    pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
                max_output_size_per_class=50,
                max_total_size=50,
                iou_threshold=0.45,
                score_threshold=0.50
            )

            end = time.time()
            print(f"-----------------second : {end - start:.5f} sec--------------------")
            start = end

            # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, xmax, ymax
            original_h, original_w, _ = frame.shape
            bboxes = utils.format_boxes(boxes.numpy()[0], original_h, original_w)

            pred_bbox = [bboxes, scores.numpy()[0], classes.numpy()[0], valid_detections.numpy()[0]]
            counts = dict()
            path = './detections/crop/cars/'

            # read in all class names from config
            class_names = utils.read_class_names(cfg.YOLO.CLASSES)

            # by default allow all classes in .names file
            allowed_classes = list(class_names.values())

            plate_file = open("./plate_file.txt", 'a', encoding='UTF-8')

            image, img_num, plate_list = utils.draw_bbox(frame, pred_bbox, plate_file, plate_list, path, img_num,
                                                         True, allowed_classes=allowed_classes,
                                                         read_plate=True)

            plate_file.close()

            end = time.time()
            print(f"-----------------third : {end - start:.5f} sec--------------------")
            start = end

            image = np.asarray(image)
            cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
            result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            cv2.imshow("result", result)
        except:
            cv2.imshow("result", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
