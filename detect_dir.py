import time
from absl import app, flags, logging
from absl.flags import FLAGS
import os
import cv2
import numpy as np
import tensorflow as tf
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import draw_outputs

flags.DEFINE_string('classes', './data/coco.names', 'path to classes file')
flags.DEFINE_string('weights', './checkpoints/yolov3.tf',
                    'path to weights file')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('image', './data/girl.png', 'path to input image')
flags.DEFINE_string('output', './output.jpg', 'path to output image')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')



def load_yolo():
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    if FLAGS.tiny:
        yolo = YoloV3Tiny(classes=FLAGS.num_classes)
    else:
        yolo = YoloV3(classes=FLAGS.num_classes)

    yolo.load_weights(FLAGS.weights)
    logging.info('weights loaded')

    
    return yolo

def test(yolo,file,inputs,outputs):
    input_file=os.path.join(inputs,file)
    output_file=os.path.join(outputs,file)
    img = tf.image.decode_image(open(input_file, 'rb').read(), channels=3)
    img = tf.expand_dims(img, 0)
    img = transform_images(img, FLAGS.size)

    t1 = time.time()
    boxes, scores, classes, nums = yolo(img)
    t2 = time.time()
    logging.info('time: {}'.format(t2 - t1))

    logging.info('detections:')
    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    for i in range(nums[0]):
        logging.info('\t{}, {}, {}'.format(class_names[int(classes[0][i])],
                                           np.array(scores[0][i]),
                                           np.array(boxes[0][i])))

    img = cv2.imread(input_file)
    img = cv2.resize(img,None,fx=0.1,fy=0.1)
    img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
    cv2.imwrite(output_file, img)
    logging.info('output saved to: {}'.format(output_file))


def main(_argv):
    
    yolo=load_yolo()

    inputs_path=r'./data/test_same_background'
    outputs_path=r'./data/outputs'

    filelist=os.listdir(inputs_path)

    for file in filelist:
        if file ==r'cube-export.csv':
            continue
        test(yolo,file,inputs_path,outputs_path)

    

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
