from absl import app, flags, logging
from absl.flags import FLAGS
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
    EarlyStopping,
    ModelCheckpoint,
    TensorBoard
)
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny, YoloLoss,
    yolo_anchors, yolo_anchor_masks,
    yolo_tiny_anchors, yolo_tiny_anchor_masks
)


flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_string('weights', '',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'image size')
flags.DEFINE_integer('batch_size', 8, 'batch size')
flags.DEFINE_float('learning_rate', 1e-3, 'learning rate')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')


def main(_argv):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    if FLAGS.tiny:
        model = YoloV3Tiny(FLAGS.size, training=True,
                           classes=FLAGS.num_classes)
    else:
        model = YoloV3(FLAGS.size, training=True, classes=FLAGS.num_classes)
    
    if len(FLAGS.weights)>0:
        model.load_weights(FLAGS.weights)
    
    export_dir='./model/yolov3_tiny/'
    tf.saved_model.save(model,export_dir)
    converter = tf.lite.TFLiteConverter.from_saved_model(export_dir)
    tflite_model = converter.convert()
    open("./model/yolov3_tiny_model.tflite", "wb").write(tflite_model)



if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
