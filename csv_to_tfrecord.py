import pandas as pd
import tensorflow as tf
import os
import cv2 as cv
import random

def create_tf_example(filename,features, label):
    img=cv.imread(filename)
    height=img.shape[0]
    width=img.shape[1]
    xmins = [features[0] / width]    
    ymins = [features[1] / height]
    xmaxs = [features[2] / width]
    ymaxs = [features[3] / height]
    image=open(filename,'rb').read()
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/width':tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
        'image/height':tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
        'image/filename':tf.train.Feature(bytes_list=tf.train.BytesList(value=[filename.encode('utf-8')])),
        'image/encoded':tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
        'image/object/bbox/xmin':tf.train.Feature(float_list=tf.train.FloatList(value=xmins)),
        'image/object/bbox/ymin':tf.train.Feature(float_list=tf.train.FloatList(value=ymins)),
        'image/object/bbox/xmax':tf.train.Feature(float_list=tf.train.FloatList(value=xmaxs)),
        'image/object/bbox/ymax':tf.train.Feature(float_list=tf.train.FloatList(value=ymaxs)),
        'image/object/class/text':tf.train.Feature(bytes_list=tf.train.BytesList(value=[label.encode('utf-8')])),
        'image/object/class/label':tf.train.Feature(int64_list=tf.train.Int64List(value=[0])),
    }))
    return tf_example


image_path=r'./data/vott-csv-export'
csv_filename=r'cube-export.csv'
csv = pd.read_csv(os.path.join(image_path,csv_filename)).values

train_writer=tf.io.TFRecordWriter("./data/train.tfrecords")
val_writer=tf.io.TFRecordWriter("./data/val.tfrecords")

line_num=0
for row in csv:
    line_num+=1
    if(line_num>1) and len(row)>0:
        image, features, label = row[0],row[1:-1], row[-1]
        example = create_tf_example(filename=os.path.join(image_path,image),features=features, label=label)
        is_train=False

        if(random.random()>0.2):
            is_train=True
        else:
            is_train=False
        print(is_train)
        if(is_train) :
            train_writer.write(example.SerializeToString())
        else:
            val_writer.write(example.SerializeToString())

val_writer.close()
train_writer.close()