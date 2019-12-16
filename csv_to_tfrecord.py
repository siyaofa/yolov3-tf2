
import pandas as pd
import tensorflow as tf
import os
import cv2 as cv
import random


def create_tf_example(filename, features, label):
	img = cv.imread(filename)
	height = img.shape[0]
	width = img.shape[1]
	xmins = [features[0] / width]
	ymins = [features[1] / height]
	xmaxs = [features[2] / width]
	ymaxs = [features[3] / height]
	image = open(filename, 'rb').read()
	tf_example = tf.train.Example(features=tf.train.Features(feature={
		'image/width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
		'image/height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
		'image/filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[filename.encode('utf-8')])),
		'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
		'image/object/bbox/xmin': tf.train.Feature(float_list=tf.train.FloatList(value=xmins)),
		'image/object/bbox/ymin': tf.train.Feature(float_list=tf.train.FloatList(value=ymins)),
		'image/object/bbox/xmax': tf.train.Feature(float_list=tf.train.FloatList(value=xmaxs)),
		'image/object/bbox/ymax': tf.train.Feature(float_list=tf.train.FloatList(value=ymaxs)),
		'image/object/class/text': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label.encode('utf-8')])),
		'image/object/class/label': tf.train.Feature(int64_list=tf.train.Int64List(value=[0])),
	}))
	return tf_example


def create_tf_example(image_filename,filename_box_labels):
	
	img = cv.imread(image_filename)
	height = img.shape[0]
	width = img.shape[1]
	image = open(image_filename, 'rb').read()
	filenames=[]
	xmins = []
	ymins = []
	xmaxs = []
	ymaxs = []
	labels=[]
	for filename_box_label in filename_box_labels:
		(filename, xmin, ymin, xmax, ymax, label)=filename_box_label
		filenames.append(filename.encode('utf-8'))
		xmins.append(xmin / width)
		ymins.append(ymin / height)
		xmaxs.append(xmax / width)
		ymaxs.append(ymax / height)
		labels.append(label.encode('utf-8'))

	tf_example = tf.train.Example(features=tf.train.Features(feature={
		'image/width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
		'image/height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
		#'image/filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=filenames)),
		'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
		'image/object/bbox/xmin': tf.train.Feature(float_list=tf.train.FloatList(value=xmins)),
		'image/object/bbox/ymin': tf.train.Feature(float_list=tf.train.FloatList(value=ymins)),
		'image/object/bbox/xmax': tf.train.Feature(float_list=tf.train.FloatList(value=xmaxs)),
		'image/object/bbox/ymax': tf.train.Feature(float_list=tf.train.FloatList(value=ymaxs)),
		'image/object/class/text': tf.train.Feature(bytes_list=tf.train.BytesList(value=labels)),
		'image/object/class/label': tf.train.Feature(int64_list=tf.train.Int64List(value=[0])),
	}))
	return tf_example


image_dict = {}


def parse_csv(csv_filename):
	image_path = os.path.split(csv_filename)[0]
	print(image_path)
	line_num = 0
	csv = pd.read_csv(csv_filename).values

	for row in csv:
		line_num += 1
		if(line_num > 1) and len(row) > 0:
			filename, xmin, ymin, xmax, ymax, label = row[0], row[1], row[2], row[3], row[4], row[5]
			filename_box_label = (filename, xmin, ymin, xmax, ymax, label)
			image = os.path.join(image_path, filename)
			if image in image_dict:
				# print(image,'in')
				image_dict[image].append(filename_box_label)
			else:
				#print(image,'not in')
				image_dict[image] = [filename_box_label]
			# example = create_tf_example(filename=os.path.join(image_path,image),features=features, label=label)
			
	return image_dict





csv_list = [r'D:\VoTT\vott-csv-export\cube-videos-export.csv',
r'D:\Github\siyaofa\yolov3-tf2\data\vott-csv-export\cube-export.csv']

image_with_label = {}

for csv_file in csv_list:
	image_dict = parse_csv(csv_file)
	#print(len(image_dict ))

train_writer=tf.io.TFRecordWriter("./data/train.tfrecords")
val_writer=tf.io.TFRecordWriter("./data/val.tfrecords")

for key in image_dict.keys():
	print(key, 'has', len(image_dict[key]))
	example =create_tf_example(key,image_dict[key])

	is_train=False

	if(random.random()>0.2):
		is_train=True
	else:
		is_train=False
		
	print(is_train)
	if(is_train):
		train_writer.write(example.SerializeToString())
	else:
		val_writer.write(example.SerializeToString())

val_writer.close()
train_writer.close()
