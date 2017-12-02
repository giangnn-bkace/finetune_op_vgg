from keras.models import load_model
from nngmodel import get_testing_model
from keras.callbacks import TensorBoard
from keras import optimizers
import nngutil
import time
import argparse

parse = argparse.ArgumentParser()
parse.add_argument("-model", default="step1.h5", help="path to h5 model file of fine-tuned model by step 1 h5")
parse.add_argument("-data", default="train_feature_extracting_layer_data", help="path to data folder")
parse.add_argument("-test", default="test_data", help="path to test data folder")
parse.add_argument("-save", default="step2.h5", help="path to h5 file to save fine-tuned weights")
args = parse.parse_args()

tensorboard = TensorBoard(log_dir='step2_logs\{}'.format(time.time()))

num_train_samples = 2173
num_test_samples = 21
num_iteration = 100

IMG_WIDTH = 224
IMG_HEIGHT = 224
BATCH_SIZE = 1
CLASS_MODE = 'binary'


lock = {'op10_conv4_1', 'op10_conv4_2', 'op10_conv4_3_CPM', 'op10_conv4_4_CPM', 'predictions'}

from_op = dict()
from_op['op10_conv1_1'] = 'conv1_1'
from_op['op10_conv1_2'] = 'conv1_2'
from_op['op10_conv2_1'] = 'conv2_1'
from_op['op10_conv2_2'] = 'conv2_2'
from_op['op10_conv3_1'] = 'conv3_1'
from_op['op10_conv3_2'] = 'conv3_2'
from_op['op10_conv3_3'] = 'conv3_3'
from_op['op10_conv3_4'] = 'conv3_4'
from_op['op10_conv4_1'] = 'conv4_1'
from_op['op10_conv4_2'] = 'conv4_2'
from_op['op10_conv4_3_CPM'] = 'conv4_3_CPM'
from_op['op10_conv4_4_CPM'] = 'conv4_4_CPM'


	
if __name__ == "__main__":
	# prepare training data
	train_data = nngutil.prepare_data(args.data, IMG_WIDTH, IMG_HEIGHT, BATCH_SIZE, CLASS_MODE)
	test_data = nngutil.prepare_data(args.test, IMG_WIDTH, IMG_HEIGHT, BATCH_SIZE, CLASS_MODE, shuffle=False)
	
	# load model
	model = load_model(args.model)
	# transfer weights from trained open pose model to the new model
	for layer in model.layers:
		layer.trainable = True
		if layer.name in lock:
			layer.trainable = False
			print(layer.name)
	
	# define training mechanism
	model.compile(loss='binary_crossentropy', optimizer=optimizers.SGD(lr=1e-4, momentum=0.9), metrics=['accuracy'])
	score = model.evaluate_generator(test_data, steps=131)
	print(score)
	# train classifier layers
	model.fit_generator(train_data, steps_per_epoch=num_train_samples, epochs=num_iteration, callbacks=[tensorboard], validation_data=test_data, validation_steps=21)
	
	#save model's weights to file
	model.save(args.save)
	#tensorboard.set_model(model)
	pred = model.predict_generator(test_data, steps=num_test_samples)
	print(pred)
	score = model.evaluate_generator(test_data, steps=num_test_samples)
	print(score)