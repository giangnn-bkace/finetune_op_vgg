from model import get_testing_model, relu, conv, pooling
from keras.models import Model
from keras.layers import Flatten, Input, Lambda, Dense
from keras.preprocessing.image import ImageDataGenerator

IMG_WIDTH = 224
IMG_HEIGHT = 224

def op10_block(x, weight_decay):
	# Block 1
	x = conv(x, 64, 3, "op10_conv1_1", (weight_decay, 0))
	x = relu(x)
	x = conv(x, 64, 3, "op10_conv1_2", (weight_decay, 0))
	x = relu(x)
	x = pooling(x, 2, 2, "op10_pool1_1")

	# Block 2
	x = conv(x, 128, 3, "op10_conv2_1", (weight_decay, 0))
	x = relu(x)
	x = conv(x, 128, 3, "op10_conv2_2", (weight_decay, 0))
	x = relu(x)
	x = pooling(x, 2, 2, "op10_pool2_1")

	# Block 3
	x = conv(x, 256, 3, "op10_conv3_1", (weight_decay, 0))
	x = relu(x)
	x = conv(x, 256, 3, "op10_conv3_2", (weight_decay, 0))
	x = relu(x)
	x = conv(x, 256, 3, "op10_conv3_3", (weight_decay, 0))
	x = relu(x)
	x = conv(x, 256, 3, "op10_conv3_4", (weight_decay, 0))
	x = relu(x)
	x = pooling(x, 2, 2, "op10_pool3_1")

	# Block 4
	x = conv(x, 512, 3, "op10_conv4_1", (weight_decay, 0))
	x = relu(x)
	x = conv(x, 512, 3, "op10_conv4_2", (weight_decay, 0))
	x = relu(x)

	# Additional non vgg layers
	x = conv(x, 256, 3, "op10_conv4_3_CPM", (weight_decay, 0))
	x = relu(x)
	x = conv(x, 128, 3, "op10_conv4_4_CPM", (weight_decay, 0))
	x = relu(x)

	return x

def get_op10_model(weight_decay):
	img_input_shape = (IMG_WIDTH, IMG_HEIGHT, 3)
	img_input = Input(shape=img_input_shape)
	img_normalized = Lambda(lambda x: x /256 - 0.5, name="op10_lambda")(img_input) #[-0.5, 0.5]
	
	op10_part = op10_block(img_normalized, weight_decay)
	
	op10_model_flat = Flatten(name='flatten')(op10_part)
	op10_model_dense1 = Dense(1000, activation='relu', name='dense_1')(op10_model_flat)
	op10_model_dense2 = Dense(1000, activation='relu', name='dense_2')(op10_model_dense1)
	op10_model_dense3 = Dense(1000, activation='relu', name='dense_3')(op10_model_dense2)
	op10_model_out = Dense(1, activation='sigmoid', name='predictions')(op10_model_dense3)
	
	op10_model = Model(img_input, op10_model_out)
	return op10_model

def prepare_data(data_path, img_width=224, img_height=224, batch_size=10, class_mode='binary', shuffle=True):
	data_gen = ImageDataGenerator()
	data = data_gen.flow_from_directory(data_path, target_size=(img_width, img_height), batch_size=batch_size, class_mode=class_mode, shuffle=shuffle)
	
	return data