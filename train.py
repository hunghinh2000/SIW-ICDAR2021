from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn import metrics
from tensorflow.distribute import MirroredStrategy
import matplotlib.pyplot as plt
import models
import dataset
import numpy as np
import argparse
import locale
import os
import math
import argparse
import time

INPUT_SIZE_LIST = {"b0": 224, "b1": 240, "b2": 260, "b3": 300, "b4": 380, "b5": 456, "b6": 528, "b7": 600}

def save_trainging_plot(history, output_filename):
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'valid'], loc='upper left')
	plt.savefig(output_filename)

def train(args):
	model_base = args["model_base"]
	data_type = args["data_type"]
	weight_path = args["weight_path"]
	batch_size = args["batch_size"]
	learning_rate = args["learning_rate"]
	epochs = args["epochs"] 
	input_size = INPUT_SIZE_LIST[model_base]

	TRAIN_IMAGE_FOLDER_PATH = f"./data/{data_type}/"

	OUTPUT_MODEL_PATH = f"./saved_model/{data_type}_{model_base}_{str(time.time())}.h5"
	OUTPUT_TRAINING_FIG_PATH = f"./plot_figure/{data_type}_{model_base}_{str(time.time())}.jpg"

	print("[INFO] Loading and cropping all raw training receipt images...")

	train_generator, valid_generator = dataset.load_data(TRAIN_IMAGE_FOLDER_PATH, input_size, batch_size)

	NUM_CLASSES = len(train_generator.class_indices)

	# Create a MirroredStrategy.
	strategy = MirroredStrategy()
	print("Number of devices: {}".format(strategy.num_replicas_in_sync))
		
	mc = ModelCheckpoint(OUTPUT_MODEL_PATH, monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
	
	# Print the training config
	print(f"[INFO] Training config: \n\
		- DATA_TYPE: {data_type} \n\
		- MODEL_BASE: {model_base}\n\
		- INPUT_SIZE: {str(input_size)}x{str(input_size)}\n\
		- EPOCHS: {str(epochs)}\n\
		- BATCH_SIZE: {str(batch_size)}\n\
		- LEARNING_RATE: {str(learning_rate)}\n")

	if weight_path is None:
		with strategy.scope():
			model = models.create_efficientnet(input_size, input_size, 3, model_base, \
									args["first_frozen_layers"], NUM_CLASSES, learning_rate, epochs)
	else:
		print(f"[INFO] Continue training from {weight_path}")
		with strategy.scope():
			model = load_model(weight_path)

	# Train the model
	print("[INFO] Training model...")
	H = model.fit(train_generator, 
		validation_data=valid_generator,
		epochs=epochs, 
		callbacks=[mc])
	# Save Figure
	print(f"[INFO] Saving the training firgure...")
	save_trainging_plot(H, OUTPUT_TRAINING_FIG_PATH)

if __name__ == "__main__":
	ap = argparse.ArgumentParser()
	ap.add_argument("-m", "--model-base", type=str, required=True,
		help="Model base of EfficientNet")
	ap.add_argument("-t", "--data-type", type=str, required=True,
		help="Specify the type of training data (handwritten or printed)")
	ap.add_argument("-w", "--weight-path", type=str,
		help="Weights for the continue training purpose.", default=None)
	ap.add_argument("-b", "--batch-size", type=int,
		help="Batch size", default=16)
	ap.add_argument("-e", "--epochs", type=int,
		help="Epochs", default=100)
	ap.add_argument("-l", "--learning-rate", type=float,
		help="Learning rate", default=1e-4)
	ap.add_argument("-f", "--first-frozen-layers", type=int,
		help="The number of first layers of EfficientNet model to not be trained", default=-1)
	args = vars(ap.parse_args())
	train(args)