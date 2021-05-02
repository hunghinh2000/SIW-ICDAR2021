import os
import csv
import cv2
import numpy as np
import time
import config
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def load_data(images_folder_path, input_size, batch_size):
	train_datagen = ImageDataGenerator(
		rescale=1./255, 
		zoom_range=[0.5,1.5], 
		width_shift_range=0.2,
		height_shift_range=0.2,
		brightness_range=[0.9, 1.2],
		rotation_range=20,
		validation_split=0.2)
	train_generator = train_datagen.flow_from_directory(
		images_folder_path, 
		target_size=(input_size, input_size), 
		batch_size=batch_size, 
		seed=42,
		subset='training')
	valid_generator = train_datagen.flow_from_directory(
		images_folder_path, 
		target_size=(input_size, input_size), 
		batch_size=batch_size,
		seed=42, 
		subset='validation')
	return train_generator, valid_generator


if __name__ == "__main__":
	load_data(f"./data/{config.DATA_TYPE}/", 64, 16)
