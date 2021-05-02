###########################################################################################
#######     SIW 2021: 1ST COMPETITION ON SCRIPT IDENTIFICATION IN THE WILD    #############
###########################################################################################

# This a code example to generate the file results_team.txt.
# This is an example in python code. You can use other languages to generate the results file.
#        Format: results.txt must be a txt file with the 55814 labels of the 55814 test samples presented in a single column 
# Please be sure that the results_team.txt file is correctly generated before to upload it to the Codalab platform


# Scripts Legend: Please use the following identifiers (integers from 0 to 12) for the different scripts
#
#0 = Arabic
#1 = Bangla
#2 = Gujrati
#3 = Gurmukhi
#4 = Hindi
#5 = Japanese
#6 = Kannada
#7 = Malayalam
#8 = Oriya
#9 = Roman
#10 = Tamil
#11 = Telugu
#12 = Thai

import numpy as np
import random
import cv2
import argparse
import time
from tqdm import tqdm
from keras import models

def load_image(img_path, input_size):
	image = cv2.imread(img_path)
	image = cv2.resize(image, input_size)
	image = image / 255.0

	image = image.reshape((1,image.shape[0],image.shape[1],3))
	return image

def create_submission(args):
	# There are 55814 samples. Note that the test set includes handwritten and printed images.
	# Path to the test set
	path = args["test_set_folder"]

	indexes = []
	with open("data_type_indexes.txt", "r") as fi:
		indexes = [int(i) for i in fi.readlines()]
	
	# Result will be stored as an array of integers (int from 0 to 12) starting with the label (script) associated to the sample 
	# sample000001.png and ending with sample055814.png.
	results=np.zeros(55814) 
	results_tasks=np.zeros(55814) 

	printed_model = models.load_model(args["handwritten_model_path"])
	handwritten_model = models.load_model(args["handwritten_model_path"])

	try:
		printed_input_size = printed_model.layers[0].input_shape[1]
	except:
		printed_input_size = printed_model.layers[0].input_shape[0][1]
	
	try:
		handwritten_input_size = handwritten_model.layers[0].input_shape[1]
	except:
		handwritten_input_size = handwritten_model.layers[0].input_shape[0][1]
	

	for ifile in tqdm(range(1,55815)): #loop between 1 and 55814    
		file_name='sample'+str(ifile).zfill(6)+'.png'         

		#Predict the script of the image (see the Scripts legend)
		if indexes[ifile-1] == 1:
			img = load_image(path+'/'+file_name, (printed_input_size, printed_input_size))
			label = np.argmax(printed_model.predict(img)[0])
		else:
			img = load_image(path+'/'+file_name, (handwritten_input_size, handwritten_input_size))
			label = np.argmax(handwritten_model.predict(img)[0])
			
		#store the label
		results[ifile-1]=label

	#Save the results into a txt file. Please, be sure that you save the results as a column vector of 55814 integers  
	np.savetxt(args["result_path"], results.astype(int),fmt="%s")

if __name__ == "__main__":
	ap = argparse.ArgumentParser()
	ap.add_argument("-t", "--test-set-folder", type=str, required=True,
		help="Path to input test set folder")
	ap.add_argument("-h", "--handwritten-model-path", type=str, required=True,
		help="Path to the saved handwritten model")
	ap.add_argument("-p", "--printed-model-path", type=str, required=True,
		help="Path to the saved printed model")
	ap.add_argument("-r", "--result-path", type=str, required=True,
		help="The destination path to save submission txt file")
	args = vars(ap.parse_args())

	create_submission(args)