import config
from tensorflow import keras as K
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3, EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB7
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


def create_efficientnet(width, height, depth, model_base, 
						first_layers_to_freeze, num_classes, learning_rate, epochs):
	inputShape = (height, width, depth)

	inputs = K.Input(shape=inputShape)

	if model_base == "b0":
		effnet = EfficientNetB0(include_top=False, input_tensor=inputs, weights="imagenet")
	elif model_base == "b1":
		effnet = EfficientNetB1(include_top=False, input_tensor=inputs, weights="imagenet")
	elif model_base == "b2":
		effnet = EfficientNetB2(include_top=False, input_tensor=inputs, weights="imagenet")
	elif model_base == "b3":
		effnet = EfficientNetB3(include_top=False, input_tensor=inputs, weights="imagenet")
	elif model_base == "b4":
		effnet = EfficientNetB4(include_top=False, input_tensor=inputs, weights="imagenet")
	elif model_base == "b5":
		effnet = EfficientNetB5(include_top=False, input_tensor=inputs, weights="imagenet")
	elif model_base == "b6":
		effnet = EfficientNetB6(include_top=False, input_tensor=inputs, weights="imagenet")
	else:
		effnet = EfficientNetB7(include_top=False, input_tensor=inputs, weights="imagenet")

	# # Print architecture of effnet
	# for i, layer in enumerate(effnet.layers[:]):
	# 	print(i, layer.name, layer.output_shape)
	# print(f"Effnet len: {len(effnet.layers[:])}")
	
	# b0: 20; b2: 33; b4: 236; b6: 45; b7: 265
	for i, layer in enumerate(effnet.layers[:first_layers_to_freeze]):
		layer.trainable = False
	for i, layer in enumerate(effnet.layers[first_layers_to_freeze:]):
		if not isinstance(layer, K.layers.BatchNormalization):
			layer.trainable = True

	model = Sequential()
	model.add(effnet)
	model.add(K.layers.Dropout(0.25))
	model.add(K.layers.Dense(effnet.layers[-1].output_shape[3]))
	model.add(K.layers.LeakyReLU())
	model.add(K.layers.GlobalAveragePooling2D())
	model.add(K.layers.BatchNormalization())
	model.add(K.layers.Dropout(0.5))
	model.add(K.layers.Dense(num_classes, activation='softmax'))

	# Freeze the batchnorm layer of our model 
	for i, layer in enumerate(model.layers[:]):
		if isinstance(layer, K.layers.BatchNormalization):
			layer.trainable = False
	
	opt = Adam(lr=learning_rate, decay=learning_rate / epochs)
	model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
	
	model.summary()

	return model

if __name__ == "__main__":
	model = create_efficientnet(config.INPUT_SIZE, config.INPUT_SIZE, 3, config.MODEL_BASE, config.FIRST_LAYERS_TO_FREEZE, config.NUM_CLASSES)
	#model.summary()