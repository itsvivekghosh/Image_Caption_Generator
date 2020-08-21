from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.applications.xception import Xception
from keras.models import load_model
from pickle import load
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import argparse


class CaptionImage(object):

	def __init__(self, img_path, model, xception_model, tokenizer):
		self.img_path = img_path
		self.max_length = 32
		self.model = model
		self.xception_model = xception_model
		self.tokenizer = tokenizer


	def __del__(self):
		pass


	def extract_features(self):
	    try:
	        image = Image.open(self.img_path)
	    except:
	        print("ERROR: Couldn't open image! Make sure the image path and extension is correct")

	    image = image.resize((299, 299))
	    image = np.array(image)

	    # for images that has 4 channels, we convert them into 3 channels
	    if image.shape[2] == 4: 
	        image = image[..., :3]

	    image = np.expand_dims(image, axis=0)
	    image = image / 127.5
	    image = image - 1.0

	    ### Extracting Features from Xception Model
	    feature = self.xception_model.predict(image)

	    ## Returning  the features, len of features ==> (2048)
	    return feature


	def word_for_id(self, integer):

		for word, index in self.tokenizer.word_index.items():
			if index == integer:
				return word
		return None


	def generate_desc(self, photo):

	    in_text = str('start')
	    for i in range(self.max_length):

	        sequence = self.tokenizer.texts_to_sequences([in_text])[0]
	        # print(sequence)
	        sequence = pad_sequences([sequence], maxlen=self.max_length)
	        # print(sequence)

	        pred = self.model.predict([photo, sequence], verbose=0)
	        # print(pred)

	        pred = np.argmax(pred)
	        # print(pred)

	        word = self.word_for_id(pred)
	        # print(word)

	        if word == 'end':
	        	break

	        if word is None:
	            break

	        in_text += ' ' + word

	    return in_text


def caption_image(img_path = 'samples/football2.png'):

	## If no path is given by the User then: Default Path is->
	if img_path == None:
		img_path = 'samples/football2.png'

	## Loading Models and Tokenizer
	tokenizer = load(open("models/tokenizer.p", "rb")) ## pickle Load
	model = load_model('models/model_9.h5') ## Loads .h5 Model
	xception_model = Xception(include_top = False, pooling = "avg") ## Loading Xception Model

	## Loading Class. CaptionImage..
	caption = CaptionImage(
		img_path = img_path,
		model = model, 
		xception_model = xception_model, 
		tokenizer = tokenizer
	)

	## Extracting Required Feautres from the image like. colors, objects, surroundings, etc.
	photo = caption.extract_features()
	img = Image.open(img_path)
	
	## Searchine the description of the Image
	description_ = caption.generate_desc(photo)
	description = list(description_.split(" "))
	description = description[1:-2]
	description = ' '.join([str(s) for s in description])

	## returning Results
	return description


def main():
	caption_image()


if __name__ == '__main__':

	## Calling Main()
	main()