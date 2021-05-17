import streamlit as st 
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras import Model
import pickle
import zipfile
import tempfile
import s3fs


'''
# LyricBox
'''
'''
### A neural network powered idea generator for artists with writer's block.
'''

'''
#### Notes:
	
	- Avoid punctuation and capitalization in your prompt.
	eg. write "i cant help it" not "I can't help it."

	- I made an effort to remove potentially offensive 
	  language from the vocabulary of the models. If, however, 
	  something unsavoury is generated, this is simply by chance 
	  and is not a view of the creator. 
'''
prompt = st.text_input('Type your prompt here.')

word_count_options = [5, 10, 15, 20, 25]
word_count = st.selectbox("How many words do you want to generate?", word_count_options)

genre_options = ['folk', 'pop', 'hip hop']
genres = st.selectbox("Which genre do you want to stylize your idea generator?", genre_options)


from tensorflow.python.lib.io import file_io
folk_file = file_io.FileIO('https://storage.googleapis.com/lyricbox/webapp/models/folk_lyrics_RNN_model4.h5', mode='rb')
pop_file = file_io.FileIO('https://storage.googleapis.com/lyricbox/webapp/models/pop_lyric_model.h5', mode='rb')
hiphop_file = file_io.FileIO('https://storage.googleapis.com/lyricbox/webapp/models/rap_lyric_model.h5', mode='rb')
  
temp_model_location_f = './folk_lyrics_RNN_model4.h5'
temp_model_file_f = open(temp_model_location_f, 'wb')
temp_model_file_f.write(folk_file.read())
temp_model_file_f.close()
folk_file.close()  
folk_model = load_model(temp_model_location)

temp_model_location_p = './pop_lyric_model.h5'
temp_model_file_p = open(temp_model_location_p, 'wb')
temp_model_file_p.write(pop_file.read())
temp_model_file_p.close()
pop_file.close()  
pop_model = load_model(temp_model_location)

temp_model_location_h = './rap_lyric_model.h5'
temp_model_file_h = open(temp_model_location_h, 'wb')
temp_model_file_h.write(hiphop_file.read())
temp_model_file_h.close()
hiphop_file.close()  
hiphop_model = load_model(temp_model_location)


#tokenizer_folk import
tokenizer_folk = pickle.load(s3.open('s3://lyricbox/tokenizers/folk_tokenizer.pkl','rb'))
#tokenizer_pop import
tokenizer_pop = pickle.load(s3.open('s3://lyricbox/tokenizers/pop_tokenizer.pkl','rb'))
#tokenizer_hiphop import
tokenizer_hiphop = pickle.load(s3.open('s3://lyricbox/tokenizers/rap_tokenizer.pkl','rb'))



def folk_generate_text(prompt, word_count, folk_model):

    # process for the model
    number_of_classes_folk = 29826
    processed_phrase = tokenizer_folk.texts_to_sequences([prompt])[0]
    for i in range(word_count):
      network_input = np.array(processed_phrase[-(len(processed_phrase)):], dtype=np.float32)
      network_input = network_input.reshape((1, (len(processed_phrase)))) 

      # the RNN gives the probability of each word as the next one
      predict_proba = folk_model.predict(network_input)[0] 
      
      # sample one word using these chances
      predicted_index = np.random.choice(number_of_classes_folk, 1, p=predict_proba)[0]

      # add new index at the end of our list
      processed_phrase.append(predicted_index)
      

  # indices mapped to words - the method expects a list of lists so we need the extra bracket
      output_phrase = tokenizer_folk.sequences_to_texts([processed_phrase])[0]

    return output_phrase



def pop_generate_text(prompt, word_count, pop_model):

	   # process for the model
    number_of_classes_pop = 26262
    processed_phrase = tokenizer_pop.texts_to_sequences([prompt])[0]
    for i in range(word_count):
      network_input = np.array(processed_phrase[-(len(processed_phrase)):], dtype=np.float32)
      network_input = network_input.reshape((1, (len(processed_phrase)))) 

      # the RNN gives the probability of each word as the next one
      predict_proba = pop_model.predict(network_input)[0] 
      
      # sample one word using these chances
      predicted_index = np.random.choice(number_of_classes_pop, 1, p=predict_proba)[0]

      # add new index at the end of our list
      processed_phrase.append(predicted_index)
      

  # indices mapped to words - the method expects a list of lists so we need the extra bracket
      output_phrase = tokenizer_pop.sequences_to_texts([processed_phrase])[0]

    return output_phrase



def hiphop_generate_text(prompt, word_count, hiphop_model):

	   # process for the model
    number_of_classes_hiphop = 47324
    processed_phrase = tokenizer_hiphop.texts_to_sequences([prompt])[0]
    for i in range(word_count):
      network_input = np.array(processed_phrase[-(len(processed_phrase)):], dtype=np.float32)
      network_input = network_input.reshape((1, (len(processed_phrase)))) 

      # the RNN gives the probability of each word as the next one
      predict_proba = hiphop_model.predict(network_input)[0] 
      
      # sample one word using these chances
      predicted_index = np.random.choice(number_of_classes_hiphop, 1, p=predict_proba)[0]

      # add new index at the end of our list
      processed_phrase.append(predicted_index)
      

  # indices mapped to words - the method expects a list of lists so we need the extra bracket
      output_phrase = tokenizer_hiphop.sequences_to_texts([processed_phrase])[0]

    return output_phrase


if st.button("Generate"):
	if genres == 'folk':
		generated_text = folk_generate_text(prompt, word_count, folk_model)
	if genres == 'pop':
		generated_text = pop_generate_text(prompt, word_count, pop_model)

	if genres == 'hip hop':
		generated_text = hiphop_generate_text(prompt, word_count, hiphop_model)

	if "nigg" not in generated_text:
		try:
			st.write(generated_text)
		except:
			raise ValueError('I think you may have input a word that is not in the vocabulary. Please try again with a different prompt.')
	else:
		st.write("Please try again.")
		
'''
-----------
'''
'''
-----------
'''

'''
#### LyricBox is a project by Liam Gentile, a Toronto based data scientist. 
##### If you have any questions or comments about this project, please contact me at liam.gentile@mail.mcgill.ca.
##### You can also find out more about this project from the Github page: https://github.com/liamgentile/LyricBox/blob/main/webapp.py.
'''

