import streamlit as st 
import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
import keras.models 
import s3fs
import h5py

'''
# Lyric Box
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
word_count = st.selectbox("How many words do you want to generate?", word_count_options, 10)

genre_options = ['folk', 'pop', 'hip hop']
genres = st.selectbox("Which genre do you want to stylize your idea generator?", genre_options, 'folk pop blend')

s3 = s3fs.S3FileSystem()
# importing models from s3 bucket
folk_model = h5py.File(s3.open("s3://lyricbox/models/folk_lyrics_RNN_model4.h5", "rb"))
folk_model = h5py.File(s3.open("s3://lyricbox/models/pop_lyric_model.h5", "rb"))
folk_model = h5py.File(s3.open("s3://lyricbox/models/rap_lyric_model.h5", "rb"))


#tokenizer_folk import
with open('tokenizers/folk_tokenizer.pkl', 'rb') as handle:
    tokenizer_folk = pickle.load(handle)
#tokenizer_pop import
with open('tokenizers/pop_tokenizer.pkl', 'rb') as handle:
    tokenizer_pop = pickle.load(handle)
#tokenizer_hiphop import
with open('tokenizers/rap_tokenizer.pkl', 'rb') as handle:
    tokenizer_hiphop = pickle.load(handle)



def folk_generate_text(prompt, word_count, folk_model):

    # process for the model
    number_of_classes_folk = 29826
    processed_phrase = tokenizer_folk.texts_to_sequences([input_phrase])[0]
    for i in range(next_words):
      network_input = np.array(processed_phrase[-(len(processed_phrase)):], dtype=np.float32)
      network_input = network_input.reshape((1, (len(processed_phrase)))) 

      # the RNN gives the probability of each word as the next one
      predict_proba = model.predict(network_input)[0] 
      
      # sample one word using these chances
      predicted_index = np.random.choice(number_of_classes_folk, 1, p=predict_proba)[0]

      # add new index at the end of our list
      processed_phrase.append(predicted_index)
      

  # indices mapped to words - the method expects a list of lists so we need the extra bracket
      output_phrase = tokenizer.sequences_to_texts([processed_phrase])[0]

    return output_phrase



def pop_generate_text(prompt, word_count, pop):

	   # process for the model
    number_of_classes_pop = 26262
    processed_phrase = tokenizer_pop.texts_to_sequences([input_phrase])[0]
    for i in range(next_words):
      network_input = np.array(processed_phrase[-(len(processed_phrase)):], dtype=np.float32)
      network_input = network_input.reshape((1, (len(processed_phrase)))) 

      # the RNN gives the probability of each word as the next one
      predict_proba = model.predict(network_input)[0] 
      
      # sample one word using these chances
      predicted_index = np.random.choice(number_of_classes_pop, 1, p=predict_proba)[0]

      # add new index at the end of our list
      processed_phrase.append(predicted_index)
      

  # indices mapped to words - the method expects a list of lists so we need the extra bracket
      output_phrase = tokenizer.sequences_to_texts([processed_phrase])[0]

    return output_phrase



def hiphop_generate_text(prompt, word_count, hiphop):

	   # process for the model
    number_of_classes_hiphop = 47324
    processed_phrase = tokenizer_hiphop.texts_to_sequences([input_phrase])[0]
    for i in range(next_words):
      network_input = np.array(processed_phrase[-(len(processed_phrase)):], dtype=np.float32)
      network_input = network_input.reshape((1, (len(processed_phrase)))) 

      # the RNN gives the probability of each word as the next one
      predict_proba = model.predict(network_input)[0] 
      
      # sample one word using these chances
      predicted_index = np.random.choice(number_of_classes_hiphop, 1, p=predict_proba)[0]

      # add new index at the end of our list
      processed_phrase.append(predicted_index)
      

  # indices mapped to words - the method expects a list of lists so we need the extra bracket
      output_phrase = tokenizer.sequences_to_texts([processed_phrase])[0]

    return output_phrase


if st.button("Generate"):
	if genres == 'folk':
		generated_text = folk_generate_text(prompt, word_count, folk_model)
	if genres == 'pop':
		generated_text = pop_generate_text(prompt, word_count, pop_model)
		st.write(generated_text)
	if genres == 'hip hop':
		generated_text = hiphop_generate_text(prompt, word_count, hiphop_model)

  	if "nigg" not in generated_text:
    		try:
			st.write(generated_text)
    		except:
      			raise ValueError('I think you may have input a word that is not in the vocabulary. Please try again with a different prompt.')
  	else:
    		st.write("Please try again.")



