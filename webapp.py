import streamlit as st 
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.python.keras.utils.data_utils import get_file
import pickle
import h5py

'''
# LyricBox
'''
'''
### A neural network powered idea generator for artists with writer's block.
'''

MODEL_URL = 'https://lyricbox.s3.us-east-2.amazonaws.com/models/folk_lyrics_RNN_model4.h5'
TOKENIZER_PATH = 'web_application/folk_tokenizer.pkl'

model_file = get_file('folk_m', MODEL_URL)
model = load_model(model_file, compile=False)

with open(TOKENIZER_PATH, 'rb') as f:
    tokenizer = pickle.load(f)

NUMBER_OF_CLASSES = 29826

prompt = st.text_input('Type your prompt here.')
WORD_COUNT = 20

def generate_text(prompt, word_count, model, tokenizer, number_of_classes):
    processed_phrase = tokenizer.texts_to_sequences([prompt])[0]
    output_phrase = prompt  # Start with the input prompt
    
    for _ in range(word_count):
        network_input = np.array(processed_phrase[-len(processed_phrase):], dtype=np.float32)
        network_input = network_input.reshape((1, len(processed_phrase)))

        # Get prediction probabilities and sample a word
        predict_proba = model.predict(network_input)[0]
        predicted_index = np.random.choice(number_of_classes, 1, p=predict_proba)[0]
        
        # Add the predicted word to the phrase
        processed_phrase.append(predicted_index)
        output_phrase = tokenizer.sequences_to_texts([processed_phrase])[0]

    return output_phrase

if st.button("Generate"):
    if prompt:  
        try:
            generated_text = generate_text(prompt, WORD_COUNT, model, tokenizer, NUMBER_OF_CLASSES)
            if "X" not in generated_text:
                st.write(generated_text)
            else:
                st.write("Please try again.")
        except Exception as e:
            st.write(f"Error generating text: {str(e)}")
    else:
        st.write("Please provide a prompt.")

'''
-----------
'''
'''
-----------
'''
'''
#### LyricBox is a project by Liam Gentile, a Toronto based software developer. 
##### If you have any questions or comments about this project, please contact me at liamanthonygentile@gmail.com.
##### You can also find out more about this project from the Github page: https://github.com/liamgentile/LyricBox.
'''
