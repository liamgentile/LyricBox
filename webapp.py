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
'''
Warning: I made an effort to remove potentially offensive language from the vocabulary of the models, but you may still encounter some unsavoury outputs.  
'''


# Constants for model URLs and tokenizer paths
MODEL_URLS = {
    'folk': 'https://lyricbox.s3.us-east-2.amazonaws.com/models/folk_lyrics_RNN_model4.h5',
    'pop': 'https://lyricbox.s3.us-east-2.amazonaws.com/models/pop_lyric_model.h5',
    'hip hop': 'https://lyricbox.s3.us-east-2.amazonaws.com/models/rap_lyric_model.h5'
}

TOKENIZER_PATHS = {
    'folk': 'web_application/folk_tokenizer.pkl',
    'pop': 'web_application/pop_tokenizer.pkl',
    'hip hop': 'web_application/rap_tokenizer.pkl'
}

models = {}
tokenizers = {}

for genre, model_url in MODEL_URLS.items():
    model_file = get_file(f'{genre}_m', model_url)
    models[genre] = load_model(model_file, compile=False)

    with open(TOKENIZER_PATHS[genre], 'rb') as f:
        tokenizers[genre] = pickle.load(f)

prompt = st.text_input('Type your prompt here.')
word_count_options = [5, 10, 15, 20, 25]
word_count = st.selectbox("How many words do you want to generate?", word_count_options)
genre_options = ['folk', 'pop', 'hip hop']
genre = st.selectbox("Which genre do you want to stylize your idea generator?", genre_options)

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

GENRE_CLASSES = {
    'folk': 29826,
    'pop': 26262,
    'hip hop': 47324
}

if st.button("Generate"):
    if prompt:  
        model = models[genre]
        tokenizer = tokenizers[genre]
        number_of_classes = GENRE_CLASSES[genre]

        try:
            generated_text = generate_text(prompt, word_count, model, tokenizer, number_of_classes)
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
