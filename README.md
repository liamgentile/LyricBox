# LyricBox
A neural network powered idea generator for artists with writer's block.

You can find the web application here: https://lyric-box.herokuapp.com.

Here is a link to a short presentation describing my project and process at a conceptual level:
https://www.loom.com/share/87a8d7df933f40089cce8fb3e3341934

## Project Summary

### Premise

  The motivation behind this project started with wanting to explore the world of natural language processing. When doing research for this project I found many AI powered creative writing models to be impractical. I kept asking myself "do we really want AI to generate full works of art?" and "is the amount of computational power required really worth the typically low quality output?".  In looking for a more practical application of NLP in the space of art, the problem of writer's block popped into my mind. I wanted to investigate: can I use machine learning applied to past lyric data to generate ideas for an artist with writer's block? These would be short bursts of text generated in response to the user's prompt. 

  I decided to build three models, each trained on songs from their respective genres. This would allow for some stylistic customization of the idea generator. The three genres I chose are Folk, Pop, and Hip Hop.

### Data Collection

I collected lyric data from the website az-lyrics.com. I used a number of different webscraping techniques and packages, namely Selenium and Zyte.  

The data collection process was composed of two steps:

#### 1. Song Name Collection

To start, I made three lists of artists from their respective genres. I formatted the artist names according to how they were written in the AZ lyrics url. For example, Bob Dylan's song list can be found here: https://www.azlyrics.com/d/dylan.html. When looping through the artist lists and scraping the song names with Selenium, I switched web pages in a for loop by using .format on azlyrics.com/{}/{}.html. You can find a more detailed description of this process and the code in the file called SongNameCollection.ipynb. 

#### 2. Song Lyric Collection

For song lyric collection I utilised my dictionary of songs mapped to their artists, and after some further URL formatting, I looped through this dictionary in order to scrape the lyrics. I pulled in the lyrics line by line in order to have more flexibility when cleaning the data later on. I used Zyte's webscraping API, and pulled out the text body from the html. You can see a more in depth explanation of this process and the code in the folder called WebScraping Notebooks.

### Data Cleaning

There were a number of data quality issues. Some of the lyrics had to be scrapped entirely due to containing a large amount of non-lyric content (extra, unwanted text from the html). I also removed duplicates, covers, and remixes. At a line by line level I removed lines from songs where there was usage of potentially offensive language and where there was non lyrical content such as a title saying "Verse" or "Chorus". In preparation for the vectorization process I removed all punctuation and capitalization from the corpus. You can see this data cleaning process in the notebook entitled DataCleaning.ipynb.

### Modeling

For modeling I used Sequential RNNs with Tensorflow and Keras. I utilised Google Colab Pro with High RAM and GPU in order to run these computationally intensive models in a reasonable amount of time. My final models had three hidden layers, with two being LSTM layers, which very generally are neural network layers that work well with sequential or time series data (and hence text data). I found that a good neuron count for the hidden layers was 700-350-175, and a suitable number of epochs was around 250. You can find notebooks detailing the final models in the folder entitled Modeling. 

### Model Optimization

For optimizing the models, I had a number of different considerations. Some of the most important considerations were:
1. subjective quality of generate_text output
2. computation times
3. accuracy scores (although this could not be trusted always)

Some of the most important parameters that I tweaked to improve model performance were:
1. window length 
2. structure of arrays 
3. Model Width (Neurons)
4. Model Length (Epochs)

### Reflections / How could this project be improved?

This project could be improved with more data, more computational power, and more time. I intentionally used less data than I could have because of the very computationally intensive nature of neural networks. Because my project is an idea generator, I was not attached to the idea of creating perfect english outputs, but rather generally sensical, interesting ideas. I think it achieves this goal, but if I had more time I would spend more time on model optimization and collect more data. 

The environmental consequences of neural networks should not be overlooked, and this is a concern that I have with building even bigger models. 


## File Reference List

WebScrapingNotebooks
- contains the notebooks in which I scraped the song names and song lyrics from AZ-lyrics.com

DataCleaning.ipynb
- a notebook in which I did most of the data preprocessing for the three lyric datasets.

Modeling
- contains the notebooks in which I arranged and ran the final models for the three genres of lyrics. 

web_application
- folder containing the necessary files for running the streamlit/heroku web app, most notably webapp.py

