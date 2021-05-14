# LyricBox
A neural network powered idea generator for artists with writer's block.

The web application for people to use LyricBox runs through Google Colab and Ngrok and is built with Python and Streamlit. In order to save memory the models and tokenizers required for the app are loaded in from Google Drive.

You can find it here:
http://38e8c8a09f5a.ngrok.io

Here is a link to a short presentation describing my project and process at a conceptual level.

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

### Model Optimization

### How could this project be improved?



## File Reference List

LyricBoxWebApp.ipynb
- a notebook that runs the web app for LyricBox. Includes the web app code.

DataCleaning.ipynb
- a notebook in which I did most of the data preprocessing for the three lyric datasets.
