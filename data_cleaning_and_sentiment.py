import re
import os

import plotly.graph_objects as go

f = open('/home/oscar/newsletter/spotify/data/US/lyrics/7 rings.txt', 'r') # 'rb' read + binary
g = open('/home/oscar/newsletter/spotify/data/US/lyrics/505.txt', 'r') # 'rb' read + binary


all_words_f = ''
all_words_g = ''

for sentence in f.readlines():
    if 'Translations' in sentence:
        continue
    else:
        all_words_f += sentence

print(all_words_f)



for sentence in g.readlines():
    if 'Translations' in sentence:
        continue
    else:
        all_words_g += sentence
         
all_words_f = re.sub(r'[\(\[].*?[\)\]]', '', all_words_f) # remove identifiers like chorus, verse, etc...
all_words_g = re.sub(r'[\(\[].*?[\)\]]', '', all_words_g) # remove identifiers like chorus, verse, etc...

all_words_f = os.linesep.join([s for s in all_words_f.splitlines() if s]) # remove empty lines
all_words_g = os.linesep.join([s for s in all_words_g.splitlines() if s]) # remove empty lines

all_words_f = all_words_f.replace('\n', ' ') # replace the \n newline thingie's with empty space
all_words_g = all_words_g.replace('\n', ' ') # replace the \n newline thingie's with empty space

import pandas as pd
#import matplotly.pyplot as plt

# plot number of words per song
df = pd.DataFrame({'songname' : ('7 rings', '505'), 
                   'length': (len(all_words_f), len(all_words_g))})


fig = go.Figure(go.Bar(
            x=df['songname'],
            y=df['length']))

#fig.show()

### lexical richness

from nltk.corpus import stopwords

words_split_f = all_words_f.split(' ')

filtered_words_f = [word for word in words_split_f if word not in stopwords.words('english')
                  and len(word) > 3]

words_split_g = all_words_g.split(' ')

filtered_words_g = [word for word in words_split_g if word not in stopwords.words('english')
                  and len(word) > 3]

df2 = pd.DataFrame({'songname' : ('7 rings', '505'),
                    'lexicalrichness': (len(filtered_words_f), len(filtered_words_g))})

fig = go.Figure(go.Bar(
            x=df2['songname'],
            y=df2['lexicalrichness']))

# fig.show()

# Wordcloud shit

# from wordcloud import WordCloud
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS

### Standard

# Generate a word cloud image
wordcloud = WordCloud(max_font_size=40).generate(all_words_g)
plt.figure()
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
# plt.show()

### In a specific shape
import numpy as np
from PIL import Image



# read the mask image
india_mask = np.array(Image.open('/home/oscar/newsletter/spotify/other/india.jpg'))

stopwords = set(STOPWORDS)
stopwords.add("said")

wc = WordCloud(background_color="white", max_words=2000, mask=india_mask,
               stopwords=stopwords, contour_width=3, contour_color='steelblue').generate(all_words_f)


# # store to file
# wc.to_file(path.join(d, "alice.png"))

# show
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.figure()
plt.imshow(india_mask, cmap=plt.cm.gray, interpolation='bilinear')
plt.axis("off")
# plt.show()

### Sentiment analysis --> https://huggingface.co/blog/sentiment-analysis-python

from transformers import pipeline

sentiment_pipeline = pipeline('sentiment-analysis')
data = ['good', 'bad', 'hello', 'god', 'evil']

print(sentiment_pipeline(data))
