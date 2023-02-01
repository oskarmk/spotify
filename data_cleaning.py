# standard imports
import os
import re
import tqdm
from typing import List

#third-party import
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import plotly.graph_objects as go
from wordcloud import WordCloud, STOPWORDS

rootdir = '/home/oscar/newsletter/spotify/data2ndpost100'

def create_country_files(rootdir: str):
    '''This function takes all the subfolders with the lyrics files from the
    respective country and joins them togethere into one full_lyrics.txt file'''

    # initialize list of strings to populate with the translated lyrics
    all_countries = {'year': [], 'lyrics': [], 'wordcount': [], 'lexrich': []}

    # loop over all the lyrics files in the countries folders
    for subdir, dirs, files in os.walk(rootdir):

        if 'lyrics' in subdir:
            full = ''

            for file in files:
                print(f'Joining {file}')
                content_clean = ''
                f = open(os.path.join(subdir, file), 'r')
                for sentence in f.readlines():
                    if 'Translations' in sentence or 'Lyrics' in sentence:
                        continue
                    else:
                        content_clean += sentence
                        
                full = full + '\n' + content_clean

                f.close()

                full = re.sub(r'[\(\[].*?[\)\]]', '', full) # remove identifiers like chorus, verse, etc...
                full = os.linesep.join([s for s in full.splitlines() if s]) # remove empty lines
                full = full.replace('\n', ' ') # replace the \n newline thingie's with empty space

            # write file in original language
            text_file = open(os.path.join(subdir[:-7], 'full_lyrics_clean.txt'), 'w')
            text_file.write(full)
            text_file.close()

            # append translation to the all countries file
            all_countries['year'].append(subdir[-12:-7])
            all_countries['lyrics'].append(full)
            all_countries['wordcount'].append(round(len(full.split(' ')) / len(files), 0))
            
            # filter out 'stopwords' and words shorter than 2 letters long
            filtered_words_f = [word for word in full.split(' ') if word not in stopwords.words('english')
                                    and len(word) > 3 and word not in ['na', 'la']]
                                
            all_countries['lexrich'].append(round(len(list(set(filtered_words_f))) / len(files), 0))


    df_all_countries = pd.DataFrame.from_dict(all_countries)                    
    df_all_countries.to_csv('all_countries.csv')
    return all_countries

def graphs(rootdir:str):
    '''This function produces two bar charts based on the lyrics manipulated in
        create_country_files'''

    df = pd.read_csv(rootdir + '/all_countries.csv')

    print(df.head(5))

    if df.shape[0] == 0:
        list_lyrics = create_country_files(rootdir=rootdir)
        df = pd.DataFrame(list_lyrics)

    else:
        pass

    df = df.sort_values('lexrich', ascending=True)

    # fig = go.Figure(go.Bar(
    #             x=df['country'],
    #             y=df['wordcount']))
    # fig.show()

    fig = go.Figure(go.Bar(
        x=df['year'],
        y=df['lexrich']))
    fig.show()

def word_cloud(rootdir: str):
    '''This function creates a wordcloud based on the countries lyrics'''
    
    list_lyrics = create_country_files(rootdir=rootdir)

    df = pd.DataFrame(list_lyrics)

    for index, row in df.iterrows():
        #all_words = ' '.join(row['lyrics'])

        lyrics = row['lyrics'].replace('-', ' ')

        filtered_words = [word for word in lyrics.split(' ') if word not in stopwords.words('english')
                                    and len(word) > 3 and word not in ['na', 'la']]

        wordcloud = WordCloud().generate(' '.join(filtered_words))
        plt.figure()
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.show()


def sentiment_analyzer(rootdir: str):
    
    df = pd.DataFrame(columns=('artist', 'pos', 'neu', 'neg'))
    sid = SentimentIntensityAnalyzer()
    
    list_lyrics = create_country_files(rootdir=rootdir)

    df = pd.DataFrame(list_lyrics)

    results = pd.DataFrame(columns=('year', 'superpos', 'pos',  'neu', 'neg', 'superneg'))
    for index, row in df.iterrows():
        #all_words = ' '.join(row['lyrics'])

        lyrics = row['lyrics'].replace('-', ' ')

        filtered_words = [word for word in lyrics.split(' ') if word not in stopwords.words('english')
                                    and len(word) > 3 and word not in ['na', 'la']]
            

        superpos = 0
        pos = 0
        neu = 0
        neg = 0
        superneg = 0
        for item in filtered_words:
            comp = sid.polarity_scores(item)
            comp = comp['compound']

            if comp >= 0.5:
                superpos += 1
            elif comp >= 0.15 and comp < 0.5:
                pos += 1
            elif comp >= -0.15 and comp < 0.15:
                neu += 1
            elif comp >= -0.5 and comp < -0.15:
                neg += 1
            else:
                superneg += 1

            print(item)
            print(comp)

        num_total = neu + pos + neg + superneg + superpos
        perc_neu = (neu/num_total)*100
        perc_pos = (pos/num_total)*100
        perc_neg = (neg/num_total)*100
        perc_superpos = (superpos/num_total)*100
        perc_superneg = (superneg/num_total)*100

        results.loc[len(results)] = [row['year'], perc_neu, perc_pos, perc_neg,
                                                     perc_superpos, perc_superneg]  
    
    results.plot.bar(x='country', stacked=True)
    plt.show()   

def sentiment_analyzer_hf(rootdir):
    from transformers import pipeline
    sentiment_pipeline = pipeline(model='sentiment-analysis')

    data = ['jhi', 'my', 'name', 'is', 'dimarco']

#create_country_files(rootdir=rootdir)
graphs(rootdir=rootdir)
#word_cloud(rootdir=rootdir)
#sentiment_analyzer_hf(rootdir=rootdir)
