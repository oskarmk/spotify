# standard imports
import os
import re
import tqdm
from typing import List

#third-party import
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
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

    df = pd.read_csv(rootdir + '/all_countries.csv')

    if df.shape[0] == 0:
        list_lyrics = create_country_files(rootdir=rootdir)

        df = pd.DataFrame(list_lyrics)

    else:
        pass

    for index, row in df.iterrows():
        #all_words = ' '.join(row['lyrics'])

        lyrics = row['lyrics'].replace('-', ' ')

        filtered_words = [word for word in lyrics.split(' ') if word not in stopwords.words('english')
                                    and len(word) > 3 and word not in ['na', 'la']]

        wordcloud = WordCloud().generate(' '.join(filtered_words))
        plt.figure()
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(row['year'])
        plt.show()


def sentiment_analyzer(rootdir: str):
    '''nltk Sentiment analyzer for the lyrics in the different decades'''

    df = pd.DataFrame(columns=('artist', 'pos', 'neu', 'neg'))
    sid = SentimentIntensityAnalyzer()

    df = pd.read_csv(rootdir + '/all_countries.csv')

    if df.shape[0] == 0:
        list_lyrics = create_country_files(rootdir=rootdir)
        df = pd.DataFrame(list_lyrics)

    else:
        pass

    results = pd.DataFrame(columns=('decade', 'superpos', 'pos',  'neu', 'neg', 'superneg'))

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

        num_total = neu + pos + neg + superneg + superpos
        perc_neu = (neu/num_total)*100
        perc_pos = (pos/num_total)*100
        perc_neg = (neg/num_total)*100
        perc_superpos = (superpos/num_total)*100
        perc_superneg = (superneg/num_total)*100

        results.loc[len(results)] = [row['year'], perc_neu, perc_pos, perc_neg,
                                                     perc_superpos, perc_superneg]

    results.plot.bar(x='decade', stacked=True)
    plt.show()

def sentiment_analyzer_hf(rootdir):
    from transformers import pipeline
    sentiment_pipeline = pipeline(model='sentiment-analysis')

    data = ['jhi', 'my', 'name', 'is', 'dimarco']

def line_charts(rootdir):

    words_to_analyze = ['love', 'hate', 'man', 'woman', 'heart', 'time', 'day', 'night']

    df = pd.DataFrame({words_to_analyze[0]: pd.Series(dtype='int'),
                      words_to_analyze[1]: pd.Series(dtype='int'),
                      words_to_analyze[2]: pd.Series(dtype='int'),
                      words_to_analyze[3]: pd.Series(dtype='int'),
                      words_to_analyze[4]: pd.Series(dtype='int'),
                      words_to_analyze[5]: pd.Series(dtype='int'),
                      words_to_analyze[6]: pd.Series(dtype='int'),
                      words_to_analyze[7]: pd.Series(dtype='int')})

    #df = pd.DataFrame(columns=words_to_analyze, dtype=int)

    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            if file == 'full_lyrics_clean.txt':
                words_count = [0] * len(words_to_analyze)
                with open(os.path.join(subdir, file)) as f:
                    content = f.read()

                    ####
                    words = nltk.word_tokenize(content) # create list of words (tokens)

                    tagged_words = nltk.pos_tag(words) # create list of tagged words (vern, noun, ...)

                    verbs = []
                    verbs_tag = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
                    for i in range(len(tagged_words)): # filter out verbs
                        if tagged_words[i][1] in verbs_tag:
                            verbs.append(tagged_words[i][0])
                        else:
                            pass

                    # remove stopwords & verbs
                    #stopwords_nltk = stopwords.words('english') + verbs # wothout verbs
                    stopwords_nltk = stopwords.words('english') # include verbs

                    list_of_words = [w for w in words if w.lower() not in stopwords_nltk and w.isalpha()
                             and len(w)>2] #change the length to check for words with different lengths
                    ####
                    #list_of_words = content.split(' ')
                    #print(len(list_of_words))
                    #match_count = 0
                    for word in list_of_words:
                        for i, w in enumerate(words_to_analyze):
                            if word.lower() == w:
                                words_count[i] += 1
                            else:
                                continue


                        # if word.lower() == words_to_analyze[0]:
                        #     match_count += 1
                        # else:
                        #     continue

                    df.loc[len(df)] = [(num / len(list_of_words))*100 for num in words_count]

                    # df.loc[len(df)] = [subdir[-4:], round((match_count/len(list_of_words)*100), 5)]

    #plot with plotly

    fig = px.line(df)
    fig.show()

    #df = df.sort_values('decade', ascending=True)
    # print(df)
    # df.plot.line(stacked=True)
    # plt.show()


def nltk_stuff(rootdir):
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            if file == 'full_lyrics_clean.txt':
                print(subdir[-4:])
                with open(os.path.join(subdir, file)) as f:
                    content = f.read()

                    words = nltk.word_tokenize(content) # create list of words (tokens)

                    tagged_words = nltk.pos_tag(words) # create list of tagged words (vern, noun, ...)

                    verbs = []
                    verbs_tag = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
                    for i in range(len(tagged_words)): # filter out verbs
                        if tagged_words[i][1] in verbs_tag:
                            verbs.append(tagged_words[i][0])
                        else:
                            pass

                    # remove stopwords & verbs
                    stopwords_nltk = stopwords.words('english') + verbs # wothout verbs
                    #stopwords_nltk = stopwords.words('english') # include verbs

                    words = [w for w in words if w.lower() not in stopwords_nltk and w.isalpha()
                             and len(w)==5] #change the length to check for words with different lengths

                    # frequency distribution
                    freqd = nltk.FreqDist(words)
                    # print(freqd.most_common(1))
                    # print(freqd.most_common(10)) # print list of 10 most common words
                    print(freqd.tabulate(10)) # print tabulated version of 10 most common words
                    # print(freqd['baby']) # print amount of appearances of the desired word
                    #----------------------

                    # Concordance --> get context of when a word appears. Sorted in order of app.
                    text = nltk.Text(words)
                    concordance_list = text.concordance_list('love', lines=2)
                    # for entry in concordance_list:
                    #     print(entry.line)
                    #----------------------

                    # Collocations --> series of words that usually appear together
                    # finder = nltk.collocations.BigramCollocationFinder.from_words(words)
                    # finder = nltk.collocations.TrigramCollocationFinder.from_words(words)
                    finder = nltk.collocations.QuadgramCollocationFinder.from_words(words)
                    # print(finder.ngram_fd.most_common(5))
                    # print(finder.ngram_fd.tabulate(5))
                    #----------------------

                    # Sentiment Analyizer --> VADER (Valence Aware Dict and sEntiment Reasoner)
                    # SEE FUNCTION ABOVE DEFINED

#### HACER UNA FUNCION QUE LEA LA DF O LA CREEE Y LA LIMPIE y returnee tres dfs: 1 todo, sin stopw,
# sin verbos

def data_creator(rootdir: str):
    '''Function to read csv with the data of all the countries and
        the words filtered in two formats, one without stopwords
        and one without stopwords and verbs'''

    # Read dataframe
    df = pd.read_csv(rootdir + '/all_countries.csv')

    # If Dataframe empty, create it with the create_country_files() function
    if df.shape[0] == 0:
        list_lyrics = create_country_files(rootdir=rootdir)
        df = pd.DataFrame(list_lyrics)
    else:
        pass

    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            if file == 'full_lyrics_clean.txt':
                with open(os.path.join(subdir, file)) as f:
                    content = f.read()

                    words = nltk.word_tokenize(content) # create list of words (tokens)

                    tagged_words = nltk.pos_tag(words) # create list of tagged words (vern, noun, ...)

                    verbs = []
                    verbs_tag = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
                    for i in range(len(tagged_words)): # filter out verbs
                        if tagged_words[i][1] in verbs_tag:
                            verbs.append(tagged_words[i][0])
                        else:
                            pass

                # remove stopwords & verbs
                stopwords_nltk_nv = stopwords.words('english') + verbs # wothout verbs
                stopwords_nltk = stopwords.words('english') # include verbs

                words_nstp = [w for w in words if w.lower() not in stopwords_nltk and w.isalpha()
                            and len(w) > 2] #change the length to check for words with different lengths
                
                words_nstp_nvrbs = [w for w in words if w.lower() not in stopwords_nltk and w.isalpha()
                            and len(w) > 2] #change the length to check for words with different lengths

    return df, words_nstp, words_nstp_nvrbs





#create_country_files(rootdir=rootdir)
#graphs(rootdir=rootdir)
#word_cloud(rootdir=rootdir)
#sentiment_analyzer(rootdir=rootdir)
#line_charts(rootdir=rootdir)
#nltk_stuff(rootdir=rootdir)
data_creator(rootdir=rootdir)