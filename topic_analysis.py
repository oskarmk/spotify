## Import modules
import csv
import gensim
import nltk
import os
import pickle
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis

from gensim.corpora import Dictionary, MmCorpus
from gensim.models import LdaModel, CoherenceModel

from langdetect import detect_langs
from langdetect.lang_detect_exception import LangDetectException

from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer # groups together inflected group of words
##-------------------
## Tokenize lyrics and create a list of all the tokenized lyrics of the decade

lyric_corpus_tokenized = []

# decade = '1950'
# path = f'/home/oscar/newsletter/spotify/data2ndpost100/{decade}/lyrics'

path = '/home/oscar/newsletter/spotify/data2ndpost100/Conglomerate'

for filename in os.listdir(path):
    
    with open(os.path.join(path, filename), 'r') as f:
        lyric = f.read()
        # tokenize lyrics
        tokenizer = RegexpTokenizer(r'\w+')
        tokenized_lyric = tokenizer.tokenize(lyric.lower())

        lyric_corpus_tokenized.append(tokenized_lyric) # list of list [[]]
##-------------------
## Remove numeric tokens or tokens with less than 3 chars

for s, song in enumerate(lyric_corpus_tokenized):
    filtered_song = []
    for token in song:
        if len(token) > 2 and not token.isnumeric():
            filtered_song.append(token)
    
    lyric_corpus_tokenized[s] = filtered_song
##-------------------
## Token Lemmatization

lemmatizer = WordNetLemmatizer()

for s, song in enumerate(lyric_corpus_tokenized):
    lemmatized_tokens = []
    for token in song:
        lemmatized_tokens.append(lemmatizer.lemmatize(token))
    
    lyric_corpus_tokenized[s] = lemmatized_tokens
##-------------------
## Remove stopwords

stopwords = stopwords.words('english')
new_stop_words = ['chorus','embed','verse','yeah','hey','whoa','woah', 'ohh', 'was', 'mmm', 'oooh','yah','yeh','mmm', 'hmm','deh','doh','jah','wa']
stopwords.extend(new_stop_words)

for s,song in enumerate(lyric_corpus_tokenized):
    filtered_text = []    
    for token in song:
        if token not in stopwords:
            filtered_text.append(token)
    lyric_corpus_tokenized[s] = filtered_text

##-------------------
# Dictionary Creating and occurence-based Filtering. Gensim requires a dict representation of the docs

dictionary = Dictionary(lyric_corpus_tokenized)
# dictionary.filter_extremes(no_below = 100, no_above = 0.8) # Filter out occurences
##-------------------
## Bag-of-words and Index to Dictionary Conversion

gensim_corpus = [dictionary.doc2bow(song) for song in lyric_corpus_tokenized]
## doc2bow -> Convert document (a list of words) into the bag-of-words
## format = list of (token_id, token_count) 2-tuples.

temp = dictionary[0]
id2word = dictionary.id2token
##-------------------
## Setting model params & executing model training

chunksize = 20
passes = 20
iterations = 40
num_topics = 3

lda_model = LdaModel(
    corpus = gensim_corpus,
    id2word=id2word,
    chunksize=chunksize,
    alpha='auto',
    iterations=iterations,
    num_topics=num_topics,
    passes=passes
)

vis_data = gensimvis.prepare(lda_model, gensim_corpus, dictionary)

# pyLDAvis.display(vis_data)
# pyLDAvis.show(vis_data)
pyLDAvis.save_html(vis_data, '.1950_LDA_k_' + str(num_topics) + '.html')
##############################################################################

