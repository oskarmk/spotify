from collections import Counter
import re

# import nltk
# nltk.download('punkt')

with open('/home/oscar/newsletter/spotify/data/FI/full_lyrics.txt') as f:
    input_string = f.read()

    regex = re.compile('[^a-zA-Z]')
    input_string = regex.sub(' ', input_string)

    # tokens = nltk.word_tokenize(input_string)

    # tagged = nltk.pos_tag(tokens)

    # print([s for s in tagged if [s1] != 'IN'])

    words = input_string.lower().split()
    words = [word for word in words if len(word) > 3]
    wordCount = Counter(words)

print(wordCount.most_common(50))