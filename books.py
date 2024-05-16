import csv
import pandas as pd 
import os
import nltk

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


corpus = 'project-dialogism-novel-corpus/'

books = corpus + 'data/'

novel_metadata = pd.read_csv(corpus + 'PDNC-Novel-Index.csv')
author_metadata = pd.read_csv(corpus + 'PDNC-Author-Index.csv')

## seperate the novels into train and test sets into an 80/20 split
train_novels = novel_metadata.sample(frac=0.8)
test_novels = novel_metadata.drop(train_novels.index)

'''
some hueristics for idenfiying quotes in the text
- quotes are usually enclosed in double quotes
- quotes are usually followed by a comma
- quotes are usually followed by a period
- quotes are usually followed by a question mark
- quotes are usually followed by an exclamation mark
- quotes usally have a speaker words such as said or asked or proper nouns
'''

## use niave bayes to classify the quotes
## use the following labels
## - quote
## - not quote

def preprocess_vector(text):
    ## create list of words to ignore such as CHAPTER
    ignore = ['CHAPTER', 'Chapter', 'chapter']
    ## returns a vector if it has punctuation or a proper noun
    has_punctuation = 0
    has_proper_noun = 0
    text = nltk.word_tokenize(text)
    for word in text:
        if word in ignore:
            has_proper_noun = 0
            has_punctuation = 0
            break 
        if word in ['.', ',', '?', '!', "'", '"']:
            has_punctuation = 1
        if word.istitle():
            has_proper_noun = 1
    return [has_punctuation, has_proper_noun]

book_files = os.listdir(books)
first_book = book_files[0]


## vector for sentence classification
x = []
with open(books + first_book + '/novel_text.txt', 'r') as file:
    text = file.read()
    ## split the text into sentences
    sentences = nltk.sent_tokenize(text)
    for sent in sentences:
        vect = preprocess_vector(sent)
        x.append([vect, sent])


## read first book csv
first_quote_info = pd.read_csv(books + first_book + '/quotation_info.csv')

# extract array from the dataframe under header quoteText
quotes = first_quote_info['quoteText'].values




        

    