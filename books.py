import csv
import pandas as pd 
import os
import nltk
import re

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


book_files = os.listdir(books)
first_book = book_files[0]


## read first book csv quotes column as a list
first_quote_info = pd.read_csv(books + first_book + '/quotation_info.csv')

# extract array from the dataframe under header subQuotationList and convert it to a list of strings
quotes = first_quote_info['subQuotationList'].apply(lambda x: eval(x) if isinstance(x, str) else x).tolist()

## flatten the list of quotes
quotes = [item for sublist in quotes for item in sublist]

for x in range(len(quotes)):
    ## tokenize the quote
    quotes[x] = nltk.sent_tokenize(quotes[x])

quotes = [item for sublist in quotes for item in sublist]


        

## vector for sentence classification
x = []
with open(books + first_book + '/novel_text.txt', 'r') as file:
    text = file.read()
    sentence = nltk.sent_tokenize(text)

    ## mark if the sentence contains a quote
    for s in sentence:
        if any(q in s for q in quotes):
            x.append((s, 'quote'))
        else:
            x.append((s, 'not quote'))


## create a dataframe from the list of tuples
df = pd.DataFrame(x, columns=['text', 'label'])

##split the data into training and testing sets
train = df.sample(frac=0.8)
test = df.drop(train.index)

## use count vectorizer to convert the text into a matrix of token counts

vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(train['text'])

## use maximum entropy to classify the quotes
clf = MultinomialNB()
clf.fit(X_train, train['label'])

## test the classifier
X_test = vectorizer.transform(test['text'])
print(clf.score(X_test, test['label']))

## test the classifier on the first book
X_first = vectorizer.transform(sentence)
print(clf.predict(X_first))
## print the quotes




        

    