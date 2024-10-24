import csv
import pandas as pd 
import os
import nltk
import re

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader

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

df = pd.DataFrame(x, columns=['sentence', 'label'])

# Load your dataframe
df = pd.DataFrame(x, columns=['sentence', 'label'])

# Step 1: Preprocess the text data with BERT Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Define a custom dataset class
class QuoteDataset(Dataset):
    def __init__(self, sentences, labels, tokenizer, max_len):
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, item):
        sentence = str(self.sentences[item])
        label = 1 if self.labels[item] == 'quote' else 0  # Convert labels to binary (1 = quote, 0 = not quote)

        encoding = self.tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# Step 2: Split the data into training and testing sets
train_sentences, test_sentences, train_labels, test_labels = train_test_split(df['sentence'], df['label'], test_size=0.2, random_state=42)

# Step 3: Create Dataset objects
train_dataset = QuoteDataset(train_sentences.tolist(), train_labels.tolist(), tokenizer, max_len=128)
test_dataset = QuoteDataset(test_sentences.tolist(), test_labels.tolist(), tokenizer, max_len=128)

# Step 4: Load BERT for Sequence Classification
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Step 5: Define Trainer and TrainingArguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer
)

# Step 6: Train the model
trainer.train()

# Step 7: Evaluate the model
results = trainer.evaluate()
print(results)

# Step 8: Generate classification report
predictions = trainer.predict(test_dataset)
y_preds = torch.argmax(torch.tensor(predictions.predictions), axis=1)
print(classification_report(test_labels, y_preds))

