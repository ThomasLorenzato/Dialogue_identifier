import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader

x = []

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
