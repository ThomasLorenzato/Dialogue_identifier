import nltk
import re
## import a book corpus from nltk
nltk.download('gutenberg')
nltk.download('punkt')
from nltk.corpus import gutenberg

emma = gutenberg.raw('austen-emma.txt')

## find dialogues in the book
dialogues = re.findall(r'\"(.+?)\"', emma)

print(dialogues[:10])