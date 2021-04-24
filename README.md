# Sentimental_Semantic_Analysis

## The CRUX

The best way to know how customer evaluates a service is to know what he/she thinks. So in order to deal with customer sentiments about a particular product/service NLP tools cpme quite handy. Similarly this projects also implements NLP power via its effficient libraries. To give a brief, we have used some of most powerful libraries and some complex algorithms to invoke the strength that customer's words carry. 

NLP Libraries:
```
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
import string
```

Complex Algorithms:
```
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
```

## Datasets That helped in Process

<img src = "https://user-images.githubusercontent.com/34812655/115942907-9431b280-a461-11eb-81ff-528b7a128e91.png" width="800" height="400">

CODE TO DO SO

<img src = "https://user-images.githubusercontent.com/34812655/115943592-af062600-a465-11eb-8553-d554f631bea7.png" width="800" height="400">


