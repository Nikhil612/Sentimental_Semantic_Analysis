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

<img src = "https://user-images.githubusercontent.com/34812655/115943738-a5c98900-a466-11eb-9fc8-719a7339cd65.png" width="800" height="400">


CODE TO DO SO:
```
kaggle dataset download -f Musical_instruments_reviews.csv https://www.kaggle.com/eswarchandt/amazon-music-reviews?select=Musical_instruments_reviews.csv to C:\Users\nj061\OneDrive\Documents\Sentimental Analysis_wordCloud
```
```
wget -x -c --load-cookies cookies.txt https://www.kaggle.com/eswarchandt/amazon-music-reviews?select=Musical_instruments_reviews.csv
```


<img src = "https://user-images.githubusercontent.com/34812655/115943592-af062600-a465-11eb-8553-d554f631bea7.png" width="800" height="400">


## DATA PREPROCESSING
```
# replacing numerical values with categorical values to reduce the classes to sentiments

data['sentiment'] = data.overall.replace({
    1:'negative',
    2:'negative',
    3:'neutral',
    4:'positive',
    5:'positive'
})
```
```
X_data = data['reviewtext'] + ' ' + data['summary']
y_data = data['sentiment']

# creating new dataframe

X_data_df = pd.DataFrame(data=X_data)
X_data_df.columns = ['review']
X_data_df.head()
```

<img src ="https://user-images.githubusercontent.com/34812655/115944006-318fe500-a468-11eb-9db1-fbe16410d7b4.png" width="800" height="400">



## RESULTS

THE BELOW SHOWS OUTCOME FOR MUSICAL_INSTRUMENT DATASET


<img src = "https://user-images.githubusercontent.com/34812655/115943809-e45f4380-a466-11eb-9675-d0a836d127a3.png" width="800" height="400">


<img src = "https://user-images.githubusercontent.com/34812655/115943815-f5a85000-a466-11eb-8ab0-b42c455041e5.png" width="800" height="400">


<img src = "https://user-images.githubusercontent.com/34812655/115943827-0658c600-a467-11eb-960d-73ca580c1a7c.png" width="800" height="400">



AACURACY COMPARISON
```

model = ['MNB', 'Random Forest',  'SVM']
acc = [MNB_accuracy, rfc_accuracy, svc_accuracy]

sns.set_style("whitegrid")
plt.figure(figsize=(5,6))
plt.yticks(np.arange(0,100,10))
plt.ylabel("Test Accuracy %")
plt.xlabel("Machine Learning Model")
sns.barplot(x= model, y= acc)
plt.show()
```

<img src = "https://user-images.githubusercontent.com/34812655/115943866-3b651880-a467-11eb-98f3-263baefbece6.png" width="800" height="600">

F-1 SCORE COMPARISON
```
model = ['MNB', 'Random Forest',  'SVM']
f1_score = [MNB_f1, rfc_f1, svc_f1]

sns.set_style("whitegrid")
plt.figure(figsize=(5,6))
plt.yticks(np.linspace(0,1,25))
plt.ylabel("f1-score")
plt.xlabel("Machine Learning Model")
sns.barplot(x= model,  y= f1_score)
plt.show()
```
<img src = "https://user-images.githubusercontent.com/34812655/115943901-76ffe280-a467-11eb-86c4-60dba601d0f7.png" width="800" height="600">

