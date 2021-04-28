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

![image](https://user-images.githubusercontent.com/34812655/115944142-d6aabd80-a468-11eb-9705-e0269d66ec4a.png)


## CREATING TEXT PROCESSING FUNCTIONS

```
string.punctuation
def final(X_data_full):
    
    # function for removing punctuations
    def remove_punct(X_data_func):
        string1 = X_data_func.lower()
        translation_table = dict.fromkeys(map(ord, string.punctuation),' ')
        string2 = string1.translate(translation_table)
        return string2
    
    X_data_full_clear_punct = []
    for i in range(len(X_data_full)):
        test_data = remove_punct(X_data_full[i])
        X_data_full_clear_punct.append(test_data)
        
    # function to remove stopwords
    def remove_stopwords(X_data_func):
        pattern = re.compile(r'\b(' + r'|'.join(stopwords.words('english')) + r')\b\s*')
        string2 = pattern.sub(' ', X_data_func)
        return string2
    
    X_data_full_clear_stopwords = []
    for i in range(len(X_data_full)):
        test_data = remove_stopwords(X_data_full[i])
        X_data_full_clear_stopwords.append(test_data)
        
    # function for tokenizing
    def tokenize_words(X_data_func):
        words = nltk.word_tokenize(X_data_func)
        return words
    
    X_data_full_tokenized_words = []
    for i in range(len(X_data_full)):
        test_data = tokenize_words(X_data_full[i])
        X_data_full_tokenized_words.append(test_data)
        
    # function for lemmatizing
    lemmatizer = WordNetLemmatizer()
    def lemmatize_words(X_data_func):
        words = lemmatizer.lemmatize(X_data_func)
        return words
    
    X_data_full_lemmatized_words = []
    for i in range(len(X_data_full)):
        test_data = lemmatize_words(X_data_full[i])
        X_data_full_lemmatized_words.append(test_data)
        
    # creating the bag of words model
    cv = CountVectorizer(max_features=1000)
    X_data_full_vector = cv.fit_transform(X_data_full_lemmatized_words).toarray()
    
    
    tfidf = TfidfTransformer()
    X_data_full_tfidf = tfidf.fit_transform(X_data_full_vector).toarray()
    
    return X_data_full_tfidf
 ```


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

# WORD CLOUD CREATION

## DATA BEING PROCESSED BEFORE WORD CLOUD CAN BE CREATED

```
def sentiment_rating(rating):
    # Replacing ratings of 1,2,3 with 0 (not good) and 4,5 with 1 (good)
    if(int(rating) == 1 or int(rating) == 2 or int(rating) == 3):
        return 0
    else: 
        return 1
df.overall = df.overall.apply(sentiment_rating) 

stop = set(stopwords.words('english'))
punctuation = list(string.punctuation)
stop.update(punctuation)

def get_simple_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    else:
        return wordnet.NOUN
        
 lemmatizer = WordNetLemmatizer()
def lemmatize_words(text):
    final_text = []
    for i in text.split():
        if i.strip().lower() not in stop:
            pos = pos_tag([i.strip()])
            word = lemmatizer.lemmatize(i.strip(),get_simple_pos(pos[0][1]))
            final_text.append(word.lower())
    return " ".join(final_text)
```

## PLOTTED WORD CLOUD

```
WordCloud of Product with Good Ratings

plt.figure(figsize = (20,20)) # Text Reviews with Poor Ratings
wc = WordCloud(min_font_size = 3,  max_words = 3000 , width = 1600 , height = 800).generate(" ".join(bad))
plt.imshow(wc,interpolation = 'bilinear')
```

![image](https://user-images.githubusercontent.com/34812655/115944250-7d8f5980-a469-11eb-99e2-e2b341416da7.png)

```
WordCloud for Product with Bad Ratings

plt.figure(figsize = (20,20)) # # Text Reviews with Good Ratings
wc = WordCloud(min_font_size = 3,  max_words = 3000 , width = 1600 , height = 800).generate(" ".join(good))
plt.imshow(wc,interpolation = 'bilinear')
```

![image](https://user-images.githubusercontent.com/34812655/115944283-b4fe0600-a469-11eb-86f4-44f1539fdba4.png)
