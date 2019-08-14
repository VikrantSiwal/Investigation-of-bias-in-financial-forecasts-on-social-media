import pandas as pd
import re
twitter=pd.read_csv("Tweets.csv",
                         header=0,encoding = 'unicode_escape')
for index, row in twitter.iterrows():
    text=row[0]
    
    """ Removes unicode strings like "\u002c" and "x96" """
    text = re.sub(r'(\\u[0-9A-Fa-f]+)',r'', text)       
    text = re.sub(r'[^\x00-\x7f]',r'',text)
    text = re.sub(r'<U\+[A-Za-z0-9]{1,}>','', text)
    text = re.sub(r'(\r)|(\n)','', text)
    
    """ Removes repetitions of RT """
    text=re.sub(r'RT ','',text)
    
    """ Removes url address with "url" """
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','',text)
    
    """ Removes "@user" """
    text = re.sub('@[^\s]+','',text)
    
    """ Removes $"""
    text = re.sub(r'\$[A-Za-z]{0,}', '', text)
    
    """ Removes hastags"""
    text = re.sub(r'#', '', text)
    
    twitter.xs(index)['text']=text
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(twitter['text'],twitter['Sentiment'], test_size=0.3, random_state=1)
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
token = RegexpTokenizer(r'[a-zA-Z0-9]+')
cv = CountVectorizer(lowercase=True,stop_words='english',ngram_range = (1,1),tokenizer = token.tokenize)
from sklearn.naive_bayes import BernoulliNB
from sklearn import metrics
text_counts_training= cv.fit_transform(X_train)
clf = BernoulliNB().fit(text_counts_training, y_train)
text_counts_testing= cv.transform(X_test)
predicted= clf.predict(text_counts_testing)
print("BernoulliNB Accuracy:",metrics.accuracy_score(y_test, predicted))
