import pandas as pd
import re
news=pd.read_csv("E:/UCC Lecture Notes and e-books/Term 3/Project-Stock Market/Final Data/Work/Machine Learing/News.csv",
                         header=0,encoding = 'unicode_escape')
for index, row in news.iterrows():
    text=row[3]
    
    """ Removes url address with "url" """
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','',text)
    
    news.xs(index)['Description']=text
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(news['Description'],news['Sentiment'], test_size=0.3, random_state=1)
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
