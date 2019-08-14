import pandas as pd
import re
news=pd.read_csv("News.csv",
                        header=0,encoding = 'unicode_escape')
for index, row in news.iterrows():
    text=row[3]
    
    """ Removes url address with "url" """
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','',text)
    
    news.xs(index)['Description']=text

#Splitting the Data into 70-30

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(news['Description'],news['Sentiment'], test_size=0.3, random_state=1)

#Machine Learning Approach

from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
from sklearn.naive_bayes import BernoulliNB
from sklearn import metrics
def ml_approach(X_train, X_test, y_train, y_test):
    token = RegexpTokenizer(r'[a-zA-Z0-9]+')
    cv = CountVectorizer(lowercase=True,stop_words='english',ngram_range = (1,1),tokenizer = token.tokenize)
    text_counts_training= cv.fit_transform(X_train)
    clf = BernoulliNB().fit(text_counts_training, y_train)
    text_counts_testing= cv.transform(X_test)
    predicted= clf.predict(text_counts_testing)
    print("BernoulliNB Accuracy:",metrics.accuracy_score(y_test, predicted))

#VADER Approach

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
news_vader=pd.DataFrame(columns=['News', 'Sentiment'])
news_vader['News']=X_test
def vader_approach(news_vader):
    analyser = SentimentIntensityAnalyzer()
    for index, row in news_vader.iterrows():
        text=row[0]
        score=analyser.polarity_scores(text)
        if score['compound']>=0.05:
            news_vader.xs(index)['Sentiment']='Positive'
        elif score['compound']<=-0.05:
            news_vader.xs(index)['Sentiment']='Negative'
        else:
            news_vader.xs(index)['Sentiment']='Neutral'
    print("Vader Accuracy:",metrics.accuracy_score(y_test, news_vader['Sentiment']))

#SenitWord Approach

news_sentiword=pd.DataFrame(columns=['News', 'Sentiment'])
news_sentiword['News']=X_test
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
from nltk import sent_tokenize, word_tokenize, pos_tag
from nltk.corpus import stopwords
def penn_to_wn(tag):
    """
    Convert between the PennTreebank tags to simple Wordnet tags
    """
    if tag.startswith('J'):
        return wn.ADJ
    elif tag.startswith('N'):
        return wn.NOUN
    elif tag.startswith('R'):
        return wn.ADV
    elif tag.startswith('V'):
        return wn.VERB
    return None

lemmatizer = WordNetLemmatizer()
def swn_polarity(text):
   
    sentiment = 0.0
    
    stop_words=set(stopwords.words("english"))
    raw_sentences = sent_tokenize(text)
    for raw_sentence in raw_sentences:
        tagged_sentence = pos_tag(word_tokenize(raw_sentence))
 
        for word, tag in tagged_sentence:
    
            if word not in stop_words:
        
                wn_tag = penn_to_wn(tag)
                if wn_tag not in (wn.NOUN, wn.ADJ, wn.ADV,wn.VERB):
                    continue

                lemma = lemmatizer.lemmatize(word, pos=wn_tag)
                if not lemma:
                    continue

                synsets = wn.synsets(lemma, pos=wn_tag)
                if not synsets:
                    continue

                # Take the first sense, the most common
                synset = synsets[0]
                swn_synset = swn.senti_synset(synset.name())
                sentiment += swn_synset.pos_score() - swn_synset.neg_score()
 
    return sentiment
def sentiword_approach(news_sentiword):
    for index, row in news_sentiword.iterrows():
        text=row[0]
        senti=swn_polarity(text)
        if senti>=0.20:
            news_sentiword.xs(index)['Sentiment']='Positive'
        elif senti<=-0.20:
            news_sentiword.xs(index)['Sentiment']='Negative'
        else:
            news_sentiword.xs(index)['Sentiment']='Neutral'
    print("SentiWordNet Accuracy:",metrics.accuracy_score(y_test, news_sentiword['Sentiment']))
