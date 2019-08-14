import pandas as pd
import re
twitter=pd.read_csv("E:/UCC Lecture Notes and e-books/Term 3/Project-Stock Market/Final Data/Work/Machine Learing/Tweets.csv",
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

#Split 70:30

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(twitter['text'],twitter['Sentiment'], test_size=0.3, random_state=1)

#Machine Leanring Approach

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

# VADER Approach

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
twitter_vader=pd.DataFrame(columns=['Text', 'Sentiment'])
twitter_vader['Text']=X_test
def vader_approach(twitter_vader):
    analyser = SentimentIntensityAnalyzer()
    for index, row in twitter_vader.iterrows():
        text=row[0]
        score=analyser.polarity_scores(text)
        if score['compound']>=0.05:
            twitter_vader.xs(index)['Sentiment']='Positive'
        elif score['compound']<=-0.05:
            twitter_vader.xs(index)['Sentiment']='Negative'
        else:
            twitter_vader.xs(index)['Sentiment']='Neutral'
    print("Vader Accuracy:",metrics.accuracy_score(y_test, twitter_vader['Sentiment']))
    
# SentiWord Approach

twitter_sentiword=pd.DataFrame(columns=['Text', 'Sentiment'])
twitter_sentiword['Text']=X_test
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
from nltk import sent_tokenize, word_tokenize, pos_tag
from nltk.corpus import stopwords
lemmatizer = WordNetLemmatizer()
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
def sentiword_approach(twitter_sentiword):
    for index, row in twitter_sentiword.iterrows():
        text=row[0]
        senti=swn_polarity(text)
        if senti>=0.20:
            twitter_sentiword.xs(index)['Sentiment']='Positive'
        elif senti<=-0.20:
            twitter_sentiword.xs(index)['Sentiment']='Negative'
        else:
            twitter_sentiword.xs(index)['Sentiment']='Neutral'
    print("SentiWordNet Accuracy:",metrics.accuracy_score(y_test, twitter_sentiword['Sentiment']))
