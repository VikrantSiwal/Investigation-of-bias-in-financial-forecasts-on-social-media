from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
from nltk import sent_tokenize, word_tokenize, pos_tag
from nltk.corpus import stopwords
import pandas as pd
import re
tweets=pd.read_csv("E:/UCC Lecture Notes and e-books/Term 3/Project-Stock Market/Final Data/Twitter/Combined/All Tweets.csv",
                         header=0,encoding = 'unicode_escape')
tweets=tweets[['text','created_at']]
for index, row in tweets.iterrows():
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
    
    """ Removes hastags"""
    text = re.sub(r'#', '', text)
    
    """ Removes $"""
    text = re.sub(r'\$[A-Za-z]{0,}', '', text)
    
    tweets.xs(index)['text']=text
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
tweets["Sentiment_score"] = ''
tweets['Positive']=''
tweets['Negative']=''
tweets['Neutral']=''
for index, row in tweets.iterrows():
    text=row[0]
    senti=swn_polarity(text)
    tweets.xs(index)['Sentiment_score']=senti
    if senti>=0.2:
        tweets.xs(index)['Positive']=1
        tweets.xs(index)['Negative']=0
        tweets.xs(index)['Neutral']=0
    elif senti<=-0.2:
        tweets.xs(index)['Positive']=0
        tweets.xs(index)['Negative']=1
        tweets.xs(index)['Neutral']=0
    else:
        tweets.xs(index)['Positive']=0
        tweets.xs(index)['Negative']=0
        tweets.xs(index)['Neutral']=1
tweets_sentiment=tweets[['Stock','created_at','Positive','Negative','Neutral']].groupby(['Stock','created_at']).sum()
tweets_sentiment['Overall']=tweets_sentiment.sum(axis=1)
tweets_sentiment['Pos']=tweets_sentiment['Positive']/tweets_sentiment['Overall']
tweets_sentiment['Neu']=tweets_sentiment['Neutral']/tweets_sentiment['Overall']
tweets_sentiment['Neg']=tweets_sentiment['Negative']/tweets_sentiment['Overall']
tweets_sentiment[['Pos','Neu','Neg']].to_csv('E:/UCC Lecture Notes and e-books/Term 3/Project-Stock Market/Final Data/Twitter/Combined/processed_tweets.csv')