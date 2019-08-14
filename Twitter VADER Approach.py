import pandas as pd
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
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
analyser = SentimentIntensityAnalyzer()
tweets['Positive']=''
tweets['Negative']=''
tweets['Neutral']=''
for index, row in tweets.iterrows():
    text=row[0]
    score=analyser.polarity_scores(text)
    if score['compound']>=0.05:
        tweets.xs(index)['Positive']=1
        tweets.xs(index)['Negative']=0
        tweets.xs(index)['Neutral']=0
    elif score['compound']<=-0.05:
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
