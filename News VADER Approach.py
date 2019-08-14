import pandas as pd
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyser = SentimentIntensityAnalyzer()
combined_news_data=pd.read_csv("E:/UCC Lecture Notes and e-books/Term 3/Project-Stock Market/Final Data/News/Combined file/All News.csv")
for index, row in combined_news_data.iterrows():
    text=row[3]
    
    """ Removes url address with "url" """
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','',text)
    
    combined_news_data.xs(index)['Description']=text
combined_news_data['Compound_score']=''
combined_news_data['Sentiment']=''
for index, row in combined_news_data.iterrows():
    text=row[3]
    score=analyser.polarity_scores(text)
    combined_news_data.xs(index)['Compound_score']=score['compound']
    if score['compound']>=0.05:
        combined_news_data.xs(index)['Sentiment']='Positive'
    elif score['compound']<=-0.05:
        combined_news_data.xs(index)['Sentiment']='Negative'
    else:
        combined_news_data.xs(index)['Sentiment']='Neutral'
news_data=pd.DataFrame(columns=['Stock', 'Date', 'Positive','Negative','Neutral'])
news_data['Stock']=combined_news_data['Stock']
news_data['Date']=combined_news_data['Date']
for index, row in combined_news_data.iterrows():
    text=row[7]
    if row[7]=='Positive':
        news_data.xs(index)['Positive']=1
        news_data.xs(index)['Negative']=0
        news_data.xs(index)['Neutral']=0
    elif row[7]=='Negative':
        news_data.xs(index)['Negative']=1 
        news_data.xs(index)['Positive']=0
        news_data.xs(index)['Neutral']=0
    else:
        news_data.xs(index)['Neutral']=1
        news_data.xs(index)['Positive']=0
        news_data.xs(index)['Negative']=0
news=news_data.groupby(['Stock','Date']).sum()
news['Overall']=news.sum(axis=1)
news['Pos']=news['Positive']/news['Overall']
news['Neu']=news['Neutral']/news['Overall']
news['Neg']=news['Negative']/news['Overall']
news[['Pos','Neu','Neg']].to_csv('E:/UCC Lecture Notes and e-books/Term 3/Project-Stock Market/Final Data/News/Combined file/processed_file.csv')