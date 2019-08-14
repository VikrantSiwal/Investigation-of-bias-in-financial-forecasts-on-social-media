from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
from nltk import sent_tokenize, word_tokenize, pos_tag
from nltk.corpus import stopwords
import pandas as pd
import re
combined_news_data=pd.read_csv("All News.csv")
for index, row in combined_news_data.iterrows():
    text=row[3]
    
    """ Removes url address with "url" """
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','',text)
    
    combined_news_data.xs(index)['Description']=text
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
combined_news_data['Sentiment_Score']=''
combined_news_data['Positive']=''
combined_news_data['Negative']=''
combined_news_data['Neutral']=''
for index, row in combined_news_data.iterrows():
    text=row[3]
    senti=swn_polarity(text)
    combined_news_data.xs(index)['Sentiment_Score']=senti
    if senti>=0.2:
        combined_news_data.xs(index)['Positive']=1
        combined_news_data.xs(index)['Negative']=0
        combined_news_data.xs(index)['Neutral']=0
    elif senti<=-0.2:
        combined_news_data.xs(index)['Positive']=0
        combined_news_data.xs(index)['Negative']=1
        combined_news_data.xs(index)['Neutral']=0
    else:
        combined_news_data.xs(index)['Positive']=0
        combined_news_data.xs(index)['Negative']=0
        combined_news_data.xs(index)['Neutral']=1
combine_news=combined_news_data[['Stock','Date','Positive','Negative','Neutral']].groupby(['Stock','Date']).sum()
combine_news['Overall']=combine_news.sum(axis=1)
combine_news['Pos']=combine_news['Positive']/combine_news['Overall']
combine_news['Neu']=combine_news['Neutral']/combine_news['Overall']
combine_news['Neg']=combine_news['Negative']/combine_news['Overall']
combine_news[['Pos','Neu','Neg']].to_csv('News_SentimentWord.csv')
