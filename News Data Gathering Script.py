from newsapi import NewsApiClient 
import json
import re
import pandas as pd
news = NewsApiClient(api_key='XXXXXXXXX')
stocks=["Apple","Amazon","Google","Facebook","Microsoft"]
date=['YYYY-MM-DD','YYYY-MM-DD'] 
source=['reuters', 'cnbc','business-insider', 'fortune', 'the-new-york-times','the-wall-street-journal']
domain=['reuters.com', 'cnbc.com','businessinsider.com', 'fortune.com', 'nytimes.com','wsj.com']
for k in range(len(source)):
    try:
        for i in range(len(stocks)):
            newsdata=pd.DataFrame(columns=['Stock', 'Date', 'Source','Description','Url','Content'])
            pattern=re.compile(r"\d{4}-\d{2}-\d{2}")
            all_articles = news.get_everything(q=stocks[i],
                                                      sources=source[k],
                                                      domains=domain[k],
                                                      from_param=date[0],
                                                      to=date[1],
                                                      language='en',
                                                      sort_by='relevancy')
            for j in all_articles['articles']:
                        newsdata=newsdata.append({'Stock':stocks[i],'Date':pattern.findall(j['publishedAt'])[0],
                                             'Source':j['source']['name'],'Description':j['description'],
                                             'Url':j['url'],'Content':j['content']},ignore_index=True)
                        filename=source[k]+"_"+stocks[i]+"_"+date[0]+"_"+date[1]+'.csv'
                        file="New folder/"+ filename
                        newsdata.to_csv(file, encoding='utf-8', index=False)
    except:
        pass
