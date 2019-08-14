import pandas as pd
df=pd.read_csv("E:/UCC Lecture Notes and e-books/Term 3/Project-Stock Market/Final Data/Compare Files/Compare.csv")
from scipy.stats.stats import pearsonr
def cor(df):
    l=df.Stock.unique()
    df_new=pd.DataFrame(columns=["Stock","Pos","Neg","Neu"])
    df_new['Stock']=list(l)
    pos=[]
    neu=[]
    neg=[]
    for i in list(l):
        d=df[df['Stock']==i]
        pos.append(pearsonr(d.Twitter_Pos,d.News_Pos)[0])
        neu.append(pearsonr(d.Twitter_Neu,d.News_Neu)[0])
        neg.append(pearsonr(d.Twittter_Neg,d.News_Neg)[0])
    df_new['Pos']=pos
    df_new['Neg']=neg
    df_new['Neu']=neu
    return df_new
cor(df)