library(NLP)
library(twitteR)
library(syuzhet)
library(syuzhet)
library(tm)
library(SnowballC)
library(stringi)
library(topicmodels)
library(ROAuth)
library(ggplot2)

consumer_key <- 'rZArj3q9ko4yYKMSejRsbLKL4'
consumer_secret <- 'KtIICeKO47Os3F8wKgSoWpcAyxMjZBBWjijdnutGyHIxoGMnPG'
access_token <- '705161064451682305-Hu2MEo79VIXHhm9pVsYZG9pcEKGNEW4'
access_secret <- 'q3Um1i1oRk5FVfV4l1WKlR6Xb8ygHrcAAtboqKZth0IIy'
setup_twitter_oauth(consumer_key, consumer_secret, access_token, access_secret)

tweets_n <- searchTwitter("$Stock", n=10000,lang = "en")
test_tweets <- twListToDF(tweets_n)