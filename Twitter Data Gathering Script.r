library(twitteR)

consumer_key <- 'XXXXXXX'
consumer_secret <- 'XXXXXXX'
access_token <- 'XXXXXXXX'
access_secret <- 'XXXXXXXX'
setup_twitter_oauth(consumer_key, consumer_secret, access_token, access_secret)

tweets_n <- searchTwitter("$Stock", n=10000,lang = "en")
test_tweets <- twListToDF(tweets_n)
write.csv(test_tweets, "Stock_name.csv")
