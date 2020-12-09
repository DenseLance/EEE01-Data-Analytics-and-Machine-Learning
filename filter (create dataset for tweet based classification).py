import csv
import ast
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from textblob import TextBlob
from google_trans_new import google_translator


# Labelling of accounts
with open("unfiltered dataset/cresci-stock-2018.tsv", "r") as f:
    data = []
    for line in f:
        data += line[:-1].split("	")
    f.close()

# ID-Bot dictionary
dataset = {data[i]: data[i + 1] for i in range(0, len(data), 2)}

# Lists
user_list = []
tweet_list = []

class Tweet:
    def __init__(self, feature, bot):
        self.id = feature["id"]
        self.user_id = feature["user_id"]
        self.conversation_id = feature["conversation_id"]
        self.tweet = feature["tweet"]
        self.mentions = ast.literal_eval(feature["mentions"])
        self.urls = ast.literal_eval(feature["urls"])
        self.photos = ast.literal_eval(feature["photos"])
        self.hashtags = ast.literal_eval(feature["hashtags"])
        self.cashtags = ast.literal_eval(feature["cashtags"])
        self.quote_url = feature["quote_url"]
        self.punctuation_count = len([char for char in self.tweet if char in string.punctuation])
        
        # Used features
        self.bot = bot
        self.is_reply = int(self.id != self.conversation_id)

        self.mentions_count = len(self.mentions)
        self.urls_count = len(self.urls)
        self.photos_count = len(self.photos)

        self.replies_count = int(feature["replies_count"])
        self.retweets_count = int(feature["retweets_count"])
        self.likes_count = int(feature["likes_count"])

        self.hashtags_count = len(self.hashtags)
        self.cashtags_count = len(self.cashtags)

        self.quote_url_presence = int(False if self.quote_url is "" else True) # 0 means tweet is unavailable

        self.video = int(feature["video"])

        # Basic feature extraction of tweets
        
        self.word_count = len(self.tweet.split())
        self.char_count = len(self.tweet) - self.tweet.count(" ") - self.punctuation_count # to be edited --> include https currently
        self.nonascii_count = len([char for char in self.tweet if char.isascii() == False])
        self.avg_word_length = self.char_count / self.word_count
        self.uppercase_count = len([char for char in self.tweet if char.isupper() == True])
        self.numerics_count = len([char for char in self.tweet if char.isdigit() == True])

        # Translation of tweets
        
        translator = google_translator()
        
        try: 
            if translator.detect(self.tweet)[0] != 'en':
                self.tweet_processed = translator.translate(self.tweet)    
            else:
                self.tweet_processed = self.tweet
            self.translated = int(translator.detect(self.tweet)[0] != 'en')
##            print(self.translated)
        except Exception as e:
            print (e)
            self.tweet_processed = self.tweet
            self.translated = int(self.tweet_processed != self.tweet)
            print (self.user_id, ":", self.tweet_processed)

        # Pre-processing of tweets

        self.tweet_processed = self.tweet.lower()

        for url in self.tweet_processed.split():
            if url.startswith('https'):
                self.tweet_processed = self.tweet_processed.replace(url,'')

        for user in self.tweet_processed.split():
            if user.startswith('@'):
                self.tweet_processed = self.tweet_processed.replace(user,'')

        for char in string.punctuation:
            self.tweet_processed = self.tweet_processed.replace(char,'')
        
        tokens = nltk.word_tokenize(self.tweet_processed)
        freq_dist = nltk.FreqDist(tokens)
        
        common_words = str(freq_dist.keys())[:5] # problem: not all tweets in eng
        for word in common_words:
            self.tweet_processed = self.tweet_processed.replace(common_words,'')
        
        rare_words = str(freq_dist.keys())[-5:] # problem: not all tweets in eng
        for word in rare_words:
            self.tweet_processed = self.tweet_processed.replace(rare_words,'')
        
        # Sentiment analysis
        
        self.sentiment_polarity = TextBlob(self.tweet_processed).sentiment[0]
        self.sentiment_subjectivity = TextBlob(self.tweet_processed).sentiment[1]
        
        # Full list
        self.list = [self.id,
                     self.is_reply,
                     self.word_count,
                     self.char_count,
                     self.nonascii_count,
                     self.avg_word_length,
                     self.uppercase_count,
                     self.numerics_count,
                     self.mentions_count,
                     self.urls_count,
                     self.photos_count,
                     self.replies_count,
                     self.retweets_count,
                     self.likes_count,
                     self.hashtags_count,
                     self.cashtags_count,
                     self.quote_url_presence,
                     self.video,
                     self.bot,
                     self.translated,
                     self.sentiment_polarity,
                     self.sentiment_subjectivity,
                     self.tweet_processed]

    def print(self):
        temp = ""
        
        for item in self.list:
            temp += str(item)
            temp += ","
        
        temp = temp[:-1]
        return temp

with open("unfiltered dataset/twint results.csv", "r", encoding = "utf-8") as f:
    csv_reader = csv.DictReader(f, delimiter = ",")
    for row in csv_reader:
        bot_or_not = dataset[str(row["user_id"])].lower()
        if bot_or_not == "bot":
            tweet_list.append(Tweet(row, 1))
        elif bot_or_not == "human":
            tweet_list.append(Tweet(row, 0))
        else:
            tweet_list.append(Tweet(row, None))
    f.close()

with open("filtered dataset/tweet based classification (by tweet).csv", "w", encoding = "utf-8") as f:
    f.write("id,is reply,word count in tweet,number of characters,number of non-acscii characters,average word length,number of uppercase words,number of numerics,number of mentions,number of urls,number of photos,number of replies,number of retweets,number of likes,number of hashtags,number of cashtags,quoting other tweets,video,bot,translation of tweet,polarity of tweet,subjectivity of tweet,processed tweet" + "\n")

    for tweet in tweet_list:
        f.write(tweet.print() + "\n")

    f.close()

with open("filtered dataset/tweet based classification (by user).csv", "w", encoding = "utf-8") as f:
    f.write("id,is reply,word count in tweet,number of characters,number of non-acscii characters,average word length,number of uppercase words,number of numerics,number of mentions,number of urls,number of photos,number of replies,number of retweets,number of likes,number of hashtags,number of cashtags,quoting other tweets,video,bot,translation of tweet,polarity of tweet,subjectivity of tweet,processed tweet" + "\n")

    curr_user = ""
    curr_data = []
    counter = 0
        
    for tweet in tweet_list:
        user = str(tweet.user_id)

        if curr_user == "":
            counter += 1
            curr_user = user
            curr_data = tweet.list
            curr_data[0] = user
        elif user == curr_user:
            counter += 1
            for i in range(1, len(tweet.list) - 1): # ignore id, bot
                curr_data[i] += tweet.list[i]
        else: # takes average; moves on to next user
            for i in range(1, len(curr_data) - 1): # ignore id, bot
                curr_data[i] = round(curr_data[i] / counter, 2) # 2 d.p.

            # add additional feature: number of tweets during 100-day period
            curr_data += [counter]
            
            user_list.append(curr_data)

            counter = 1
            curr_user = user
            curr_data = tweet.list
            curr_data[0] = user

    for user in user_list:
        temp = ""
        
        for feature in user:
            temp += str(feature)
            temp += ","
        
        temp = temp[:-1]
        f.write(temp + "\n")
    
    f.close()

# Note: some users may not have tweets during the 100-day period, hence ignored
print(f"Number of tweets: {len(tweet_list)}")
print(f"Number of accounts: {len(user_list)}")
