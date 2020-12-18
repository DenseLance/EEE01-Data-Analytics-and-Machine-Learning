import re
import csv
import ast
import string
from textblob import TextBlob
from langdetect import DetectorFactory, detect_langs

"""
Language detection algorithm is non-deterministic,
which means that if you try to run it on a text which is either too short or too ambiguous,
you might get different results everytime you run it.
Enforcing consistent results should be done.
"""
DetectorFactory.seed = 0

# Labelling of accounts
with open("unfiltered dataset/cresci-stock-2018.tsv", "r") as f:
    data = []
    for line in f:
        data += line[:-1].split("	")
    f.close()

# ID-Bot dictionary
dataset = {data[i]: data[i + 1] for i in range(0, len(data), 2)}

# User-tweet dictionary
user_tweet_dict= {}

# Lists
user_list = []
tweet_list = []

class Tweet:
    def __init__(self, feature, bot):
        # Store tweet info
        self.dict = {}
        
        self.dict["id"] = feature["id"]
        
        self.user_id = feature["user_id"]
        self.conversation_id = feature["conversation_id"]
        self.dict["tweet is reply"] = int(self.dict["id"] != self.conversation_id)
        
        self.tweet = feature["tweet"]
        self.dict["number of hashtags in tweet"] = len(ast.literal_eval(feature["hashtags"]))
        self.dict["number of cashtags in tweet"] = len(ast.literal_eval(feature["cashtags"]))
        self.dict["number of mentions in tweet"] = len(ast.literal_eval(feature["mentions"]))
        self.dict["number of urls in tweet"] = len(ast.literal_eval(feature["urls"]))
        self.dict["number of photos in tweet"] = len(ast.literal_eval(feature["photos"]))
        self.dict["number of videos in tweet"] = int(feature["video"])
        self.dict["presence of quoted tweet in tweet"] = int(False if feature["quote_url"] is "" else True) # 0 means tweet is unavailable

        self.dict["number of replies to tweet"] = int(feature["replies_count"])
        self.dict["number of retweets to tweet"] = int(feature["retweets_count"])
        self.dict["number of likes to tweet"] = int(feature["likes_count"])

        # Remove URLs for basic feature extraction
        for url in self.tweet.split():
            if url.startswith("https://") or url.startswith("http://"):
                self.tweet = self.tweet.replace(url, "")

        # Basic feature extraction of tweets
        self.dict["number of words in tweet"] = len(self.tweet.split())
        self.dict["number of special characters (ASCII) in tweet"] = len([char for char in self.tweet if char in string.punctuation]) # include hashtags, cashtags, mentions
        self.dict["number of non-ASCII characters in tweet"] = len([char for char in self.tweet if char.isascii() == False])
        self.dict["number of uppercase characters in tweet"] = len([char for char in self.tweet if char.isupper() == True]) # indicative of user behaviour
        self.dict["number of digits in tweet"] = len([char for char in self.tweet if char.isdigit() == True])
        
        # Pre-processing of tweet + detect language + sentiment analysis
        self.tweet_processed = self.tweet.lower()
        self.tweet_processed = re.sub(r"#(\w+)", "", self.tweet_processed) # remove hashtags
        self.tweet_processed = re.sub(r"\$(\w+)", "", self.tweet_processed) # remove cashtags
        self.tweet_processed = re.sub(r"@(\w+)", "", self.tweet_processed) # remove mentions
        
        try:
            self.tweet_language = [repr(lang)[:2] for lang in detect_langs(self.tweet_processed)]
        except: # no language detected, hence taken as english
            self.tweet_language = ["en"]
        self.dict["language"] = self.tweet_language[0] # language detected that has highest confidence score
        self.dict["language in tweet is english"] = int("en" in self.tweet_language)
        
        for char in string.punctuation:
            self.tweet_processed = self.tweet_processed.replace(char, "") # remove special characters (ASCII)
        self.tweet_processed = "".join([char for char in self.tweet_processed if ord(char) < 128]) # remove non-ASCII characters
        self.tweet_processed = " ".join(self.tweet_processed.split()) # format tweet
        
        self.sentiment = TextBlob(self.tweet_processed).sentiment # textblob is not that accurate
        self.dict["polarity of tweet"] = round(self.sentiment.polarity, 2) # 2 d.p.
        self.dict["subjectivity of tweet"] = round(self.sentiment.subjectivity, 2) # 2 d.p.

        # Bot or not
        self.dict["bot"] = bot

    def headers(self):
        temp = ""

        for header in self.dict:
            temp += str(header) + ","

        return temp[:-1]

    def features(self):
        temp = ""

        for header in self.dict:
            temp += str(self.dict[header]) + ","

        return temp[:-1]

with open("filtered dataset/user based classification.csv", "r") as f:
    csv_reader_user = csv.DictReader(f, delimiter = ",")

    for row in csv_reader_user:
        user_list.append(row["id"]) # all users here set their main language to english

    f.close()

with open("unfiltered dataset/twint results.csv", "r", encoding = "utf-8") as f:
    csv_reader = csv.DictReader(f, delimiter = ",")

    for row in csv_reader:
        if row["user_id"] in user_list: # checks if user's main language is english
            bot_or_not = dataset[str(row["user_id"])].lower()
            if bot_or_not == "bot":
                tweet_list.append(Tweet(row, 1))
            elif bot_or_not == "human":
                tweet_list.append(Tweet(row, 0))
            else:
                tweet_list.append(Tweet(row, None))

    f.close()

print("Download tweets from twint results.csv: COMPLETED")

with open("filtered dataset/tweet based classification (by tweet).csv", "w", encoding = "utf-8") as f:
    # Write headers
    f.write(tweet_list[0].headers() + "\n")
    
    for tweet in tweet_list:
        # Write tweet features
        f.write(tweet.features() + "\n")

    f.close()

# Note: some users may not have tweets during the 100-day period, hence ignored
print("Tweet based classification (by tweet): COMPLETED")
print(f"Number of tweets: {len(tweet_list)}")

# Store user's tweets as a list
for tweet in tweet_list:
    try: # if user is in user_tweet_dict
        user_tweet_dict[tweet.user_id].append(tweet)
    except: # else create new user key
        user_tweet_dict[tweet.user_id] = [tweet]

with open("filtered dataset/tweet based classification (by user).csv", "w", encoding = "utf-8") as f:
    ignored_features = ["id", "language", "bot", "number of tweets"] # not averaged
    
    # Write headers
    headers = list(tweet_list[0].dict.keys())
    headers.remove("id")
    headers.remove("language") # remove id, language; place them at the front for easy referencing
    temp = "id,language,"

    for header in headers:
        temp += str(header) + ","

    f.write(temp + "number of tweets" + "\n")
    
    for user in user_tweet_dict:
        user_dict = {"id": user, "language": "en"}
        
        for tweet in user_tweet_dict[user]:
            for feature in tweet.dict:
                if feature not in ignored_features:
                    try: # if feature is in user_dict
                        user_dict[feature] += tweet.dict[feature]
                    except: # else create new feature key
                        user_dict[feature] = tweet.dict[feature]

        user_dict["bot"] = user_tweet_dict[user][0].dict["bot"]
        user_dict["number of tweets"] = len(user_tweet_dict[user])
        
        for feature in user_dict:
            if feature not in ignored_features:
                user_dict[feature] = round(user_dict[feature] / user_dict["number of tweets"], 2) # takes average of feature with respect to number of tweets
        
        temp = ""
        for header in user_dict:
            temp += str(user_dict[header]) + ","

        # Write user's tweet features
        f.write(temp[:-1] + "\n")
    
    f.close()

# Note: some users may not have tweets during the 100-day period, hence ignored
print("Tweet based classification (by user): COMPLETED")
print(f"Number of accounts: {len(user_tweet_dict)}")
