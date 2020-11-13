import csv
import ast

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

        # Used features
        self.bot = bot

        self.is_reply = int(self.id != self.conversation_id)

        self.tweet_int = int(any(char.isdigit() for char in self.tweet))
        self.tweet_special = int(not self.tweet.isascii())
        # --> insert text analysis here

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
        
        # Full list
        self.list = [self.id,
                     self.is_reply,
                     self.tweet_int,
                     self.tweet_special,
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
                     self.bot]

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

with open("filtered dataset/tweet based classification (by tweet).csv", "w") as f:
    f.write("id,is reply,integer in tweet,special character in tweet,number of mentions,number of urls,number of photos,number of replies,number of retweets,number of likes,number of hashtags,number of cashtags,quoting other tweets,video,bot" + "\n")

    for tweet in tweet_list:
        f.write(tweet.print() + "\n")

    f.close()

with open("filtered dataset/tweet based classification (by user).csv", "w") as f:
    f.write("id,is reply,integer in tweet,special character in tweet,number of mentions,number of urls,number of photos,number of replies,number of retweets,number of likes,number of hashtags,number of cashtags,quoting other tweets,video,bot,number of tweets" + "\n")

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
