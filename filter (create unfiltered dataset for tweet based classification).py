import re
import time
import json
import twint
from datetime import datetime, timedelta

# Change warnings to error to catch it in try-except block
# Warning: ChunkedEncodingError due to forcibly being kicked out
##import warnings
##warnings.filterwarnings("error") # this throws more output regarding warnings ignored

# User features
with open("unfiltered dataset/cresci-stock-2018_tweets.json", "r") as f:
    sort = json.loads(f.readline())
    f.close()

# Lists
date_list = []
user_list = []
deleted_list = []

# Deleted users
with open("filtered dataset/deleted accounts.csv", "r") as f:
    for line in f:
        deleted_list.append(int(line[:-1].split(",")[0]))
    f.close()

for i in range(len(sort)):
    if sort[i]["user"]["id"] not in deleted_list:
        # Users that will be used
        date_list.append(sort[i]["created_at"])
        user_list.append(sort[i]["user"]["id"])

# Number of accounts used
print(f"Number of accounts: {len(user_list)}")

# Unfiltered tweet features
class Tweets:
    def __init__(self, user_list, date_list):
        # Some users cannot be found at some points in time due to scraping limitations
        self.not_found_user = []
        self.not_found_date = []
        
        for i in range(len(user_list)):
            print(f"Currently scraping tweets from: {user_list[i]}")
            
            c = twint.Config()
            c.User_id = str(user_list[i]) # Note: user is able to change their name/screen name

            date = datetime.strptime(date_list[i], "%a %b %d %X %z %Y")
            c.Until = date.strftime('%Y-%m-%d')
            c.Since = (date - timedelta(days = 100)).strftime('%Y-%m-%d')

            # Settings
            c.Filter_retweets = True # remove retweets, as they are not made by user
            c.Hide_output = True

            # Save to results
            c.Custom["tweet"] = ["id",
                                 "conversation_id",
                                 "created_at",
                                 "user_id",
                                 "username",
                                 "name",
                                 "place",
                                 "tweet",
                                 "mentions",
                                 "urls",
                                 "photos",
                                 "replies_count",
                                 "retweets_count",
                                 "likes_count",
                                 "hashtags",
                                 "cashtags",
                                 "link",
                                 "retweet",
                                 "quote_url",
                                 "video",
                                 "user_rt_id",
                                 "near",
                                 "geo",
                                 "source",
                                 "retweet_date"]
            c.Output = "unfiltered dataset/twint results.csv"
            c.Store_csv = True

            try:
                twint.run.Search(c)
            except Exception as e:
                # Twitter account not found; to be repeated until account is found by Twint
                print(e) ## TO BE REMOVED ##
                print(user_list[i]) ## TO BE REMOVED ##

                # Sleep
                sleep_time = 60 # 1 min
                print(f"Sleeping for {sleep_time} seconds.")
                time.sleep(sleep_time)

                # Add to not found
                self.not_found_user.append(user_list[i])
                self.not_found_date.append(date_list[i])

        print("Finished scraping tweets from Twitter.")
        print(f"Number of users not found: {len(self.not_found_user)}")

    def manual_checking(self): # send for manual checking
        with open("users not found.txt", "w") as f:
            for user in self.not_found_user:
                f.write(str(user) + "\n")
            f.close()

t = Tweets(user_list, date_list)
t.manual_checking()
