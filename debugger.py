import ast
import time
import json
import twint
import tweepy
from datetime import datetime, timedelta

# Edit your own Twitter API keys
consumer_key = "****"
consumer_secret = "****"
access_token = "****"
access_token_secret = "****"

# Lists
date_list = []
user_list = []
userid_list = []
deleted_list = []

# List of accounts that have issues
with open("users not found.txt", "r") as f:
    sort = []
    for line in f:
        sort.append(line[:-1])
    f.close()

# Tweepy auth
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, wait_on_rate_limit = True, wait_on_rate_limit_notify = True)

# Checking accounts again
for i in range(len(sort)):
    # Checking for Twitter accounts that are already deleted/shadowbanned using Twitter API
    # Twint is used to bypass limitations of Twitter API (30-day max)
    # Inconsistent results for Twint: https://github.com/twintproject/twint/issues/604
    try:
        api.get_user(int(sort[i]))
        userid_list.append(int(sort[i]))
    except Exception as e:
        error_code = ast.literal_eval(str(e))[0]["code"]
        if error_code == 50:
            error = "deleted"
        elif error_code == 63:
            error = "suspended"
        else:
            error = "missing"

        print(str(sort[i]) + "," + error)
        deleted_list.append(int(sort[i]))
        continue

# User features
with open("unfiltered dataset/cresci-stock-2018_tweets.json", "r") as f:
    sort = json.loads(f.readline())
    f.close()

for i in range(len(sort)):
    if sort[i]["user"]["id"] in userid_list:
        # Users that will be used
        date_list.append(sort[i]["created_at"])
        user_list.append(sort[i]["user"]["id"])

# Scraping tweets
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
            c.Output = "debug log.csv"
            c.Store_csv = True

            try:
                twint.run.Search(c)
            except Exception as e:
                # Twitter account not found; to be repeated until account is found by Twint
                print(e)
                print(user_list[i])

                # Sleep
                sleep_time = 60 # 1 min
                print(f"Sleeping for {sleep_time} seconds.")
                time.sleep(sleep_time)

                # Add to not found
                self.not_found_user.append(user_list[i])
                self.not_found_date.append(date_list[i])

        print("Finished scraping tweets from Twitter.")
        print(f"Number of users not found: {len(self.not_found_user)}")

print("Scraping tweets...")
t = Tweets(user_list, date_list)
