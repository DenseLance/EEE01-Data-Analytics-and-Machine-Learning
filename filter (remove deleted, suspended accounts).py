import ast
import json
import tweepy

# Edit your own Twitter API keys
consumer_key = "****"
consumer_secret = "****"
access_token = "****"
access_token_secret = "****"

# User features
with open("unfiltered dataset/cresci-stock-2018_tweets.json", "r") as f:
    sort = json.loads(f.readline())
    f.close()

# Tweepy auth
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, wait_on_rate_limit = True, wait_on_rate_limit_notify = True)

# Check for dataset
for i in range(len(sort)):
    # Checking for Twitter accounts that are already deleted/shadowbanned using Twitter API
    # Twint is used to bypass limitations of Twitter API (30-day max)
    # Inconsistent results for Twint: https://github.com/twintproject/twint/issues/604
    try:
        api.get_user(int(sort[i]["user"]["id"]))
    except Exception as e:
        error_code = ast.literal_eval(str(e))[0]["code"]
        if error_code == 50:
            error = "deleted"
        elif error_code == 63:
            error = "suspended"
        else:
            error = "missing"
        with open("filtered dataset/deleted accounts.csv", "a") as f:
            f.write(str(sort[i]["user"]["id"]) + "," + error + "\n")
            f.close()
        continue
