import re
import json
from datetime import datetime, timedelta
import nltk
from nltk.corpus import stopwords
from unidecode import unidecode

# Labelling of accounts
with open("unfiltered dataset/cresci-stock-2018.tsv", "r") as f:
    data = []
    for line in f:
        data += line[:-1].split("	")
    f.close()

# ID-Bot dictionary
dataset = {data[i]: data[i + 1] for i in range(0, len(data), 2)}

# User features
with open("unfiltered dataset/cresci-stock-2018_tweets.json", "r") as f:
    sort = json.loads(f.readline())
    f.close()

# Lists
bot_list = []
date_list = []
user_list = []
deleted_list = []

# Deleted users --> removing this somehow increases accuracy
with open("filtered dataset/deleted accounts.csv", "r") as f:
    for line in f:
        deleted_list.append(int(line[:-1].split(",")[0]))
    f.close()

for i in range(len(sort)):
    if sort[i]["user"]["id"] not in deleted_list:
        # Users that will be used
        date_list.append(sort[i]["created_at"])
        user_list.append(sort[i]["user"])
        bot_or_not = dataset[str(sort[i]["user"]["id"])].lower()
        if bot_or_not == "bot":
            bot_list.append(1)
        elif bot_or_not == "human":
            bot_list.append(0)
        else:
            bot_list.append(None)

# Number of accounts used
print(f"Number of accounts: {len(user_list)}")

# Filtered user features
class User:
    def __init__(self, feature, date, bot):
        self.id = feature["id"]
        self.account_labelling_date = datetime.strptime(date, "%a %b %d %X %z %Y")
        self.account_creation_date = datetime.strptime(feature["created_at"],  "%a %b %d %X %z %Y")
        self.name = feature["name"]
        self.screen_name = feature["screen_name"]
        self.description = feature["description"]
        self.url = feature["url"]
        try:
            self.banner = feature["profile_banner_url"]
        except KeyError:
            self.banner = None
          
                    
        # Used features
        self.bot = bot
        
        self.days = (self.account_labelling_date - self.account_creation_date).days # days between account creation and account labelling

        self.name_int = int(any(char.isdigit() for char in self.name))
        self.name_special = int(not self.name.isascii())
        
        self.screen_name_int = int(any(char.isdigit() for char in self.screen_name))
        
        self.description_special = int(not self.description.isascii())
        self.description_hashtag = len(re.findall(r"#(\w+)", self.description))
        
        # --> insert text analysis here

        self.url_presence = int(False if self.url is None else True)

        self.followers_count = int(feature["followers_count"])
        self.friends_count = int(feature["friends_count"])
        self.listed_count = int(feature["listed_count"])
        self.favourites_count = int(feature["favourites_count"])
        self.statuses_count = int(feature["statuses_count"])

        self.verified = int(feature["verified"])

        self.banner_presence = int(False if self.banner is None else True)

        self.default_profile = int(feature["default_profile"])
        self.default_profile_image = int(feature["default_profile_image"])

        # Full list
        self.list = [self.id,
                     self.days,
                     self.name_int,
                     self.name_special,
                     self.screen_name_int,
                     self.description_special,
                     self.description_hashtag,
                     self.url_presence,
                     self.followers_count,
                     self.friends_count,
                     self.listed_count,
                     self.favourites_count,
                     self.statuses_count,
                     self.verified,
                     self.banner_presence,
                     self.default_profile,
                     self.default_profile_image,
                     self.bot]

    def print(self):
        temp = ""
        
        for item in self.list:
            temp += str(item)
            temp += ","
        
        temp = temp[:-1]
        return temp

with open("filtered dataset/user based classification.csv", "w") as f:
    f.write("id,number of days between account creation and account labelling,integer in name,special character in name,integer in screen name,special character in description,number of hashtags in description,url,number of followers,number of friends,number of listed,number of favourites,number of statuses,verified,banner,default profile,default profile image,bot" + "\n")
    
    for i in range(len(user_list)):
        user = User(user_list[i], date_list[i], bot_list[i])
        f.write(user.print() + "\n")
    
    f.close()
