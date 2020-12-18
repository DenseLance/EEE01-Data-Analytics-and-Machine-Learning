import re
import json
import nltk
import string
from textblob import TextBlob
from datetime import datetime, timedelta
from langdetect import DetectorFactory, detect_langs

"""
Language detection algorithm is non-deterministic,
which means that if you try to run it on a text which is either too short or too ambiguous,
you might get different results everytime you run it.
Enforcing consistent results should be done.
"""
DetectorFactory.seed = 40

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

# Deleted users --> removing these users somehow increases accuracy
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

# Filtered user features
class User:
    def __init__(self, feature, date, bot):
        # Store user info
        self.dict = {}

        self.dict["id"] = feature["id"]
        self.dict["language"] = feature["lang"]
        
        self.account_labelling_date = datetime.strptime(date, "%a %b %d %X %z %Y")
        self.account_creation_date = datetime.strptime(feature["created_at"], "%a %b %d %X %z %Y")
        self.dict["number of days between account creation and account labelling"] = (self.account_labelling_date - self.account_creation_date).days

        self.name = feature["name"]
        self.dict["presence of digits in name"] = int(any(char.isdigit() for char in self.name))
        self.dict["presence of special characters (ASCII) in name"] = int(any(char in string.punctuation for char in self.name))
        self.dict["presence of non-ASCII characters in name"] = int(not self.name.isascii())
        self.dict["length of name"] = len(self.name)
        
        self.screen_name = feature["screen_name"]
        self.dict["presence of digits in screen name"] = int(any(char.isdigit() for char in self.screen_name))
        self.dict["length of screen name"] = len(self.screen_name)

        self.url = feature["url"]
        self.dict["presence of url"] = int(False if self.url is None else True)

        self.dict["number of followers"] = int(feature["followers_count"])
        self.dict["number of friends"] = int(feature["friends_count"])
        self.dict["number of listed"] = int(feature["listed_count"])
        self.dict["number of favourites"] = int(feature["favourites_count"])
        self.dict["number of statuses"] = int(feature["statuses_count"])

        self.dict["verified account"] = int(feature["verified"])

        try:
            self.banner = feature["profile_banner_url"]
        except KeyError:
            self.banner = None
        self.dict["presence of banner"] = int(False if self.banner is None else True)

        self.dict["presence of default profile"] = int(feature["default_profile"])
        self.dict["presence of default profile image"] = int(feature["default_profile_image"])
        
        self.description = feature["description"]

        # Remove URLs for basic feature extraction
        self.dict["number of urls in description"] = 0
        for url in self.description.split():
            if url.startswith("https://") or url.startswith("http://"):
                self.description = self.description.replace(url, "")
                self.dict["number of urls in description"] += 1

        # Basic feature extraction of description
        self.dict["number of hashtags in description"] = len(re.findall(r"#(\w+)", self.description))
        self.dict["number of cashtags in description"] = len(re.findall(r"\$(\w+)", self.description))
        self.dict["number of mentions in description"] = len(re.findall(r"@(\w+)", self.description))
        self.dict["number of words in description"] = len(self.description.split())
        self.dict["number of special characters (ASCII) in description"] = len([char for char in self.description if char in string.punctuation]) # include hashtags, cashtags, mentions
        self.dict["number of non-ASCII characters in description"] = len([char for char in self.description if char.isascii() == False])
        self.dict["number of uppercase characters in description"] = len([char for char in self.description if char.isupper() == True]) # indicative of user behaviour
        self.dict["number of digits in description"] = len([char for char in self.description if char.isdigit() == True])        
        
        # Pre-processing of description + detect language + sentiment analysis
        self.description_processed = self.description.lower()
        self.description_processed = re.sub(r"#(\w+)", "", self.description_processed) # remove hashtags
        self.description_processed = re.sub(r"\$(\w+)", "", self.description_processed) # remove cashtags
        self.description_processed = re.sub(r"@(\w+)", "", self.description_processed) # remove mentions

        try:
            self.description_language = [repr(lang)[:2] for lang in detect_langs(self.description)]
        except: # no language detected taken as english
            self.description_language = [feature["lang"]]
        self.dict["language in description is english"] = int(feature["lang"] in self.description_language)
        
        for char in string.punctuation:
            self.description_processed = self.description_processed.replace(char, "") # remove special characters (ASCII)
        self.description_processed = "".join([char for char in self.description_processed if ord(char) < 128]) # remove non-ASCII characters
        self.description_processed = " ".join(self.description_processed.split()) # format description
        
        self.sentiment = TextBlob(self.description_processed).sentiment # textblob is not that accurate
        self.dict["polarity of description"] = round(self.sentiment.polarity, 2) # 2 d.p.
        self.dict["subjectivity of description"] = round(self.sentiment.subjectivity, 2) # 2 d.p.

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

with open("filtered dataset/user based classification.csv", "w") as f:
    user_count = 0
    for i in range(len(user_list)):
        user = User(user_list[i], date_list[i], bot_list[i])
        
        # Write headers
        if i == 0:
            f.write(user.headers() + "\n")

        # Write user features
        if user.dict["language"] == "en":
            f.write(user.features() + "\n")
            user_count += 1
    
    f.close()

# Number of accounts used
print("User based classification: COMPLETED")
print(f"Number of accounts: {user_count}")
