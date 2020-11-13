import csv

# Users not found during filtering
with open("users not found.txt", "r") as f:
    users = []

    for line in f:
        users.append(int(line[:-1]))
    f.close()

# Make reference to dataset
with open("unfiltered dataset/twint results.csv", "r", encoding = "utf-8") as f:
    tweets = []

    csv_reader = csv.DictReader(f, delimiter = ",")
    for row in csv_reader:
        tweets.append(row)
    f.close()

# Number of tweets that entered the dataset even though there was error during filtering for that user (which are to be removed)
problem_users = {}

for user in users:
    print(f"User: {user}")
    tweets_dataset = []
    for tweet in tweets:
        if int(tweet["user_id"]) == user:
            tweets_dataset.append(tweet)

    if len(tweets_dataset) > 0:
        problem_users[user] = tweets_dataset

    print(f"Number of tweets in twint results.csv: {len(tweets_dataset)}")

