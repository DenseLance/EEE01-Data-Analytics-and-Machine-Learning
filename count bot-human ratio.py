with open("filtered dataset/user based classification.csv", "r") as f:
    f.readline()

    total = 0
    bot = 0
    human = 0
    for line in f:
        data = line[:-1].split(",")
        temp = int(data[-1])
        # temp = int(data[-2]) # for tweet based classification (by user)
        total += 1
        if temp == 1:
            bot += 1
        elif temp == 0:
            human += 1
        else:
            print("ERROR ALERT")
        
    f.close()

print("Total:", total)
print("Bots:", bot)
print("Humans:", human)
print("Percentage are bots:", round(bot / total * 100, 1))
print("Percentage are humans:", round(human / total * 100, 1))
