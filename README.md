# EEE01 Data Analytics and Machine Learning

**Dataset used**: cresci-stock-2018

**Description**: Automated accounts that act in coordinate fashion. Labels and user objects.

[[Download dataset here!](https://botometer.osome.iu.edu/bot-repository/datasets/cresci-stock-2018/cresci-stock-2018.tar.gz)]

**Citation:** Cresci, Stefano, Lillo, Fabrizio, Regoli, Daniele, Tardelli, Serena, & Tesconi, Maurizio. (2019). Cashtag Piggybacking dataset - Twitter dataset enriched with financial data [Data set]. Zenodo. http://doi.org/10.5281/zenodo.2686862



## For quick use

Follow steps in **classification**. Filtering is not required as we have already generated the filtered dataset for you.



**User based classification:** 12711 users; 6815 bots, 5896 humans *(53.6%/46.4%)*

**Tweet based classification by tweet:** 1221471 tweets; 36830 bots, 1184641 humans  *(3.0%/97.0%)*

**Tweet based classification by user:** 4316 users; 1276 bots, 3040 humans *(29.6%/70.4%)*



## Filter

Filtering of datasets should be done in this particular order:
1. `filter (remove deleted, suspended accounts).py`
2. `filter (create dataset for user based classification).py`
3. `filter (create unfiltered dataset for tweet based classification).py`
4. `filter (create dataset for tweet based classification).py`



## Classification
There are two types of classification that we are employing:
1. user based classification
2. tweet based classification
   - by tweet
   - by user



### User based classification

Follow steps 1 and 2 of **filter** if you would like to obtain the dataset from scratch.



Run `classification.py` to get results.



To switch to user based classification:

**Line 12:** `dataset = pd.read_csv("filtered dataset/user based classification.csv")`

**Line 88:** `plt.suptitle("User Based Classification", fontweight = "bold", fontsize = "x-large", x = 0.51, y = 0.99)`



### Tweet based classification

Follow steps 1, 3 and 4 of **filter** if you would like to obtain the dataset from scratch.



To switch between tweet based classification *by tweet*:

**Line 12:** `dataset = pd.read_csv("filtered dataset/tweet based classification (by tweet).csv")`

**Line 88:** `plt.suptitle("Tweet Based Classification (By Tweet)", fontweight = "bold", fontsize = "x-large", x = 0.51, y = 0.99)`



Run `classification.py` to get results.



To switch to tweet based classification *by user*:

**Line 12:** `dataset = pd.read_csv("filtered dataset/tweet based classification (by user).csv")`

**Line 88:** `plt.suptitle("Tweet Based Classification (By User)", fontweight = "bold", fontsize = "x-large", x = 0.51, y = 0.99)`



## Debugging

Debugging is used as last resort in case of certain errors. For manual debugging of `filter (create unfiltered dataset for tweet based classification).py`, use `debugger.py`. It is **recommended** to save the output in IDLE for future reference. Results are stored in `debug log.csv`.



After manual debugging, use `debug checker.py` to check for additional results scraped from Twitter during debugging. See comments in `debug checker.py` for more information.



To remove rows from `unfiltered dataset/twint results.csv`, use [Delimit](http://www.delimitware.com/index.html)'s extract function.

> Filter to apply when extracting rows: `$4 <> user_id`

Change `user_id` to the user_id of the user that you would like to delete.



To add rows to `unfiltered dataset/twint results.csv`, copy and paste from `debug log.csv`.



## To-do list

- [x] `filter (remove deleted, suspended accounts).py`
- [x] `filter (create dataset for user based classification).py`
- [x] `filter (create unfiltered dataset for tweet based classification).py`
- [x] `filter (create dataset for tweet based classification).py`



### Comments

- When we did tweet based classification by tweet, we found that the f1 measure is very low. Hence, we switched to average of all tweets made by user during the same 100-day period.
- We also note that our dataset is not large enough to sieve out the specific names of hashtags and cashtags in tweets and description etc as it may result in higher bias and an inaccurate algorithm.
- In the future we might add NULL values to accounts that do not tweet to `filtered dataset/tweet based classification (by tweet).csv`, as we note that there are a large number of bots that do not tweet when comparing our datasets for user based classification and tweet based classification.