# fake_news_classification
This is a course project for CSC2515(Machine Learning and Data Mining) using Naive Bayes to classify fake or real news headlines.

Data: Fake and real news headlines

We have compiled 1298 “fake news” headlines (which mostly include headlines of articles classified as biased etc.) and 1968 “real” news headlines, where the “fake news” headlines are from https://www.kaggle.com/mrisdal/fake-news/data and “real news” headlines are from https://www.kaggle.com/therohk/million-headlines. We further cleaned the data by removing words from fake news titles that are not a part of the headline, removing special characters from the headlines, and restricting real news headlines to those after October 2016 containing the word “trump”. For your interest, the cleaning script is available at clean_script.py, but you do not need to run it. The cleaned-up data is available below:

Real news headlines: clean_real.txt
Fake news headlines: clean_fake.txt
Each headline appears as a single line in the data file. Words in the headline are separated by spaces.


