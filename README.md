[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/Spurryag/Twitter_mining_miniproject/blob/master/Twitter_Project.ipynb)

# The Problem : 

Conduct a basic webscraping as well as twitter mining exercise and to demonstrate how to apply exploration techniques combined with machine learning algorithms for NLP (Tu Kaiserslauten PhD interview project/ German Deep Learning Research Centre Research Position). 

More specifically, the aims are to : 

* Crawl X tweets of N active politicians
* Create a classifier to assign tweets to politicians
* Investigate which politicians are similar to each other

# My Approach: 

The twitter API was used to mine selected tweets, based on certain queries, before analysing the collected data and applying relevant machine learning algorithms.

# My Results:

* With regards to classification: the implemented Neural Network (accuracy of 0.58) outperformed the Logistic Regression, Support Vector Machine (linear Kernel) and Random Forest. 
* To examine similarity of politicians, cosine similarity was used and reveals that the most similar politicians are: Adam Smith and	Bob Corker (0.996126).
