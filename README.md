[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/Spurryag/Twitter_mining_miniproject/blob/master/Twitter_Project.ipynb)

# Summary: 

The purpose of this notebook is to conduct a basic twitter mining exercise and to demonstrate how to apply exploration techniques and machine learning algorithms for NLP . To do so, the notebook will first resort to the use of the twitter API to mine selected tweets, based on certain queries, before moving to explore the collected data and apply machine learning algorithms.

# Objectives:

* Crawl X tweets of N active politicians
* Create a classifier to assign tweets to politicians
* Investigate which politicians are similar to each other


# Results:

* With regards to classification: the implemented Neural Network (accuracy of 0.58) outperformed the Logistic Regression, Support Vector Machine (linear Kernel) and Random Forest. 
* To examine similarity of politicians, cosine similarity was used and reveals that the most similar politicians are: Adam Smith	Bob Corker (0.996126).
