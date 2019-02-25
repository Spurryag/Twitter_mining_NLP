#import packages
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import random
import spacy
import nltk
from nltk.tokenize.toktok import ToktokTokenizer
import re
from bs4 import BeautifulSoup
import unicodedata
from collections import defaultdict
import string
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from sklearn.metrics import log_loss
from tqdm import tqdm
stopwords = stopwords.words('english')
sns.set_context('notebook')
#import Bi-Directional LSTM 
from keras.models import Sequential
from keras.layers import CuDNNLSTM, Dense, Bidirectional
import math         

#Import the datasets
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
sub = pd.read_csv('../input/sample_submission.csv')

#Define the split for the dataset
train, val_df = train_test_split(train, test_size=0.1)

#Define the embedding to be used
embeddings_index = {}
f = open('../input/embeddings/glove.840B.300d/glove.840B.300d.txt')
for line in tqdm(f):
    values = line.split(" ")
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

# Convert values to embeddings
def text_to_array(text):
    empyt_emb = np.zeros(300)
    text = text[:-1].split()[:30]
    embeds = [embeddings_index.get(x, empyt_emb) for x in text]
    embeds+= [empyt_emb] * (30 - len(embeds))
    return np.array(embeds)

# train_vects = [text_to_array(X_text) for X_text in tqdm(train["question_text"])]
val_vects = np.array([text_to_array(X_text) for X_text in tqdm(val_df["question_text"][:3000])])
val_y = np.array(val_df["target"][:3000])

## Data providers
batch_size = 128

def batch_gen(train):
    n_batches = math.ceil(len(train) / batch_size)
    while True: 
        train = train.sample(frac=1.)  # Shuffle the data.
        for i in range(n_batches):
            texts = train.iloc[i*batch_size:(i+1)*batch_size, 1]
            text_arr = np.array([text_to_array(text) for text in texts])
            yield text_arr, np.array(train["target"][i*batch_size:(i+1)*batch_size])
            
            
#Define the model architecture
model = Sequential()
model.add(Bidirectional(CuDNNLSTM(64, return_sequences=True),
                        input_shape=(30, 300)))
model.add(Bidirectional(CuDNNLSTM(64)))
model.add(Dense(1, activation="sigmoid"))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])



mg = batch_gen(train)
#remember to change the number of epochs
model.fit_generator(mg, epochs=10,
                    steps_per_epoch=100,
                    validation_data=(val_vects, val_y),
                    verbose= True)



# prediction part
batch_size = 256
def batch_gen(test):
    n_batches = math.ceil(len(test) / batch_size)
    for i in range(n_batches):
        texts = test.iloc[i*batch_size:(i+1)*batch_size, 1]
        text_arr = np.array([text_to_array(text) for text in texts])
        yield text_arr

test = pd.read_csv("../input/test.csv")

all_preds = []
for x in tqdm(batch_gen(test)):
    all_preds.extend(model.predict(x).flatten())

#Submit predictions
y_te = (np.array(all_preds) > 0.5).astype(np.int)
submit_df = pd.DataFrame({"qid": test["qid"], "prediction": y_te})
submit_df.to_csv("submission.csv", index=False)