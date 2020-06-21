import nltk
import random
import os
from nltk.corpus import stopwords
import string
from collections import Counter
from nltk.corpus import stopwords
from string import punctuation
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from os import listdir
import glob
import re
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Embedding
from keras.layers import Conv1D, MaxPooling1D

# this is the joint all opinions, one opinion one row
# load doc into memory
def load_doc(filename):
# open the file as read only
    file = open(filename, 'r', encoding="utf8", errors='ignore')
# read all text
    text = file.read()
# close the file
    file.close()
    return text
#Customize sort
pat=re.compile("(\d+)\D*$")
def key_func(x):
        mat=pat.search(os.path.split(x)[-1]) # match last group of digits
        if mat is None:
            return x
        return "{:>10}".format(mat.group(1))

#clean integers
def my_preprocessor(text):
    text=re.sub("\\W"," ",text) # remove special chars
    #text=re.sub("\\s+(in|the|all|for|and|on)\\s+"," _connector_ ",text) # normalize certain words
    tokens=word_tokenize(text)
    tokens = [token.lower() for token in tokens]
    #stop_words = set(stopwords.words('english'))
    #tokens = [w for w in tokens if not w in stop_words]
    return ' '.join(tokens)

# load doc, clean and return line of tokens
def doc_to_line(filename):
# load the doc
    text = load_doc(filename)
# clean doc
    tokens = my_preprocessor(text)
    return ''.join(tokens)
# load all docs in a directory
def process_docs(directory):
    lines = list()
# walk through all files in the folder
    for filename in sorted(listdir(directory), key= key_func):
# create the full path of the file to open
        path = directory + '/' + filename 
# load and clean the doc
        line = doc_to_line(path)
# add to list
        lines.append(line)
    return lines
#Save the doc
def save_list(lines, filename):
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
# prepare training/test reviews
text_lines = process_docs('/Users/cecilia/Desktop/movie/totest/')
save_list(text_lines, 'TTTest.txt')

A = sorted(glob.glob1("/Users/cecilia/Desktop/movie/totest/", "*"), key = key_func)
y=[int(re.match('.*(?:\D|^)(\d+)', i).group(1)) for i in A]

with open("testlab.txt", "w") as f:
    for i in y:
        f.write(str(i) +"\n")

text_lines = process_docs('/Users/cecilia/Desktop/movie/totrain/')
save_list(text_lines, 'TTTrain.txt')

# charging the labels train
# charging the target
B = sorted(glob.glob1("/Users/cecilia/Desktop/movie/totrain/", "*"), key = key_func)
y=[int(re.match('.*(?:\D|^)(\d+)', i).group(1)) for i in B]

with open("tt.txt", "w") as f:
    for i in y:
        f.write(str(i) +"\n")

#charging the labels
y=[]
with open("/Users/cecilia/Desktop/movie/tt.txt", "r") as f:
    for line in f:
        y.append(int(line.strip()))

#checking
print(y[-1])

#charging the corpus
reviews=[]
with open("/Users/cecilia/Desktop/movie/TTTrain.txt") as total: 
    reviews = ([(review) for review in enumerate(total.readlines())])

#checking
print(reviews[-1])
#assigning the labels to a vector
y_train= [columm[1] for columm in reviews]
#assigning the corpus to a vector
train= [columm[0] for columm in reviews]
#working on the training dataset
#vectorizing my corpus
from sklearn.feature_extraction.text import TfidfVectorizer 
from keras.preprocessing import sequence 
# set settings and fit the count vectorizer min_df=5
tfidf_vec=TfidfVectorizer(min_df=5, stop_words="english")
tf_idf=tfidf_vec.fit_transform(train)
# Calculating the cosin similarity 
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel
cosine_similarities = linear_kernel(tf_idf[0:1], tf_idf).flatten()
related_docs_indices = cosine_similarities.argsort()[:-5:-1]
related_docs_indices
#Make sense of the tf_idf representation
def top_tfidf_feats(row, features, top_n=25):
    topn_ids = np.argsort(row)[::-1][:top_n]
    top_feats = [(features[i], row[i]) for i in topn_ids]
    df = pd.DataFrame(top_feats)
    df.columns = ['feature', 'tfidf']
    return df

def top_feats_in_doc(Xtr, features, row_id, top_n=25):
  row = np.squeeze(Xtr[row_id].toarray())
  return top_tfidf_feats(row, features, top_n)
#showing the ten most important features in the first movie review
top_feats_in_doc(tf_idf, features, 0, 10)
#the averge of most important word of all documents
def top_mean_feats(Xtr, features, grp_ids=None, min_tfidf=0.1, top_n=25):
    if grp_ids:
        D = Xtr[grp_ids].toarray()
    else:
        D = Xtr.toarray()
    D[D < min_tfidf] = 0
    tfidf_means = np.mean(D, axis=0)
    return top_tfidf_feats(tfidf_means, features, top_n)

top_mean_feats(tf_idf, features, top_n=20)
#ploting the top 15 most important features by each label
def top_feats_by_class(Xtr, y, features, min_tfidf=0.1, top_n=25):
    dfs = []
    labels = np.unique(y)
    for label in labels:
        ids = np.where(y==label)
        feats_df = top_mean_feats(Xtr, features, ids, min_tfidf=min_tfidf, top_n=top_n)
        feats_df.label = label
        dfs.append(feats_df)
    return dfs

def plot_tfidf_classfeats_h(dfs):
    fig = plt.figure(figsize=(12, 9), facecolor="w")
    x = np.arange(len(dfs[0]))
    for i, df in enumerate(dfs):
        ax = fig.add_subplot(1, len(dfs), i+1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_frame_on(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        ax.set_xlabel("Mean Tf-Idf Score", labelpad=10, fontsize=10)
        ax.set_title("label = " + str(df.label), fontsize=10)
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-2,2))
        ax.barh(x, df.tfidf, align='center', color='#3F5D7D')
        ax.set_yticks(x)
        ax.set_ylim([-1, x[-1]+1])
        yticks = ax.set_yticklabels(df.feature)
        plt.subplots_adjust(bottom=0.09, right=0.97, left=0.15, top=0.95, wspace=0.52)
    plt.show()
plot_tfidf_classfeats_h(dfs)

#the CNN model
#the training data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train)
sequences = tokenizer.texts_to_matrix(train, mode='tfidf')
word_index = tokenizer.word_index
x_train = pad_sequences(sequences, maxlen=25000)

#the labels
# prepare target
def prepare_targets(y_train):
    le = LabelEncoder()
    le.fit(y_train)
    y_train_enc = le.transform(y_train)
    return y_train_enc

y= prepare_targets(y_train)

# building the embedding layer and loading the pre-trained weights
#using pre traned data
import numpy as np
#preparing the embedding layer
import os
embeddings_index = {}
f = open(os.path.join("/content/drive/My Drive/", 'glove.6B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))
# Create a word_index to compute the matrice embeddings
embedding_matrix = np.zeros((len(word_index) + 1, 100))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

#CNN architecture
myclass= Sequential()
myclass.add(Embedding(len(word_index) + 1,100, weights=[embedding_matrix], input_length=25000, trainable=False))
myclass.add(Conv1D(100, kernel_size=5, activation='relu'))
myclass.add(MaxPooling1D(5))
myclass.add(Conv1D(100, kernel_size=5, activation='relu'))
myclass.add(MaxPooling1D(5))
myclass.add(Conv1D(100, kernel_size=5, activation='relu')) 
myclass.add(MaxPooling1D(25))
myclass.add(Flatten())
myclass.add(Dense(units=100, activation='relu'))
myclass.add(Dense(8, activation='softmax'))
myclass.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])

myclass.summary()

MModel=myclass.fit(x_train, labels, epochs=5, batch_size=128,validation_split=0.3)

#plot of accuracy and loss for validation
import matplotlib.pyplot as plt
plt.plot(MModel.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Validation'], loc='upper left')
plt.show()

plt.plot(MModel.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Validation'], loc='upper left')
plt.show()

#plot of accuracy and loss for training
import matplotlib.pyplot as plt
plt.plot(MModel.history['acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train'], loc='upper left')
plt.show()

plt.plot(MModel.history['loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train'], loc='upper left')
plt.show()

#saving the model
myclass.save('/content/drive/My Drive/mylast.h5')
###########################
#the second part of the project test the trained model
from keras.models import load_model
mymodel = load_model('/content/drive/My Drive/mylast.h5')
#charging the labels
y1=[]
with open("/content/drive/My Drive/testlab.txt", "r") as f:
    for line in f:
        y1.append(int(line.strip()))

#charging the corpus
opinions=[]
with open("/content/drive/My Drive/TTTest.txt") as totale: 
    opinions = ([(opinion, y1[k]) for k,opinion in enumerate(totale.readlines())])

# preparing the labels
def prepare_targets(y_train):
    le = LabelEncoder()
    le.fit(y_train)
    y_train_enc = le.transform(y_train)
    return y_train_enc

y= prepare_targets(y_t)

#preparing the corpus
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
token = Tokenizer()
token.fit_on_texts(x_t)
seq = token.texts_to_matrix(x_t, mode='tfidf')
x_test = pad_sequences(seq, maxlen=25000)

#make the predictions
classes = mymodel.predict_classes(x_test, batch_size=100)

#calculating the confussion matrix and plot
from sklearn.metrics import confusion_matrix
con_mat = confusion_matrix(y, classes)

import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
df_cm = pd.DataFrame(con_mat, range(8), range(8))
sn.set(font_scale=1.2) # for label size
sn.heatmap(con_mat, annot=True, annot_kws={"size": 12}, fmt="d") # font size
plt.show()

#calculating the precision and recall
def precision(label, confusion_matrix):
    col = confusion_matrix[:,label]
    return confusion_matrix[label, label] / col.sum()
    
def recall(label, confusion_matrix):
    row = confusion_matrix[label, :]
    return confusion_matrix[label, label] / row.sum()

def precision_macro_average(confusion_matrix):
    rows, columns = confusion_matrix.shape
    sum_of_precisions = 0
    for label in range(rows):
        sum_of_precisions += precision(label, confusion_matrix)
    return sum_of_precisions / rows

def recall_macro_average(confusion_matrix):
    rows, columns = confusion_matrix.shape
    sum_of_recalls = 0
    for label in range(columns):
        sum_of_recalls += recall(label, confusion_matrix)
    return sum_of_recalls / columns

print("label precision recall")
for y in range(8):
    print(f"{y:5d} {precision(y, con_mat):9.3f} {recall(y, con_mat):6.3f}")

print("recall total:", recall_macro_average(con_mat))

#calculating the accuracy level
def accuracy(confusion_matrix):
    diagonal_sum = confusion_matrix.trace()
    sum_of_all_elements = confusion_matrix.sum()
    return diagonal_sum / sum_of_all_elements 

labels= to_categorical(y, num_classes=8)
