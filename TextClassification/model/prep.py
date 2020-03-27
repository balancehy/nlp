#!/usr/bin/env python
import numpy as np
import os, sys
import matplotlib.pyplot as plt
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import collections
from scipy.sparse import coo_matrix

def load_data(root):
    """
    Load data as concatenated lines.
    Inputs:
        root: root path of data.
    Return:
        dictionary whose keys are topics, values are list of strings.
    """
    contents = {} # {topic : documents}
    
    for topic in os.listdir(root):
        if topic[0] is not '.': # not hidden folders
            fpath = root+'/'+topic
            docs = []
            for file in os.listdir(fpath):
                with open(fpath+'/'+file, 'r') as f:
                    docs.append( "".join(f.readlines()) )
            contents[topic] = docs.copy()

    return contents

def analyze_data(data_raw, data_clean):
    """
    Plot the document number in each topic. Plot the histgram of document length counted by characters
    Input:
        data_raw: dictionary whose keys are topics, values are list of strings.
    """
    topic_docnum = {}
    doclen = []
    for topic, docs in data_raw.items():
        topic_docnum[topic] = len(docs)
        for doc in docs:
            doclen.append(len(doc))
    doclen = np.array(doclen)
    
    fig = plt.figure(0)
    ax = fig.add_axes([0,0,1,1])
    ax.barh(list(topic_docnum.keys()), list(topic_docnum.values()), 0.4)
    ax.set_title("Numer of documents in each topic.")
    ax.set_xlabel("Counts")
    plt.show()
    
    print("Maxiumn and minimum document length counted by charactors: ", np.max(doclen), np.min(doclen))
    print("Number of unique document length: ", len(doclen))
    
    plt.figure(1)
    hist = plt.hist(doclen, bins=50, range=(0, 10000))
    plt.title("Histogram of document length")
    plt.xlabel("Document length(char)")
    plt.ylabel("Counts")
    plt.show()

    doclen_word = doc_length_word(data_clean)
    print("Maximum document length(word): ", np.max(doclen_word))
    plt.figure(2)
    hist = plt.hist(doclen_word, bins=50, range=(0, 1500))
    plt.title("Histogram of document length")
    plt.xlabel("Document length(word)")
    plt.ylabel("Counts")
    plt.show()
    
def doc_length_word(data_clean):
    doclen = []
    for topic, docs in data_clean.items():
        for doc in docs:
            doclen.append(len(doc))
    doclen = np.array(doclen)
    
    return doclen
    
        
def clean_data(data_raw, filter_size=100):
    """
    Clean each document by tokenizing, lemmatizing, removing punctuations, stopwords, single characters, numbers.
    Inputs:
        data_raw: dictionary whose keys are topics, values are list of strings.
        filter_size: if the most common words in a document has larger than "filter_size" frequency(after removing
        stopwords), this document is noisy and not included
    Return:
        dictionary whose keys are topics, values are list of list of words.  
    """
    lemmatizer = WordNetLemmatizer()
    stopwords_set = set(stopwords.words("english"))
    data_clean = {}
    for topic, docs in data_raw.items():
        newdocs = []
        for doc in docs:
            # Use regular expression for tokenizing. Match all patterns as "word","word'word", "word'"
            string = re.sub("(?:[\.\w]+@[\.\w]+)", " ", doc.lower()) # Remove emails
            words = re.findall("[a-zA-Z]+\'[a-zA-Z]+|(?<!\')[a-zA-Z]+\'|[a-zA-Z]+", string)
            newwords = []
            for w in words:
                if w not in stopwords_set:
                    w = lemmatizer.lemmatize(w)
                    if len(w) > 1:
                        newwords.append(w)
            
            word_count = collections.Counter()
            word_count.update(newwords)
            if word_count.most_common(1)[0][1] < filter_size:
                # Filter the doc whose most common term's occurence is more than filter_size
                newdocs.append(newwords)
        
        data_clean[topic] = newdocs
    
    return data_clean

def get_feature(data, feat_select=None, feat_idf=None, min_df=10, 
                max_feature=1000, norm_by_doclen=False):
    """
    Select features(words) from each document to form tf-idf vector. 
    Inputs:
        data: cleaned data.
        feat_select: words from the whole corpus. On training set, it is calculated. On testing dataset, it is
        adopted from training set.
        feat_idf: log(N/doc freqency) for each selected feature. On training set, it is calculated. 
        On testing dataset, it is adopted from training set.
        min_df: If any feature's document frequency is below the threshold, it is ignored to avoid too rare feature.
        max_feature: maximum number of feature to calculate tf-idf
        norm_by_doclen: determine whether to normalize term frequency by the document length that contains the term.
    Return:
        X: numpy array. tf-idf vectors. Each row represents a document.
        Y: numpy array. Labels.
        feat_select: effective on training dataset.
        feat_idf: effective on training dataset
    """
    feature_count = collections.Counter()
    feature_count_intopic = dict()
    tf_count = list()
    doc_length = list()
    if feat_select is None: df_count = dict()
    n_doc = 0
    label = list()
    
    for idx, item in enumerate(data.items()):
        topic, docs = item
        intopic = collections.Counter()
        for feats in docs:
            n_doc += 1
            label.append(idx)
            tf = collections.Counter()
            feature_count.update(feats)
            intopic.update(feats)
            tf.update(feats)
            if feat_select is None:
                for feat in tf: # get unique features in a doc
                    if feat not in df_count:
                        df_count[feat] = 1
                    else:
                        df_count[feat] += 1
                    
            tf_count.append(tf)
            doc_length.append(len(feats))
        
        feature_count_intopic[topic] = intopic
    
    n_feat = len(feature_count)
    print("Total number of features: ", n_feat)
    print("Total number of documents: ", n_doc)
    
    if feat_select is None:
        
        feat_select = set() # use set() to select unique features
        # Select one feature from one topic every time until reaching maximum feature size.
        # "intopic": features in each topic sorted by frequency in descending order
        # "idx": list of pointers pointing to the feature in each topic
        intopic = [[y[0] for y in x.most_common()] for x in feature_count_intopic.values()] 
        idx = [0 for _ in range(len(intopic))]
        topic_id = 0
        total = 0
        while len(feat_select) < max_feature and total<len(feature_count):
            topic_id = topic_id % len(intopic)
            if idx[topic_id] < len(intopic[topic_id]):
                feat = intopic[topic_id][idx[topic_id]]
                if df_count[feat] >= min_df:
                    feat_select.add(feat)
                idx[topic_id] += 1
                total += 1
                
            topic_id += 1
        
        df_arr = np.array([df_count[feat] for feat in feat_select])
        feat_idf = np.log(n_doc / df_arr) # feature idf value
    
    # Generate model input and output
    X = []
    row, col = list(), list()
    for i, tf in enumerate(tf_count):
        temp = []
        tempidf = []
        for j, feat in enumerate(feat_select):
            if tf[feat] > 0:
                temp.append(tf[feat])
                tempidf.append(feat_idf[j])
                col.append(j)
                row.append(i)
                
        temp = np.array(temp) * np.array(tempidf)
        if norm_by_doclen: # normalize term frequency by its document length
            temp /= doc_length[i]
        norm = np.linalg.norm(temp)
        if norm != 0:
            temp /= norm
        X.append(temp)
    
    X = np.concatenate(X)
    row = np.array(row)
    col = np.array(col)
    X = coo_matrix((X, (row, col)), shape=(len(tf_count), len(feat_select)))
    Y = np.array(label)
    
    return X, Y, feat_select, feat_idf