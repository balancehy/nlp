#!/usr/bin/env python
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, hstack, vstack
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import collections
import re


def load_file(fpath):
    with open(fpath, 'r') as f:
        docs = "".join(f.readlines()) 

    return docs

def data_clean(doc):
    lemmatizer = WordNetLemmatizer()
    stopwords_set = set(stopwords.words("english"))
    string = re.sub("(?:[\.\w]+@[\.\w]+)", " ", doc.lower()) # Remove emails
    words = re.findall("[a-zA-Z]+\'[a-zA-Z]+|(?<!\')[a-zA-Z]+\'|[a-zA-Z]+", string)
    newwords = []
    for w in words:
        if w not in stopwords_set:
            w = lemmatizer.lemmatize(w)
            if len(w) > 1:
                newwords.append(w)
    
    return newwords

def get_model_inputs(data, feat_select, feat_idf, norm_by_doclen=False):

    tf_count = list()
    doc_length = list()
    
    for feats in data:
        tf = collections.Counter()
        tf.update(feats)
        tf_count.append(tf)
        doc_length.append(len(feats))

    X = list()
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
    
    return X