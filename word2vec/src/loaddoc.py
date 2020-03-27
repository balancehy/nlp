
import collections
import numpy as np
import pickle
import re


def load_books(filenames, keep_punc=True):
    """
    Read files from a list of file names. Extract words or punctuations depending on the setting.
    keep_punc: if True, consider punctuations as words, otherwise eliminate them
    return:
        1. word_counter: unique words and their counts in the documents 
        2. word_list: a list of words in the documents
        3. word_sent_list: a nested list. Each element is a sentence seperated by sentence splitters. Each sentence contains a list of words.
    """
    word_counter = collections.Counter()
    word_list = []
    words_sents_list = []
    doc = ""
    num_lines, num_words = (0, 0)
    for filename in filenames:
        with open(filename, 'r') as f:
            for line in f.readlines():
                doc += (line.lower().strip('\n') + ' ')
                num_lines += 1
    # Parse          
    if keep_punc:
        word_list = re.findall("[\\w']+|[;:\-\(\)&.,!?\"]", doc)
        num_words = len(word_list)
        word_counter.update(word_list)
    else:
        sentences = re.split("[?.;!]+", doc)
        for sent in sentences:
            if len(sent) != 0:
                words = re.findall("\w+\'\w+|(?<!\')\w+\'|\w+|,", sent)
                words_sents_list.append(words)
                word_list.extend(words)
                num_words += len(words)
                word_counter.update(words)

    return word_counter, word_list, words_sents_list, num_lines, num_words


def build_dict(word_counter, vocab_size=50000):
    """
    Builds dictionary from word_counter. Sort the keys from the most to least common.
    """
    top_words = word_counter.most_common(vocab_size)
    top_words.sort(key=lambda t: -t[1])
    dictionary = dict()
    for idx, word in enumerate(map(lambda t: t[0], top_words)):
        dictionary[word] = idx

    return dictionary


def doc2num(word_list, words_sents_list, dictionary):
    """
    Maps list of words to np.array of integers using key/value pairs in
    dictionary. Words not found in dictionary will be mapped to len(dictionary). Also if words_sents_list is not empty, return its integer representation.
    """
    word_array = []
    unknown_val = len(dictionary)
    for word in word_list:
        word_array.append(dictionary.get(word, unknown_val))

    words_sents_array = []
    for sent in words_sents_list:
        arr = []
        for word in sent:
            arr.append(dictionary.get(word, unknown_val))
        words_sents_array.append(np.array(arr))
        
    return np.array(word_array, dtype=np.int32), words_sents_array


def build_word_array(filenames, vocab_size, keep_punc=True, sent_token=False):
    """
    Convenience function that runs: 1) load_books(), 2) build_dict(),
        and doc2num() in sequence and returns integer word array of documents
    filenames: list of file names (including path, if needed)
    vocab_size: Upper limit on vocabulary size. If number of unique words
        greater than vocab_size, will take most commonly occurring words
    keep_punc: if True, consider punctuations as words, return array of words represented by integer
    sent_token: if True, first split into sentences, then parse words. Otherwise, directly parse words. Must be set together with keep_punc=False to take effect.
    """
    word_counter, word_list, words_sents_list, num_lines, num_words = load_books(filenames, keep_punc=keep_punc)
    dictionary = build_dict(word_counter, vocab_size)
    reverse_dict = {v: k for k, v in dictionary.items()}
    word_array, words_sents_array = doc2num(word_list, words_sents_list, dictionary)

    if keep_punc or not sent_token:
        return word_array, dictionary, reverse_dict, num_lines, num_words
    else:
        return words_sents_array, dictionary, reverse_dict, num_lines, num_words


def save_word_array(filename, word_array, dictionary):

    word_array_dict = dict()
    word_array_dict['word_array'] = word_array
    word_array_dict['dictionary'] = dictionary
    with open(filename + '.p', 'wb') as f:
        pickle.dump(word_array_dict, f)


def load_word_array(filename):

    with open(filename + '.p', 'rb') as f:
        word_array_dict = pickle.load(f)

    return word_array_dict['word_array'], word_array_dict['dictionary']
