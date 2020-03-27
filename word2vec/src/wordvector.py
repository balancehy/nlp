#!usr/bin/python3

from sklearn.manifold import TSNE
import numpy as np
from scipy.spatial.distance import cdist
import pickle


class WordVector(object):
    def __init__(self, embed_matrix, dictionary):

        self._embed_matrix = embed_matrix
        self._dictionary = dictionary
        self._reverse_dictionary = {v: k for k, v in dictionary.items()}

    def n_closest(self, word, num_closest=5, metric='cosine'):
        """
        Given a word (in string format), return list of closest words based
        on given distance measures.
        word: input word
        num_closest: number of closest words to return
        metric: passed to scipy.spatial.distance.cdist
            'euclidean'
            'cosine'
        """
        word_vector = self.get_vector_by_name(word)
        closest_indices = self.closest_row_indices(word_vector, num_closest + 1, metric)
        word_list = []
        for i in closest_indices:
            w = self._reverse_dictionary[i]
            if w != word:
                word_list.append(w)

        return word_list

    def words_in_range(self, start, end):
        """
        Returns list of words with dictionary values starting at start and
        ending at (end-1).
        start: Start of slice (inclusive)
        end: End of slice (exclusive)
        """
        word_list = []
        for i in range(start, end):
            word_list.append(self._reverse_dictionary[i])

        return word_list

    def most_common(self, num=5):
        """
        Returns n most common words.
        num: Number of words to return
        """
        return self.words_in_range(0, num)

    def analogy(self, a, b, c, num=1, metric='cosine'):
        """
        Predict d in the analogy: a is to b; as c is to d.
        a: Word string in dictionary
        b: Word string in dictionary
        c: Word string in dictionary
        num: Number of candidates to return for d.
        metric: distance measure to us in predicting word d:
            (see n_closest() for options)
        """
        va = self.get_vector_by_name(a)
        vb = self.get_vector_by_name(b)
        vc = self.get_vector_by_name(c)
        vd = vb - va + vc
        closest_indices = self.closest_row_indices(vd, num, metric)
        d_list = []
        for i in closest_indices:
            d_list.append(self._reverse_dictionary[i])

        return d_list

    def project_2d(self, start, end):
        """
        Projects word embeddings into 2 dimensions for visualization (using
        t-SNE algorithm as implemented in sklearn). Projects words in range
        [start, end).
        The sklearn TNSE algorithm is memory intensive. Projecting more than
        ~500 words at a time can cause memory overflow.
        start: Start of slice (inclusive)
        end: End of slice (exclusive)
        :return: tuple:
            2D numpy array of shape = (start-end, 2)
            List of words, length = (start-end), corresponds to row ordering
                of returned numpy array
        """
        tsne = TSNE(n_components=2)
        embed_2d = tsne.fit_transform(self._embed_matrix[start:end, :])
        word_list = []
        for i in range(start, end):
            word_list.append(self._reverse_dictionary[i])

        return embed_2d, word_list

    def get_vector_by_name(self, word):
        """
        Return 1D numpy word vector by word. Word must exist in dictionary
        provided during object construction.
        word: String
        """
        return np.ravel(self._embed_matrix[self._dictionary[word], :])

    def get_vector_by_num(self, num):
        """
        Return 1D numpy word vector by num (word integer value). Word integer
        value must correspond to row number in embed_matrix providing during
        object construction.
        num: int
        """
        return np.ravel(self._embed_matrix[num, :])

    def get_reverse_dict(self):
        """
        Return copy of reverse word dictionary
        """
        return self._reverse_dictionary.copy()

    def closest_row_indices(self, word_vector, num, metric):
        """
        Return num closest row indices in sorted order (from closest
        to furthest.
        word_vector: word vector to measure distance from
        num: number of indices to return
        metric: (as passed to scipy.spatial.distance.cdist):
            'euclidean'
            'cosine'
            (see scipy.spatial.distance for more distance options)
        """
        dist_array = np.ravel(cdist(self._embed_matrix, word_vector.reshape((1, -1)),
                                    metric=metric))
        sorted_indices = np.argsort(dist_array)

        return sorted_indices[:num]

    def save(self, filename):
        """
        Save learned WordVector to file for future fast load.
        filename: Filename (with path, if needed) for save.
        :return: None
        Note: no unit test coverage
        """
        embedding = {'embed_matrix': self._embed_matrix,
                     'dictionary': self._dictionary}
        with open(filename + '.p', 'wb') as f:
            pickle.dump(embedding, f)

    @staticmethod
    def load(filename):
        """
        Load WordVector from file and return WordVector object
        filename: Filename as used in WordVector.save() method
        :return: WordVector object
        Note: no unit test coverage
        """
        with open(filename + '.p', 'rb') as f:
            embedding = pickle.load(f)

        return WordVector(embedding['embed_matrix'],
                          embedding['dictionary'])

