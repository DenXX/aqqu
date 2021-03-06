"""
Copyright 2015, University of Freiburg.

Elmar Haussmann <haussmann@cs.uni-freiburg.de>
"""
from gensim import models
from gensim import matutils
from numpy import dot
import logging
import data

logger = logging.getLogger(__name__)
MIN_WORD_SIMILARITY = 0.4
EMBEDDING_DIMENSION = 300


class WordEmbeddings(object):

    def __init__(self, model_fname):
        self.embeddings = models.Word2Vec.load(model_fname)
        # TODO(denxx): For some reason in pickled model the vector_size isn't specified.
        self.embeddings.vector_size = EMBEDDING_DIMENSION

    def synonym_score(self, word_a, word_b):
        """
        Returns a synonym score for the provided words.
        If the two words are not considered a synonym
        0.0 is returned.
        :param word_a:
        :param word_b:
        :return:
        """
        similarity = self.similarity(word_a, word_b)
        if similarity > MIN_WORD_SIMILARITY:
            return similarity
        else:
            return 0.0

    def n_similarity(self, words_a, words_b):
        words_a = set([word for word in words_a if word in self.embeddings]) 
        words_b = set([word for word in words_b if word in self.embeddings])
        if not words_a or not words_b:
            return 0.0
        return self.embeddings.n_similarity(words_a, words_b)

    def similarity(self, word_a, word_b):
        try:
            a_vector = self.embeddings[word_a]
            b_vector = self.embeddings[word_b]
            diff = dot(matutils.unitvec(a_vector), matutils.unitvec(b_vector))
            return diff
        except KeyError:
            #logger.debug("'%s' or '%s' don't have a word vector" % (word_a.encode("utf-8"),
            #                                                        word_b.encode("utf-8")))
            return 0.0 if word_a != word_b else 1.0

    def __getitem__(self, word):
        if word in self.embeddings:
            return self.embeddings[word]
        else:
            return self.embeddings.seeded_vector(word)


class WordDerivations(object):

    def __init__(self, derivations_file):
        self.word_derivations = data.read_word_derivations(derivations_file)

    def get_word_suffix(self, token):
        """Return the suffixed word depending on POS-tag.

        :param token:
        :return:
        """
        word = token.lemma
        if token.pos == "JJ" or token.pos == "JJS" or token.pos == "RB":
            word += ".a"
        elif token.pos.startswith('V'):
            word += ".v"
        elif token.pos.startswith('N'):
            word += ".n"
        return word

    def has_derivations(self, token_a):
        """Do derivations for the token exist.

        Useful for efficiency.

        :param token_a:
        :return:
        """
        word = self.get_word_suffix(token_a)
        if word in self.word_derivations:
            return True
        return False

    def is_derivation(self, token_a, word_b):
        """Returns True if word_b can be derived from word_a.

        :param word_a:
        :param word_b:
        :return:
        """
        word = self.get_word_suffix(token_a)
        if word in self.word_derivations:
            if word_b in self.word_derivations[word]:
                return True
        return False
