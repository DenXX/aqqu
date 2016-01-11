
import gzip
import logging
import numpy as np
from gensim import matutils

import globals
from text2kb.utils import get_embeddings

__author__ = 'dsavenk'

logger = logging.getLogger(__name__)

_cqa_tokenrelation_pmi = None
_cqa_relation_avg_embedding = None
_DEFAULT_PMI_SCORE = 0.0


def read_relation_token_pmi_file():
    global _cqa_tokenrelation_pmi
    global _cqa_relation_avg_embedding
    if _cqa_tokenrelation_pmi is None:
        _cqa_tokenrelation_pmi = dict()
        _cqa_relation_avg_embedding = dict()
        logger.info("Reading CQA word-relation counts...")
        cqa_wordrel_counts_file = globals.config.get("WebSearchFeatures", "cqa-wordrel-counts-file")
        with gzip.open(cqa_wordrel_counts_file, 'r') as input_file:
            for line in input_file:
                line = line.strip().split("\t")
                word = line[0]
                relations = line[1].split("|")
                pmi = float(line[-1])
                if word not in _cqa_tokenrelation_pmi:
                    _cqa_tokenrelation_pmi[word] = dict()
                for rel in relations:
                    if rel not in _cqa_relation_avg_embedding:
                        _cqa_relation_avg_embedding[rel] = dict()
                    _cqa_tokenrelation_pmi[word][rel] = pmi
                    _cqa_relation_avg_embedding[rel][word] = pmi

        # We will compute average vector of relation terms weighted by pmi
        embeddings = get_embeddings()
        for rel, worddict in _cqa_relation_avg_embedding.iteritems():
            sum_pmi = sum(pmi for word, pmi in worddict.iteritems() if word in embeddings.embeddings and pmi > 0)
            emb_vectors = np.array([embeddings[word] * pmi / sum_pmi for word, pmi in worddict.iteritems()
                                    if word in embeddings.embeddings and pmi > 0])
            avg_vector = matutils.unitvec(emb_vectors.sum(axis=0))
            _cqa_relation_avg_embedding[rel] = avg_vector
        logger.info("Done reading CQA word-relation counts.")


def get_cqa_token_relation_pmi_score(token, relation):
    read_relation_token_pmi_file()
    return _cqa_tokenrelation_pmi.get(token, dict()).get(relation, _DEFAULT_PMI_SCORE)


def get_cqa_relation_avg_embedding(relation):
    read_relation_token_pmi_file()
    return _cqa_relation_avg_embedding.get(relation, np.zeros(get_embeddings().embeddings.vector_size))


def generate_cqa_based_features(candidate):
    question_tokens = [t.token.lower() for t in candidate.query.query_tokens]
    exact_pmi_scores = []
    embedding_scores = []
    embeddings = get_embeddings()
    for relation in candidate.relations:
        relation_name = relation.name
        avg_relation_embedding = get_cqa_relation_avg_embedding(relation_name)
        for token in question_tokens:
            exact_pmi_scores.append(get_cqa_token_relation_pmi_score(token, relation_name))
            if token in embeddings.embeddings:
                embedding_scores.append(np.dot(avg_relation_embedding, matutils.unitvec(embeddings[token])))

    # If embeddings scores are empty, add 0.0.
    if not embedding_scores:
        embedding_scores.append(0.0)

    features = {
        "cqa_features:sum_pmi_scores": sum(exact_pmi_scores),
        "cqa_features:avg_pmi_score": 1.0 * sum(exact_pmi_scores) / len(exact_pmi_scores),
        "cqa_features:avg_nonzero_pmi_score": 1.0 * sum(exact_pmi_scores) / (
            len(filter(lambda sc: sc != 0.0, exact_pmi_scores)) + 1),
        "cqa_features:max_pmi_score": max(exact_pmi_scores),
        "cqa_features:min_pmi_score": min(exact_pmi_scores),
        "cqa_features:sum_emb_scores": sum(embedding_scores),
        "cqa_features:avg_emb_score": 1.0 * sum(embedding_scores) / len(embedding_scores),
        "cqa_features:avg_nonzero_emb_score": 1.0 * sum(embedding_scores) / (
            len(filter(lambda sc: sc != 0.0, embedding_scores)) + 1),
        "cqa_features:max_emb_score": max(embedding_scores),
        "cqa_features:min_emb_score": min(embedding_scores),
    }

    return features
