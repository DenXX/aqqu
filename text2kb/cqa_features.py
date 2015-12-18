import gzip
import logging

import globals

__author__ = 'dsavenk'

logger = logging.getLogger(__name__)

_cqa_tokenrelation_pmi = None
_DEFAULT_PMI_SCORE = 0.0


def get_cqa_token_relation_pmi_score(token, relation):
    global _cqa_tokenrelation_pmi
    if _cqa_tokenrelation_pmi is None:
        _cqa_tokenrelation_pmi = dict()
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
                for relation in relations:
                    _cqa_tokenrelation_pmi[word][relation] = pmi
        logger.info("Done reading CQA word-relation counts.")
    return _cqa_tokenrelation_pmi.get(token, dict()).get(relation, _DEFAULT_PMI_SCORE)


def generate_cqa_based_features(candidate):
    question_tokens = [t.token.lower() for t in candidate.query_candidate.query.query_tokens]
    pmi_scores = []
    for relation in candidate.relations:
        relation_name = relation.name
        for token in question_tokens:
            pmi_scores.append(get_cqa_token_relation_pmi_score(token, relation_name))
    return {
        "cqa_features:sum_pmi_scores": sum(pmi_scores),
        "cqa_features:avg_pmi_score": 1.0 * sum(pmi_scores) / len(pmi_scores),
        "cqa_features:avg_nonzero_pmi_score": 1.0 * sum(pmi_scores) / (len(filter(lambda score: score != 0.0,
                                                                                  pmi_scores)) + 1),
        "cqa_features:max_pmi_score": max(pmi_scores),
        "cqa_features:min_pmi_score": min(pmi_scores),
    }
