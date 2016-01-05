
import cPickle as pickle

import re
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.linear_model import SGDClassifier

from sklearn.feature_extraction import DictVectorizer

from sklearn.preprocessing import LabelEncoder, StandardScaler

import globals
import scorer_globals
from entity_linker.entity_linker import KBEntity
from query_translator import translator
from query_translator.evaluation import load_eval_queries

import logging

from query_translator.features import FeatureExtractor, get_grams_feats, get_query_text_tokens, get_n_grams_features
from query_translator.learner import get_evaluated_queries

logging.basicConfig(format='%(asctime)s : %(levelname)s '
                           ': %(module)s : %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_npmi_ngram_type_pairs():
    globals.read_configuration('config.cfg')
    scorer_globals.init()

    datasets = ["webquestions_split_train", ]

    parameters = translator.TranslatorParameters()
    parameters.require_relation_match = False
    parameters.restrict_answer_type = False

    n_gram_type_counts = dict()
    type_counts = dict()
    n_gram_counts = dict()
    total = 0
    year_pattern = re.compile("[0-9]{4}")
    for dataset in datasets:
        queries = get_evaluated_queries(dataset, True, parameters)
        for index, query in enumerate(queries):
            if query.oracle_position != -1 and query.oracle_position <= len(query.eval_candidates):
                correct_candidate = query.eval_candidates[query.oracle_position - 1]
                logger.info(query.utterance)
                logger.info(correct_candidate.query_candidate)
                n_grams = set(get_n_grams_features(correct_candidate.query_candidate))
                answer_entities = [mid for answer in query.target_result
                                   if year_pattern.match(answer) is None
                                   for mid in KBEntity.get_entityid_by_name(answer, keep_most_triples=True)]
                correct_notable_types = set(filter(lambda x: x,
                                                   [KBEntity.get_notable_types(entity_mid)
                                                    for entity_mid in answer_entities]))

                for notable_type in correct_notable_types:
                    if notable_type not in type_counts:
                        type_counts[notable_type] = 0
                    type_counts[notable_type] += 1

                for n_gram in n_grams:
                    if n_gram not in n_gram_counts:
                        n_gram_counts[n_gram] = 0
                    n_gram_counts[n_gram] += 1

                    for notable_type in correct_notable_types:
                        pair = (n_gram, notable_type)
                        if pair not in n_gram_type_counts:
                            n_gram_type_counts[pair] = 0
                        n_gram_type_counts[pair] += 1

                total += 1

    npmi = dict()
    from math import log
    for n_gram_type_pair, n_gram_type_count in n_gram_type_counts.iteritems():
        n_gram, type = n_gram_type_pair
        npmi[n_gram_type_pair] = (log(n_gram_type_count) - log(n_gram_counts[n_gram]) - log(type_counts[type]) +
                                    log(total)) / (-log(n_gram_type_count) + log(total))

    with open("type_model_npmi.pickle", 'wb') as out:
        pickle.dump(npmi, out)

    import operator
    npmi = sorted(npmi.items(), key=operator.itemgetter(1), reverse=True)
    print "\n".join(map(str, npmi[:50]))



def train_type_model():
    globals.read_configuration('config.cfg')
    parser = globals.get_parser()
    scorer_globals.init()

    datasets = ["webquestions_split_train", ]

    parameters = translator.TranslatorParameters()
    parameters.require_relation_match = False
    parameters.restrict_answer_type = False

    feature_extractor = FeatureExtractor(False, False, n_gram_types_features=True)
    features = []
    labels = []
    for dataset in datasets:
        queries = get_evaluated_queries(dataset, True, parameters)
        for index, query in enumerate(queries):
            tokens = [token.lemma for token in parser.parse(query.utterance).tokens]
            n_grams = get_grams_feats(tokens)

            answer_entities = [mid for answer in query.target_result
                               for mid in KBEntity.get_entityid_by_name(answer, keep_most_triples=True)]
            correct_notable_types = set(filter(lambda x: x,
                                               [KBEntity.get_notable_types(entity_mid)
                                                for entity_mid in answer_entities]))

            other_notable_types = set()
            for candidate in query.eval_candidates:
                entities = [mid for entity_name in candidate.prediction
                            for mid in KBEntity.get_entityid_by_name(entity_name, keep_most_triples=True)]
                other_notable_types.update(set([KBEntity.get_notable_types(entity_mid) for entity_mid in entities]))
            incorrect_notable_types = other_notable_types.difference(correct_notable_types)

            for type in correct_notable_types.union(incorrect_notable_types):
                if type in correct_notable_types:
                    labels.append(1)
                else:
                    labels.append(0)
                features.append(feature_extractor.extract_ngram_features(n_grams, [type, ], "type"))

    with open("type_model_data.pickle", 'wb') as out:
        pickle.dump((features, labels), out)

    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)
    vec = DictVectorizer(sparse=True)
    X = vec.fit_transform(features)
    feature_selector = SelectPercentile(chi2, percentile=5).fit(X, labels)
    vec.restrict(feature_selector.get_support())
    X = feature_selector.transform(X)
    type_scorer = SGDClassifier(loss='log', class_weight='auto',
                                n_iter=1000,
                                alpha=1.0,
                                random_state=999,
                                verbose=5)
    type_scorer.fit(X, labels)
    with open("type-model.pickle", 'wb') as out:
        pickle.dump((vec, type_scorer), out)


if __name__ == "__main__":
    extract_npmi_ngram_type_pairs()
    exit()

    globals.read_configuration('config.cfg')
    parser = globals.get_parser()
    scorer_globals.init()

    datasets = ["webquestions_split_train", ]
    # datasets = ["webquestions_split_train_externalentities", "webquestions_split_dev_externalentities",]
    # datasets = ["webquestions_split_train_externalentities3", "webquestions_split_dev_externalentities3",]

    data = []
    for dataset in datasets:
        queries = load_eval_queries(dataset)
        for index, query in enumerate(queries):
            tokens = [token.token for token in parser.parse(query.utterance).tokens]
            answer_entities = [mid for answer in query.target_result
                               for mid in KBEntity.get_entityid_by_name(answer, keep_most_triples=True)]
            notable_types = [KBEntity.get_notable_types(entity_mid) for entity_mid in answer_entities]
            data.append((tokens, notable_types))
            logger.info(tokens)
            logger.info([KBEntity.get_entity_name(notable_type) for notable_type in notable_types])

    with open("question_tokens_notable_types.pickle", 'wb') as out:
        pickle.dump(data, out)
