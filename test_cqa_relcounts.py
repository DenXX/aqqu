import gzip

import operator

import globals
import scorer_globals
from query_translator import translator
from query_translator.features import get_query_text_tokens
from query_translator.learner import get_evaluated_queries

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Dump qa entity pairs.")
    parser.add_argument("--config",
                        default="config.cfg",
                        help="The configuration file to use.")
    parser.add_argument("--output",
                        help="The file to dump results to.")
    args = parser.parse_args()
    globals.read_configuration(args.config)
    scorer_globals.init()

    cqa_wordrel_counts_file = globals.config.get('WebSearchFeatures', 'cqa-wordrel-counts-file')
    word_rel_scores = dict()

    print "Reading cqa token relation counts..."
    with gzip.open(cqa_wordrel_counts_file, 'r') as inp:
        for line_index, line in enumerate(inp):
            fields = line.strip().split("\t")
            word = fields[0]
            relations = fields[1].split("|")
            pmi = float(fields[-1])
            if word not in word_rel_scores:
                word_rel_scores[word] = dict()
            for relation in relations:
                word_rel_scores[word][relation] = pmi

    parameters = translator.TranslatorParameters()
    parameters.require_relation_match = False
    parameters.restrict_answer_type = False

    datasets = ["webquestionstrain", "webquestionstest",]
    # "webquestionstrain_externalentities", "webquestionstest_externalentities"]

    count = 0
    correct_relations = set()
    for dataset in datasets:
        queries = get_evaluated_queries(dataset, True, parameters)
        for index, query in enumerate(queries):
            print "-----"
            relation_scores = []
            print query.utterance
            for candidate in query.eval_candidates:
                question_tokens = [t.token.lower() for t in candidate.query_candidate.query.query_tokens]
                for relation in candidate.query_candidate.relations:
                    relation_name = relation.name
                    scores = []
                    for token in question_tokens:
                        scores.append(word_rel_scores.get(token, dict()).get(relation_name, 0.0))
                    relation_scores.append((relation_name, sum(scores)))
            print sorted(relation_scores, key=operator.itemgetter(1), reverse=True)[:20]
