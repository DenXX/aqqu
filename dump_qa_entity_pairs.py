import gzip
from sys import stderr

import globals
import scorer_globals
from entity_linker.entity_linker import KBEntity
from query_translator import translator
from query_translator.learner import get_evaluated_queries

def print_sparql_queries():
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

    parameters = translator.TranslatorParameters()
    parameters.require_relation_match = False
    parameters.restrict_answer_type = False

    dataset = "webquestions_test_filter"

    sparql_backend = globals.get_sparql_backend(globals.config)
    queries = get_evaluated_queries(dataset, True, parameters)
    for index, query in enumerate(queries):
        print "--------------------------------------------"
        print query.utterance
        print "\n".join([str((entity.__class__, entity.entity)) for entity in query.eval_candidates[0].query_candidate.query.identified_entities])
        for eval_candidate in query.eval_candidates:
            query_candidate = eval_candidate.query_candidate
            query_candidate.sparql_backend = sparql_backend
            notable_types = query_candidate.get_answers_notable_types()
            if notable_types:
                print notable_types
                print query_candidate.graph_as_simple_string().encode("utf-8")
                print query_candidate.to_sparql_query().encode("utf-8")
                print "\n\n"



if __name__ == "__main__":
    # print_sparql_queries()
    # exit()

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

    parameters = translator.TranslatorParameters()
    parameters.require_relation_match = False
    parameters.restrict_answer_type = False

    # datasets = ["webquestions_split_train", "webquestions_split_dev",]
    # datasets = ["webquestions_split_train_externalentities", "webquestions_split_dev_externalentities",]
    datasets = ["webquestions_split_train_externalentities3", "webquestions_split_dev_externalentities3",]

    count = 0
    correct_relations = set()
    for dataset in datasets:
        queries = get_evaluated_queries(dataset, True, parameters)
        for index, query in enumerate(queries):
            # Correct answer
            # entity_names.update(query.target_result)

            # if query.oracle_position != -1:
            #     if dataset == datasets[0]:
            #         correct_relations.update([r.name for r in query.eval_candidates[query.oracle_position - 1].query_candidate.relations])
            #     else:
            #         for relation in query.eval_candidates[query.oracle_position - 1].query_candidate.relations:
            #             if relation.name not in correct_relations:
            #                 print query.utterance
            #                 print relation.name
            #                 print query.eval_candidates[query.oracle_position - 1].query_candidate
            #                 print "-----"

            # This loop will print questions without good candidate
            if query.oracle_position == -1:
                entities = set()
                for candidate in query.eval_candidates:
                    for entity in candidate.query_candidate.matched_entities:
                        if isinstance(entity.entity.entity, KBEntity):
                            entities.add((entity.entity.name, entity.entity.entity.id))
                print ">>>", query.utterance
                print entities

            # for candidate in query.eval_candidates:
            #     answer_entities = set(mid for entity_name in candidate.prediction
            #                           for mid in KBEntity.get_entityid_by_name(entity_name, keep_most_triples=True))
            #     question_entities = set(mid for entity in candidate.query_candidate.matched_entities
            #                             for mid in KBEntity.get_entityid_by_name(entity.entity.name,
            #                                                                      keep_most_triples=True))
            #     for question_entity in question_entities:
            #         for answer_entity in answer_entities:
            #             print question_entity + "\t" + answer_entity

            if index % 100 == 0:
                print >> stderr, "Processed %d queries" % index
