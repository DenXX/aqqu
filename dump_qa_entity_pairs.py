import gzip
from sys import stderr

import globals
import scorer_globals
from entity_linker.entity_linker import KBEntity
from query_translator import translator
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

    parameters = translator.TranslatorParameters()
    parameters.require_relation_match = False
    parameters.restrict_answer_type = False

    datasets = ["webquestionstrain", "webquestionstest",]
                # "webquestionstrain_externalentities", "webquestionstest_externalentities"]

    count = 0
    for dataset in datasets:
        queries = get_evaluated_queries(dataset, True, parameters)
        for index, query in enumerate(queries):
            # Correct answer
            # entity_names.update(query.target_result)

            # if query.oracle_position == -1:
            #     entities = set()
            #     for candidate in query.eval_candidates:
            #         for entity in candidate.query_candidate.matched_entities:
            #             if isinstance(entity.entity.entity, KBEntity):
            #                 entities.add((entity.entity.name, entity.entity.entity.id))
            #     print query.utterance
            #     print entities

            for candidate in query.eval_candidates:
                answer_entities = set(mid for entity_name in candidate.prediction
                                      for mid in KBEntity.get_entityid_by_name(entity_name, keep_most_triples=True))
                question_entities = set(mid for entity in candidate.query_candidate.matched_entities
                                        for mid in KBEntity.get_entityid_by_name(entity.entity.name,
                                                                                 keep_most_triples=True))
                for question_entity in question_entities:
                    for answer_entity in answer_entities:
                        print question_entity + "\t" + answer_entity

            if index % 100 == 0:
                print >> stderr, "Processed %d queries" % index
