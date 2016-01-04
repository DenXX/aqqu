
import cPickle as pickle
import globals
import scorer_globals
from entity_linker.entity_linker import KBEntity
from query_translator.evaluation import load_eval_queries

import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s '
                           ': %(module)s : %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
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
