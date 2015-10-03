from corenlp_parser.parser import CoreNLPParser
from text2kb.websearch_features import WebSearchFeatureGenerator

__author__ = 'dsavenk'

import cPickle as pickle
import functools
import globals
import logging
import multiprocessing
import sys
from entity_linker.entity_linker import EntityLinker

logging.basicConfig(format='%(asctime)s : %(levelname)s '
                           ': %(module)s : %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def find_document_entities(document, parser, entity_linker):
    content = document.content()
    if len(content.strip()) > 0:
        tokens = parser.parse(content).tokens
        return entity_linker.identify_entities_in_document(tokens, min_surface_score=0.1)
    return []


def main():
    globals.read_configuration('config.cfg')
    entity_linker = EntityLinker.init_from_config()
    parser = CoreNLPParser.init_from_config()
    feature_generator = WebSearchFeatureGenerator.init_from_config()
    pool = multiprocessing.Pool(1)
    processing = functools.partial(find_document_entities, parser=parser, entity_linker=entity_linker)
    document_entities = pool.map(processing, [doc for serp in feature_generator.question_serps.itervalues()
                                              for doc in serp][:10])
    with open(sys.argv[1], 'w') as out:
        pickle.dump(document_entities, out)


if __name__ == "__main__":
    main()
