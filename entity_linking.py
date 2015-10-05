from corenlp_parser.parser import CoreNLPParser
from text2kb.websearch_features import WebSearchFeatureGenerator

__author__ = 'dsavenk'

import cPickle as pickle
import functools
import globals
import logging
from datetime import datetime
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
        tokens = parser.parse(content).tokens[:1000]
        return entity_linker.identify_entities_in_document(tokens, min_surface_score=0.5)
    return []


def main_parse():
    globals.read_configuration('config.cfg')
    parser = CoreNLPParser.init_from_config()
    feature_generator = WebSearchFeatureGenerator.init_from_config()
    print datetime.now()
    with open(sys.argv[1], 'w') as out_file:
        index = 0
        for serp in feature_generator.question_serps.itervalues():
            for doc in serp[:10]:
                content = doc.content()
                if len(content) > 0:
                    document = (doc.url, parser.parse(content))
                    pickle.dump(document, out_file)
            print "Query #", index, datetime.now()
            index += 1

def test_load():
    data = []
    with open(sys.argv[1], 'r') as input:
        data.append(pickle.load(input))


def main():
    globals.read_configuration('config.cfg')
    entity_linker = EntityLinker.init_from_config()
    parser = CoreNLPParser.init_from_config()
    feature_generator = WebSearchFeatureGenerator.init_from_config()
    doc_entities = dict()
    for serp in feature_generator.question_serps.itervalues():
        for doc in serp[:10]:
            doc_entities[doc.url] = find_document_entities(doc, parser, entity_linker)
        
    with open(sys.argv[1], 'w') as out:
        pickle.dump(doc_entities, out)


if __name__ == "__main__":
    main_parse()