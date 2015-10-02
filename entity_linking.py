from corenlp_parser.parser import CoreNLPParser
from text2kb.websearch_features import WebSearchFeatureGenerator

__author__ = 'dsavenk'

import globals
import logging
import sys
from entity_linker.entity_linker import EntityLinker, KBEntity

logging.basicConfig(format='%(asctime)s : %(levelname)s '
                           ': %(module)s : %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    globals.read_configuration('config.cfg')
    entity_linker = EntityLinker.init_from_config()
    parser = CoreNLPParser.init_from_config()

    feature_generator = WebSearchFeatureGenerator.init_from_config()
    doc = feature_generator.question_serps[u'what character did natalie portman play in star wars?'][10]
    content = doc.content()

    # while True:
    # print "Please enter some text: "
    # text = sys.stdin.readline().strip()
    tokens = parser.parse(content).tokens
    res = entity_linker.identify_entities_in_document(tokens)
    for entity in res:
        print entity


if __name__ == "__main__":
    main()
