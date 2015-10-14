from corenlp_parser.parser import CoreNLPParser
from text2kb.web_features import WebFeatureGenerator

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


def main_parse():
    globals.read_configuration('config.cfg')
    parser = CoreNLPParser.init_from_config()
    feature_generator = WebFeatureGenerator.init_from_config()
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


def main_entities():
    globals.read_configuration('config.cfg')
    feature_generator = WebFeatureGenerator.init_from_config()
    import operator
    while True:
        print "Please enter a question:"
        question = sys.stdin.readline().strip()
        if question in feature_generator.question_serps:
            docs = feature_generator.question_serps[question][:10]
            entities = {}
            for doc in docs:
                for entity in feature_generator._document_entities[doc.url]:
                    e = (entity['mid'], entity['name'])
                    if e not in entities:
                        entities[e] = 0
                    entities[e] += entity['count']
            top_entities = entities.items()
            top_entities.sort(key=operator.itemgetter(1), reverse=True)
            print top_entities[:50]


def main():
    globals.read_configuration('config.cfg')
    entity_linker = globals.get_entity_linker()
    config_options = globals.config
    serp_files = config_options.get('WebSearchFeatures', 'serp-files').split(',')
    documents_files = config_options.get('WebSearchFeatures', 'documents-files').split(',')
    content_file = config_options.get('WebSearchFeatures', 'documents-content-file')

    doc_entities = dict()
    from text2kb.web_features import _read_serp_files, _read_document_content
    question_search_results = _read_serp_files(serp_files, documents_files)
    documents_content = _read_document_content(content_file, return_parsed_tokens=True)
    index = 0
    for serp in question_search_results.itervalues():
        for doc in serp[:10]:
            if doc.url in documents_content:
                doc_entities[doc.url] = entity_linker.identify_entities_in_document(documents_content[doc.url],
                                                                                    min_surface_score=0.5)
        index += 1
        if index % 100 == 0:
            logger.info("%d SERPs processed" % index)
    with open(sys.argv[1], 'w') as out:
        pickle.dump(doc_entities, out)


def main_entity_link_text():
    globals.read_configuration('config.cfg')
    entity_linker = globals.get_entity_linker()
    parser = globals.get_parser()
    from text2kb.web_features import _read_serp_files
    serp_files = globals.config.get('WebSearchFeatures', 'serp-files').split(',')
    documents_files = globals.config.get('WebSearchFeatures', 'documents-files').split(',')
    question_search_results = _read_serp_files(serp_files, documents_files)
    import operator
    while True:
        print "Please enter some text: "
        text = sys.stdin.readline().strip().decode('utf-8')
        tokens = parser.parse(text).tokens
        print entity_linker.identify_entities_in_document(tokens, max_token_window=5)
        entities = {}
        tokens = {}
        if text in question_search_results:
            for doc in question_search_results[text][:10]:
                print doc
                title = doc.title
                snippet = doc.snippet
                snippet_tokens = parser.parse(title + "\n" + snippet).tokens
                for token in snippet_tokens:
                    if token.lemma not in tokens:
                        tokens[token.lemma] = 0
                    tokens[token.lemma] += 1
                for entity in entity_linker.identify_entities_in_document(snippet_tokens):
                    if entity['mid'] not in entities:
                        entities[entity['mid']] = entity
                    else:
                        entities[entity['mid']]['count'] += entity['count']
        print sorted(entities.values(), key=operator.itemgetter('count'), reverse=True)[:50]
        print sorted(tokens.items(), key=operator.itemgetter(1), reverse=True)[:50]

if __name__ == "__main__":
    main()