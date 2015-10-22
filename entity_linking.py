from corenlp_parser.parser import CoreNLPParser
from text2kb.web_features import WebFeatureGenerator

__author__ = 'dsavenk'

import cPickle as pickle
import globals
import logging
from datetime import datetime
import sys

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
    from text2kb.web_features import get_questions_serps
    from text2kb.web_features import get_documents_entities
    serps = get_questions_serps()
    doc_entities = get_documents_entities()
    import operator
    while True:
        print "Please enter a question:"
        question = sys.stdin.readline().strip()
        if question in serps:
            docs = serps[question][:10]
            entities = {}
            for doc in docs:
                for entity in doc_entities[doc.url].itervalues():
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

    doc_entities = dict()
    from text2kb.web_features import get_questions_serps, get_documents_content_dict
    question_search_results = get_questions_serps()
    documents_content = get_documents_content_dict(return_parsed_tokens=True)
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
    from text2kb.web_features import get_questions_serps
    question_search_results = get_questions_serps()
    globals.logger.setLevel("DEBUG")
    import operator
    while True:
        print "Please enter some text: "
        text = sys.stdin.readline().strip().decode('utf-8')
        tokens = parser.parse(text).tokens
        print "Entities:", entity_linker.identify_entities_in_document(tokens, max_token_window=5)
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
        # print sorted(tokens.items(), key=operator.itemgetter(1), reverse=True)[:50]


def entity_link_snippets():
    globals.read_configuration('config.cfg')
    entity_linker = globals.get_entity_linker()
    from text2kb.web_features import get_questions_serps
    question_search_results = get_questions_serps()
    doc_snippet_entities = dict()
    for index, serp in enumerate(question_search_results.itervalues()):
        for doc in serp[:globals.SEARCH_RESULTS_TOPN]:
            snippet_tokens = doc.get_snippet_tokens()
            entities = entity_linker.identify_entities_in_document(snippet_tokens)
            for entity in entities:
                entity['matches'] = []
                for position in entity['positions']:
                    entity['matches'].append(snippet_tokens[position[0]:position[1]])
            doc_snippet_entities[doc.url] = entities
        if index % 100 == 0:
            logger.info("Processed %d serps" % index)
    logger.info("Pickling the dictionary...")
    with open(sys.argv[1], 'w') as out:
        pickle.dump(doc_snippet_entities, out)
    logger.info("Pickling the dictionary DONE!")


def test_new_entity_linker():
    globals.read_configuration('config.cfg')
    from query_translator.translator import QueryTranslator
    query_translator = QueryTranslator.init_from_config()
    while True:
        question = sys.stdin.readline().strip()
        print "Translation: ", query_translator.translate_query(question)


if __name__ == "__main__":
    # main_entities()  # For entity linking from SERP for a question
    main_entity_link_text()  # For entity linking from arbitrary text
    # entity_link_snippets()
    # test_new_entity_linker()