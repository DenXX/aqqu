from tqdm import tqdm

from corenlp_parser.parser import CoreNLPParser
from entity_linker.entity_linker import EntityLinker, WebSearchResultsExtenderEntityLinker
from query_translator import translator
from query_translator.evaluation import load_eval_queries
from text2kb.utils import tokenize, get_questions_serps, WebSearchResult

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


def find_entity_mentions(text, use_tagme=False):
    if use_tagme:
        import urllib, httplib, json
        params = urllib.urlencode({
            # Request parameters
            'text': text,
        })

        data = None
        try:
            host, port = globals.config.get("EntityLinker", "tagme-service-url").split(":")
            conn = httplib.HTTPConnection(host, port)
            conn.request("GET", "/get_entities?%s" % params)
            response = conn.getresponse()
            data = response.read()
            conn.close()
        except Exception as ex:
            logger.error(ex.message)
            return []
        if not data:
            return []
        return [{'mid': e['entity'],
                'name': e['entity'],
                'surface_score': float(e['coherence']),
                'score': float(e['rho']),
                'positions': (e['start'], e['end']),
                'count': 1} for e in json.loads(data)]
    else:
        entity_linker = globals.get_entity_linker()
        parser = globals.get_parser()
        tokens = parser.parse(text).tokens
        return entity_linker.identify_entities_in_document(tokens, max_token_window=5, get_main_name=True)


def main_entities():
    globals.read_configuration('config.cfg')
    from text2kb.utils import get_questions_serps
    from text2kb.utils import get_documents_entities
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


def main_doc_entities_from_content():
    entity_linker = globals.get_entity_linker()
    document_entities_file = globals.config.get('WebSearchFeatures', 'documents-entities-file')
    doc_entities = dict()
    from text2kb.utils import get_documents_content_dict
    from text2kb.utils import get_questions_serps
    question_search_results = get_questions_serps()
    documents_content = get_documents_content_dict(return_parsed_tokens=True)
    for serp in tqdm(question_search_results.values()):
        for doc in serp[:globals.SEARCH_RESULTS_TOPN]:
            if doc.url in documents_content:
                doc_entities[doc.url] = entity_linker.identify_entities_in_document(documents_content[doc.url],
                                                                                    min_surface_score=0.5)
    with open(document_entities_file, 'wx') as out:
        pickle.dump(doc_entities, out)


def main_entity_link_text():
    globals.read_configuration('config.cfg')
    entity_linker = globals.get_entity_linker()
    parser = globals.get_parser()
    from text2kb.utils import get_questions_serps
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


def entity_link_snippets():
    entity_linker = globals.get_entity_linker()
    snippet_entities_file = globals.config.get('WebSearchFeatures', 'document-snippet-entities')
    from text2kb.utils import get_questions_serps
    question_search_results = get_questions_serps()
    doc_snippet_entities = dict()
    for index, serp in enumerate(question_search_results.itervalues()):
        for doc in serp[:globals.SEARCH_RESULTS_TOPN]:
            snippet_tokens = doc.get_snippet_tokens(skip_parsing=False)
            if not snippet_tokens:
                continue
            entities = entity_linker.identify_entities_in_document(snippet_tokens)
            for entity in entities:
                entity['matches'] = []
                for position in entity['positions']:
                    entity['matches'].append(snippet_tokens[position[0]:position[1]])
            doc_snippet_entities[doc.url] = entities
        if index % 100 == 0:
            logger.info("Processed %d serps" % index)
    logger.info("Pickling the dictionary...")
    with open(snippet_entities_file, 'wx') as out:
        pickle.dump(doc_snippet_entities, out)
    logger.info("Pickling the dictionary DONE!")


def test_new_entity_linker():
    globals.read_configuration('config.cfg')
    from query_translator.translator import CandidateGenerator
    query_translator = CandidateGenerator.get_from_config(globals.config)
    while True:
        question = sys.stdin.readline().strip()
        print "Translation: ", query_translator.translate_query(question)


def get_number_of_external_entities():
    import scorer_globals
    globals.read_configuration('config_webentity.cfg')
    parser = CoreNLPParser.init_from_config()
    entity_linker = WebSearchResultsExtenderEntityLinker.init_from_config()
    entity_linker.topn_entities = 100000
    scorer_globals.init()

    parameters = translator.TranslatorParameters()
    parameters.require_relation_match = False
    parameters.restrict_answer_type = False

    datasets = ["webquestions_split_train", "webquestions_split_dev",]
    # datasets = ["webquestions_split_train_externalentities", "webquestions_split_dev_externalentities",]
    # datasets = ["webquestions_split_train_externalentities3", "webquestions_split_dev_externalentities3",]

    external_entities_count = []
    for dataset in datasets:
        queries = load_eval_queries(dataset)
        for index, query in enumerate(queries):
            entities = entity_linker.identify_entities_in_tokens(parser.parse(query.utterance).tokens, text=query.utterance, find_dates=False)
            print "-------------------------"
            print query.utterance
            print "\n".join(map(str, sorted(entities, key=lambda entity: entity.external_entity_count, reverse=True)))

            external_entities_count.append(0)
            for entity in entities:
                if entity.external_entity:
                    external_entities_count[-1] += 1
            if index % 100 == 0:
                print >> sys.stderr, "%s queries processed" % index
    print "========================================="
    print external_entities_count
    print sum(external_entities_count)
    print len(external_entities_count)


def get_question_terms():
    import scorer_globals
    globals.read_configuration('config_webentity.cfg')
    scorer_globals.init()
    datasets = ["webquestionstrain", "webquestionstest",]

    question_tokens = set()
    for dataset in datasets:
        queries = load_eval_queries(dataset)
        for index, query in enumerate(queries):
            question_tokens.update(token for token in tokenize(query.utterance))
    print question_tokens


def main_parse():
    document_content_file = globals.config.get('WebSearchFeatures', 'documents-content-file')
    parser = CoreNLPParser.init_from_config()
    question_serps = get_questions_serps()
    with open(document_content_file, 'wx') as out_file:
        index = 0
        for serp in tqdm(question_serps.values()):
            for doc in serp[:10]:
                content = doc.content()
                if len(content) > 0:
                    document = (doc.url, parser.parse(content[:10000]))
                    pickle.dump(document, out_file)
            index += 1


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--config',
                        default='config.cfg',
                        help='The configuration file to use.')
    args = parser.parse_args()
    globals.read_configuration(args.config)

    #get_number_of_external_entities()

    # get_question_terms()

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- PARSE DOCUMENT CONTENT AND CACHE PARSE: BEGIN
    # main_parse()
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- PARSE DOCUMENT CONTENT AND CACHE PARSE: END

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- GET DOCUMENT CONTENT ENTITIES: BEGIN
    main_doc_entities_from_content()
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- GET DOCUMENT CONTENT ENTITIES: END

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- GET DOCUMENT SNIPPET ENTITIES: BEGIN
    # entity_link_snippets()
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- GET DOCUMENT SNIPPET ENTITIES: END

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    # globals.read_configuration('config.cfg')
    # linker = globals.get_entity_linker()
    # parser = globals.get_parser()
    # while True:
    #     print "Enter text: "
    #     line = sys.stdin.readline()
    #     tokens = parser.parse(line).tokens
    #     for en in linker.identify_entities_in_tokens(tokens, line):
    #         print en.entity.id, en.entity.get_notable_type()
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


    # test_new_entity_linker()
    # main_entities()  # For entity linking from SERP for a question
    # main_entity_link_text()  # For entity linking from arbitrary text
    # entity_link_snippets()
    # test_new_entity_linker()

    # globals.read_configuration('config_webentity.cfg')
    # entity_linker = WebSearchResultsExtenderEntityLinker.init_from_config()
    # entity_linker.topn_entities = 1000000
    # parser = CoreNLPParser.init_from_config()
    # while True:
    #    print "Please enter a question:"
    #    question = sys.stdin.readline().strip()
    #    tokens = parser.parse(question).tokens
    #    print "\n".join(map(str, entity_linker.identify_entities_in_tokens(tokens, text=question)))
