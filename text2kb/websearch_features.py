from __future__ import print_function
#from corenlp_parser.parser import CoreNLPParser
#from entity_linker.entity_linker import EntityLinker
from query_translator.query_candidate import QueryCandidate

__author__ = 'dsavenk'

import cPickle as pickle
import json
import logging
import re
import os.path
from bs4 import BeautifulSoup, Comment
import justext

import globals

logger = logging.getLogger(__name__)


def extract_text(html_text):
    html = BeautifulSoup(html_text, 'html5lib')

    texts = html.findAll(text=True)

    def visible(element):
        if element.parent.name in ['style', 'script', '[document]', 'title']:
            return False
        elif isinstance(element, Comment):
            return False
        return len(element.string.strip()) > 0 and not element.string.strip().startswith("<")

    visible_texts = filter(visible, texts)

    # get text
    text = [line.strip() for element in visible_texts for line in element.string.splitlines() if len(line.strip()) > 0]
    return '\n'.join(text)


def contains_answer(text, answer):
    tokens = answer.lower().split()
    return 1.0 * sum((1 if token in text else 0 for token in tokens)) / len(tokens)


def distance_features(candidate):
    query_text_tokens = [x.lower() for x in get_query_text_tokens(candidate)]


class WebSearchResult:
    """
    Represents a search results. It lazily reads document content
    in a provided file if asked for its content.
    """

    def __init__(self, url, title, snippet, document_location):
        self.url = url
        self.title = title
        self.snippet = snippet
        self.document_location = document_location

    def content(self):
        """
        :return: Text content of the given document
        """
        try:
            if os.path.isfile(self.document_location):
                import codecs
                with codecs.open(self.document_location, 'r', 'utf-8') as input_document:
                    content = input_document.read()
                    text = justext.justext(content, justext.get_stoplist("English"))
                    res = []
                    for paragraph in text:
                        if not paragraph.is_boilerplate:
                            res.append(paragraph.text)
                    return '\n'.join(res)
                    # return extract_text(content)
            else:
                logger.warning("Document not found: " + str(self.document_location))
        except Exception as exc:
            logger.warning(exc)
        return ""


class WebSearchFeatureGenerator:
    def __init__(self, serp_file, documents_file, entities_file):
        """
        Creates an instance of the web-search based feature generator.
        :param serp_file: A json file with serps for questions.
        :param documents_file: A json file with downloaded docs paths.
        """

        self.question_serps = dict()
        self._read_serp_files(serp_file, documents_file)
        self._document_cache = dict()
        # Entity linker returns a dictionary from url to a list of entities. Each
        # entity is a dictionary with keys: 'mid', 'name', 'surface_score', 'score', 'count'
        self._document_entities = self._read_document_entities(entities_file)
        # self.entity_linker = EntityLinker.init_from_config()
        # self.parser = CoreNLPParser.init_from_config()

    def _read_serp_files(self, serp_file, documents_file):
        # Read a file with search results
        """
        Reads json files and serp results and corresponding documents
        locations and stores this in the self.question_serps field.
        :param serp_file: A json file with serps for questions.
        :param documents_file: A json file with downloaded docs paths.
        """
        with open(serp_file, 'r') as json_input:
            serp_json = json.load(json_input)

        # Read a file with documents paths.
        with open(documents_file, 'r') as json_input:
            url2doc = dict((doc['url'], doc['contentDocument'])
                           for doc in json.load(json_input)['elems'])
        for search_results in serp_json['elems']:
            question = search_results['question']
            question_serp = list()
            self.question_serps[question] = question_serp
            for result in search_results['results']['elems']:
                question_serp.append(
                    WebSearchResult(result['url'],
                                    result['title'],
                                    result['snippet'],
                                    url2doc[result['url']]))


    def _read_document_entities(self, entities_file):
        with open(entities_file, 'r') as input:
            return pickle.load(input)


    @staticmethod
    def init_from_config():
        config_options = globals.config
        train_serp_file = config_options.get('WebSearchFeatures', 'train-serp-file')
        train_documents_file = config_options.get('WebSearchFeatures', 'train-documents-file')
        test_serp_file = config_options.get('WebSearchFeatures', 'test-serp-file')
        test_documents_file = config_options.get('WebSearchFeatures', 'test-documents-file')
        entities_file = config_options.get('WebSearchFeatures', 'document-entities-file')
        logger.info("Reading web search results data...")
        res = WebSearchFeatureGenerator(train_serp_file, train_documents_file, entities_file)
        res._read_serp_files(test_serp_file, test_documents_file)
        return res

    def generate_features(self, candidate):
        answers = candidate.query_results

        if len(answers) == 0:
            return {
                'top_document_match': 0.0,
                'search_doc_count': 0.0,
                'search_doc_match_count': 0.0,
                'search_doc_nonmatch_count': 0.0,
                'search_snippets_count': 0.0,
                'search_doc_entity_count': 0.0
            }

        answers_top_doc = 0
        answers_match_count = 0
        answers_non_match_count = 0
        answers_doc_counts = [0, ] * len(answers)
        answers_snip_counts = [0, ] * len(answers)
        answers_entities_counts = [0, ] * len(answers)
        question = candidate.query.original_query
        if question in self.question_serps:
            # TODO(denxx): need to keep top-50 if possible.
            for rank, doc in enumerate(self.question_serps[question][:10]):
                if doc.document_location not in self._document_cache:
                    document_content = doc.content()
                    #if len(document_content.strip()) > 0:
                    #    tokens = self.parser.parse(document_content).tokens
                    #    document_entities = self.entity_linker.identify_entities_in_document(tokens)
                    #else:
                    #    document_entities = []
                    document_content = set(document_content.lower().split())
                    self._document_cache[doc.document_location] = document_content
                    #self._document_entities_cache[doc.document_location] = document_entities

                document_content = self._document_cache[doc.document_location]
                if doc.url in self._document_entities:
                    document_entities = set(entity['name']
                                            for entity in self._document_entities[doc.url])
                else:
                    document_entities = set()
                document_snippet = set(doc.snippet.lower().split())
                for i, answer in enumerate(answers):
                    doc_contains_answer = contains_answer(document_content, answer)
                    if doc_contains_answer > 0.5:
                        answers_match_count += 1
                    else:
                        answers_non_match_count += 1

                    snippet_contains_answer = contains_answer(document_snippet, answer)
                    answers_doc_counts[i] += doc_contains_answer
                    if rank == 0:
                        answers_top_doc += doc_contains_answer
                    answers_snip_counts[i] += snippet_contains_answer
                    if answer in document_entities:
                        answers_entities_counts[i] += 1

        return {
                'top_document_match': 1.0 * answers_top_doc / len(answers),
                'search_doc_count': 1.0 * sum(answers_doc_counts) / len(answers),
                'search_doc_match_count': 1.0 * answers_match_count / len(answers),
                'search_doc_nonmatch_count': 1.0 * answers_non_match_count / len(answers),
                'search_snippets_count': 1.0 * sum(answers_snip_counts) / len(answers),
                'search_doc_entity_count': 1.0 * sum(answers_entities_counts) / len(answers)
                }


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s : %(levelname)s "
                               ": %(module)s : %(message)s",
                        level=logging.INFO)
    import sys
    feature_generator = WebSearchFeatureGenerator(sys.argv[1], sys.argv[2])
    doc = feature_generator.question_serps[u'what character did natalie portman play in star wars?'][0]
    content = doc.content()
    pass
