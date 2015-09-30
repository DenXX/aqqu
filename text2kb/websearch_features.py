from __future__ import print_function
from query_translator.query_candidate import QueryCandidate

__author__ = 'dsavenk'

import json
import logging
import os.path
from bs4 import BeautifulSoup

import globals

logger = logging.getLogger(__name__)


def extract_text(html_text):
    html = BeautifulSoup(html_text, 'html5lib')
    for script in html(["script", "style"]):
        script.extract()  # rip it out

    # get text
    text = html.get_text()
    # break into lines and remove leading and trailing space on each
    lines = (line.strip() for line in text.splitlines())
    # break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    # drop blank lines
    return '\n'.join(chunk for chunk in chunks if chunk)


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
                    return content
                    # return extract_text(content)
            else:
                logger.warning("Document not found: " + str(self.document_location))
        except Exception as exc:
            logger.warning(exc)
        return ""


def contains_answer(text, answer):
    text = text.lower()
    tokens = answer.lower().split()
    return 1.0 * sum((1 if token in text else 0 for token in tokens)) / len(tokens) > 0.7


class WebSearchFeatureGenerator:
    def __init__(self, serp_file, documents_file):
        """
        Creates an instance of the web-search based feature generator.
        :param serp_file: A json file with serps for questions.
        :param documents_file: A json file with downloaded docs paths.
        """

        self.question_serps = dict()
        self._read_serp_files(serp_file, documents_file)

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

    @staticmethod
    def init_from_config():
        config_options = globals.config
        train_serp_file = config_options.get('WebSearchFeatures', 'train-serp-file')
        train_documents_file = config_options.get('WebSearchFeatures', 'train-documents-file')
        test_serp_file = config_options.get('WebSearchFeatures', 'test-serp-file')
        test_documents_file = config_options.get('WebSearchFeatures', 'test-documents-file')
        logger.info("Reading web search results data...")
        res = WebSearchFeatureGenerator(train_serp_file, train_documents_file)
        res._read_serp_files(test_serp_file, test_documents_file)
        return res

    def generate_features(self, candidate):
        answers = candidate.query_results

        if len(answers) == 0:
            return {'search_doc_count': 0.0, 'search_snippets_count': 0.0}

        answers_doc_counts = [0, ] * len(answers)
        answers_snip_counts = [0, ] * len(answers)
        question = candidate.query.original_query
        if question in self.question_serps:
            for doc in self.question_serps[question]:
                document_content = doc.content()
                for i, answer in enumerate(answers):
                    if contains_answer(document_content, answer):
                        answers_doc_counts[i] += 1
                    if contains_answer(doc.snippet, answer):
                        answers_snip_counts[i] += 1

        return {'search_doc_count': 1.0 * sum(answers_doc_counts) / len(answers),
                'search_snippets_count': 1.0 * sum(answers_snip_counts) / len(answers)}


if __name__ == "__main__":
    import sys
    feature_generator = WebSearchFeatureGenerator(sys.argv[1], sys.argv[2])
    pass
