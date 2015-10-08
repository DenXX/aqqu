from math import sqrt
import justext
import cPickle as pickle
import json
import logging
import globals
import os
from query_translator.features import get_query_text_tokens

__author__ = 'dsavenk'

TOPN = 10
WINDOW_SIZE = 3

logger = logging.getLogger(__name__)

# Global caches for search results, document contents and list of entities mentioned in the document.
question_search_results = None
documents_content = None
documents_entities = None


def _read_serp_files(serp_files, document_files):
    global question_search_results

    logger.info("Reading search results...")
    question_search_results = dict()
    serps = []
    for serp_file in serp_files:
        with open(serp_file, 'r') as json_input:
            serps.extend(json.load(json_input)['elems'])

    # Read a file with documents paths.
    url2doc = dict()
    for document_file in document_files:
        with open(document_file, 'r') as json_input:
            url2doc.update(dict((doc['url'], doc['contentDocument'])
                                for doc in json.load(json_input)['elems']))

    for search_results in serps:
        question = search_results['question']
        question_serp = list()
        question_search_results[question] = question_serp
        for result in search_results['results']['elems']:
            question_serp.append(
                WebSearchResult(result['url'],
                                result['title'],
                                result['snippet'],
                                url2doc[result['url']]))
    logger.info("Reading search results done!")


def _read_document_content(document_content_file):
    logger.info("Reading documents content...")
    global documents_content
    documents_content = dict()
    with open(document_content_file, 'r') as content_input:
        # Unpickle until the end of file is reached
        while True:
            try:
                url, content = pickle.load(content_input)
                tokens = ((token.token.lower(), token.lemma.lower()) for token in content.tokens)
                documents_content[url] = tokens
            except (EOFError, pickle.UnpicklingError):
                break
    logger.info("Reading documents content done!")


def _read_entities(document_entities_file):
    logger.info("Reading documents entities...")
    global documents_entities
    documents_entities = dict()
    with open(document_entities_file) as entities_file:
        url_entities = pickle.load(entities_file)
    for url, entities in url_entities.iteritems():
        doc_entities = dict()
        documents_entities[url] = doc_entities
        for entity in entities:
            documents_entities[entity['name'].lower()] = entity
    logger.info("Reading documents entities done!")


def answer_contains(answer_tokens, doc_tokens_set):
    if len(answer_tokens) == 0:
        return 0.0
    return sum(1.0 for answer_token in answer_tokens if answer_token in doc_tokens_set) / len(answer_tokens)


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
        self.content_tokens = None
        self.token_to_pos = None

    def mentioned_entities(self):
        """
        :return: A list of mentioned entities.
        """
        global documents_entities
        if documents_entities and self.url in documents_entities:
            return documents_entities[self.url]
        return None

    def parsed_content(self):
        """
        :return: Parsed content of the current document
        """
        global documents_content
        if documents_content and self.url in documents_content:
            return documents_content[self.url]
        # TODO(denxx): Here we need to parse, but to parse we need parser. Should make a global parser.
        return None

    def get_content_tokens_set(self):
        """
        :return: Returns a set of document tokens. The set is computed and cached the first time the method
        is called.
        """
        if self.content_tokens:
            return self.content_tokens
        tokens = self.parsed_content()
        self.content_tokens = set(token[0] for token in tokens) if tokens else set()
        return self.content_tokens

    def get_token_to_positions_map(self):
        """
        :return: Returns a map from token to its position in the document. The map
        is computed the first time the method is called and then the result is cached.
        """
        if self.token_to_pos:
            return self.token_to_pos
        self.token_to_pos = dict()
        for pos, token in enumerate(self.parsed_content()):
            token = token[0].lower()
            if token not in self.token_to_pos:
                self.token_to_pos[token] = []
            self.token_to_pos[token].append(pos)
        return self.token_to_pos

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


def compute_tokenset_similarity(tokenset_1, tokenset_2):
    """
    Computes the cosine similarity between two sets of tokens.
    :param question_tokens:
    :param answer_tokens_neighborhood:
    :return:
    """
    return 1.0 * len(tokenset_1.intersection(tokenset_2)) / \
           (sqrt(len(tokenset_1))) * sqrt(len(tokenset_2))


class WebFeatureGenerator:
    """
    Generates candidate features based on web search results.
    """

    def __init__(self, serp_files, documents_files, entities_file, content_file):
        # Reading data if needed
        if not question_search_results:
            _read_serp_files(serp_files, documents_files)
        if not documents_entities:
            _read_entities(entities_file)
        if not documents_content:
            _read_document_content(content_file)

    @staticmethod
    def init_from_config():
        config_options = globals.config
        serp_files = config_options.get('WebSearchFeatures', 'serp-files').split(',')
        documents_files = config_options.get('WebSearchFeatures', 'documents-files').split(',')
        entities_file = config_options.get('WebSearchFeatures', 'documents-entities-file')
        content_file = config_options.get('WebSearchFeatures', 'documents-content-file')
        return WebFeatureGenerator(serp_files, documents_files, entities_file, content_file)

    def generate_features(self, candidate):
        """
        Generates features for the candidate answer bases on search results documents.
        :param candidate: Candidate answer to generate features for
        :return: A dictionary from feature name to feature value.
        """
        answers = candidate.query_results

        # Ignoring empty and extra long answers, these are unlikely to be correct anyway.
        if len(answers) == 0 or len(answers) > 10:
            return dict()

        question = candidate.query.original_query

        # If we don't have anything for the question...
        if question not in question_search_results:
            logger.warning("No documents found for question: " + question)
            return dict()

        question_tokens = set(get_query_text_tokens(candidate))
        stopwords = globals.get_stopwords()
        answers_lower = map(unicode.lower, answers)
        answers_tokens = [filter(lambda token: token not in stopwords, answer.split())
                          for answer in answers_lower]

        # Counters which will be used to compute features
        answer_occurances_entity = [0, ] * len(answers)
        answer_doc_occurances_entity = [0, ] * len(answers)
        answer_doc_occurances_text = [0, ] * len(answers)
        answer_context_question_similarity = [0.0, ] * len(answers)
        answer_neighbourhood = []
        for i in xrange(len(answers)):
            answer_neighbourhood.append(set())

        for rank, doc in enumerate(question_search_results[question][:TOPN]):
            doc_content = doc.parsed_content()
            doc_tokens = doc.get_content_tokens_set()
            doc_entities = doc.mentioned_entities()
            seen_answers = [False, ] * len(answers)

            # Check which answer entities are present in the document
            if doc_entities:
                for i, answer in enumerate(answers_lower):
                    if answer in doc_entities:
                        answer_occurances_entity[i] += doc_entities[answer]['count']
                        seen_answers[i] = True

            for i, seen in enumerate(seen_answers):
                if seen:
                    answer_doc_occurances_entity[i] += 1

            if doc_content:
                token_to_pos = doc.get_token_to_positions_map()
                for i, answer_tokens in enumerate(answers_tokens):
                    if answer_contains(answer_tokens, doc_tokens) > 0.5:
                        answer_doc_occurances_text[i] += 1

                    # Get a set of positions of answer tokens in the target document.
                    answer_tokens_positions = set(pos
                                                  for answer_token in answer_tokens
                                                  if answer_token in token_to_pos
                                                  for pos in token_to_pos[answer_token])

                    # Get a set of actual tokens in the neighbourhood of answer tokens.
                    answer_tokens_neighborhood = set(doc_content[neighbor][1]  # [1] is lemma
                                                     for pos in answer_tokens_positions
                                                     for neighbor in range(max(0, pos - WINDOW_SIZE),
                                                                           min(len(doc_content),
                                                                               pos + WINDOW_SIZE + 1))
                                                     if neighbor not in answer_tokens_positions)
                    answer_context_question_similarity[i] += compute_tokenset_similarity(question_tokens,
                                                                                         answer_tokens_neighborhood)

        return {
            'web_results:answer_entity_occurances': 1.0 * sum(answer_occurances_entity) / len(answers),
            'web_results:answer_entity_doc_occurances': 1.0 * sum(answer_doc_occurances_entity) / len(answers),
            'web_results:answer_text_doc_occurances': 1.0 * sum(answer_doc_occurances_text) / len(answers),
            'web_results:answer_context_question_similarity':
                1.0 * sum(answer_context_question_similarity) / len(answers),
        }


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s : %(levelname)s '
                               ': %(module)s : %(message)s',
                        level=logging.INFO)
    globals.read_configuration('config.cfg')
    feature_generator = WebFeatureGenerator.init_from_config()
    print feature_generator
