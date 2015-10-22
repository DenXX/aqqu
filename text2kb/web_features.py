from math import sqrt
import justext
import cPickle as pickle
import json
import logging
from math import log
import operator
import globals
import os
from query_translator.features import get_query_text_tokens
from collections import Counter

__author__ = 'dsavenk'

WINDOW_SIZE = 5
SLIDING_WINDOW_SIZE = 20

logger = logging.getLogger(__name__)

# Global caches for search results, document contents and list of entities mentioned in the document.
_question_search_results = None
_documents_content = None
_documents_entities = None
_document_snippet_entities = None


def get_questions_serps():
    global _question_search_results

    if _question_search_results is None:
        logger.info("Reading search results...")
        serp_files = globals.config.get('WebSearchFeatures', 'serp-files').split(',')
        document_files = globals.config.get('WebSearchFeatures', 'documents-files').split(',')
        _question_search_results = dict()
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
            _question_search_results[question] = question_serp
            for result in search_results['results']['elems']:
                question_serp.append(
                    WebSearchResult(result['url'],
                                    result['title'],
                                    result['snippet'],
                                    url2doc[result['url']]))
        logger.info("Reading search results done!")
    return _question_search_results


def get_documents_content_dict(return_parsed_tokens=False):
    if _documents_content is None:
        logger.info("Reading documents content...")
        global _documents_content
        document_content_file = globals.config.get('WebSearchFeatures', 'documents-content-file')

        _documents_content = dict()
        with open(document_content_file, 'r') as content_input:
            # Unpickle until the end of file is reached
            index = 0
            while True:
                try:
                    url, content = pickle.load(content_input)
                    if not return_parsed_tokens:
                        tokens = [(token.token.lower(), token.lemma.lower()) for token in content.tokens]
                        _documents_content[url] = tokens
                    else:
                        _documents_content[url] = content.tokens
                except (EOFError, pickle.UnpicklingError):
                    break
                index += 1
                if index % 1000 == 0:
                    logger.info("Read " + str(index) + " documents...")
        logger.info("Reading documents content done!")
    return _documents_content


def get_documents_entities():
    global _documents_entities
    if _documents_entities is None:
        logger.info("Reading documents entities...")
        document_entities_file = globals.config.get('WebSearchFeatures', 'documents-entities-file')
        _documents_entities = dict()
        with open(document_entities_file) as entities_file:
            url_entities = pickle.load(entities_file)
        for url, entities in url_entities.iteritems():
            doc_entities = dict()
            _documents_entities[url] = doc_entities
            for entity in entities:
                doc_entities[entity['name'].lower()] = entity
        logger.info("Reading documents entities done!")
    return _documents_entities


def get_documents_snippet_entities():
    global _document_snippet_entities
    if _document_snippet_entities is None:
        document_snippet_entities_file = globals.config.get('WebSearchFeatures', 'document-snippet-entities')
        logger.info("Reading documents snippet entities...")
        with open(document_snippet_entities_file) as entities_file:
            _document_snippet_entities = pickle.load(entities_file)
        logger.info("Reading documents snippet entities done!")
    return _document_snippet_entities


def _answer_contains(answer_tokens, doc_tokens_set):
    if len(answer_tokens) == 0:
        return 0.0
    return sum(1.0 for answer_token in answer_tokens if answer_token in doc_tokens_set) / len(answer_tokens)


def _compute_tokenset_similarity(question_tokenset, answer_tokenset):
    """
    Computes the cosine similarity between two sets of tokens.
    :param question_tokens:
    :param answer_tokens_neighborhood:
    :return:
    """
    if len(question_tokenset) == 0 or len(answer_tokenset) == 0:
        return 0
    res = 0
    sum = 0
    for token, count in answer_tokenset.iteritems():
        sum += count
        if token in question_tokenset:
            res += count
    return 1.0 * res / (sqrt(len(question_tokenset))) / sqrt(sum)


def _compute_sliding_window_score(matched_positions, token_to_pos, lemma_to_pos, answer_tokens):
    total_token_counts = dict((token, len(pos)) for token, pos in token_to_pos.iteritems())
    total_token_counts.update(dict((lemma, len(pos)) for lemma, pos in lemma_to_pos.iteritems()))
    begin = 0
    end = 0
    token_counts = dict()
    answer_token_counts = dict()
    score = 0
    best_beg = 0
    best_end = 0
    current_score = 0
    while begin < len(matched_positions):
        while end < len(matched_positions) and \
                                matched_positions[end][0] - matched_positions[begin][0] < SLIDING_WINDOW_SIZE:
            token = matched_positions[end][1]
            if token not in token_counts:
                token_counts[token] = 0
                current_score += log(1 + 1.0 / total_token_counts[token])
            if token in answer_tokens:
                if token not in answer_token_counts:
                    answer_token_counts[token] = 0
                answer_token_counts[token] += 1
            token_counts[token] += 1
            end += 1

        if len(answer_token_counts.keys()) > 0.7 * len(answer_tokens) and current_score > score:
            score = current_score
            best_beg = begin
            best_end = end

        # Exclude the beginning token
        token = matched_positions[begin][1]
        token_counts[token] -= 1
        if token_counts[token] == 0:
            del token_counts[token]
            current_score -= log(1 + 1.0 / total_token_counts[token])
        if token in answer_tokens:
            answer_token_counts[token] -= 1
            if answer_token_counts[token] == 0:
                del answer_token_counts[token]
        begin += 1
    return score, best_beg, best_end


def _compute_min_distance_question_answer(question_tokens_positions, answer_tokens_positions, document_length):
    INF = 1000000
    diff = INF
    question_tokens_pos = set(map(lambda x: x[0], question_tokens_positions))
    answer_tokens_pos = map(lambda x: x[0], answer_tokens_positions.difference(question_tokens_pos))
    question_tokens_pos = sorted(question_tokens_pos)
    answer_tokens_pos = sorted(answer_tokens_pos)
    q_index = 0
    a_index = 0

    while q_index < len(question_tokens_pos) and a_index < len(answer_tokens_pos):
        q_pos = question_tokens_pos[q_index]
        a_pos = answer_tokens_pos[a_index]
        diff = min(diff, abs(q_pos - a_pos))
        if q_pos < a_pos:
            q_index += 1
        else:
            a_index += 1

    return 1.0 / (document_length + 1) * diff if diff < INF else 1.0


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
        self.lemma_to_pos = None
        self.snippet_tokens = None
        self.snippet_entities = None

    def mentioned_entities(self):
        """
        :return: A list of mentioned entities.
        """
        global _documents_entities
        if _documents_entities and self.url in _documents_entities:
            return _documents_entities[self.url]
        return None

    def parsed_content(self):
        """
        :return: Parsed content of the current document
        """
        if _documents_content is not None and self.url in _documents_content:
            return _documents_content[self.url]
        # TODO(denxx): Here we need to parse, but to parse we need parser. Should make a global parser.
        return None

    def get_content_tokens_set(self):
        """
        :return: Returns a set of document tokens. The set is computed and cached the first time the method
        is called.
        """
        if self.content_tokens is not None:
            return self.content_tokens
        tokens = self.parsed_content()
        self.content_tokens = set(token[0] for token in tokens) if tokens else set()
        return self.content_tokens

    def get_token_to_positions_map(self):
        """
        :return: Returns a map from token to its position in the document. The map
        is computed the first time the method is called and then the result is cached.
        """
        if self.token_to_pos is None:
            self.token_to_pos = dict()
            self.lemma_to_pos = dict()
            if self.parsed_content() is not None:
                for pos, token in enumerate(self.parsed_content()):
                    lemma = token[1].lower()
                    token = token[0].lower()
                    if token not in self.token_to_pos:
                        self.token_to_pos[token] = []
                    if lemma not in self.lemma_to_pos:
                        self.lemma_to_pos[lemma] = []
                    self.token_to_pos[token].append(pos)
                    self.lemma_to_pos[lemma].append(pos)
        return self.token_to_pos, self.lemma_to_pos

    def get_snippet_tokens(self):
        if self.snippet_tokens is None:
            parser = globals.get_parser()
            self.snippet_tokens = parser.parse(self.title + u'\n' + self.snippet).tokens
        return self.snippet_tokens

    def get_snippet_entities(self):
        doc_snippet_entities = get_documents_snippet_entities()
        if self.url in doc_snippet_entities:
            return doc_snippet_entities[self.url]

        logger.warning("Didn't find cached document snippet entities.")
        # If we didn't find snippet entities in cache, use entity linker.
        if self.snippet_entities is None:
            entity_linker = globals.get_entity_linker()
            self.snippet_entities = dict(
                (entity['name'].lower(), entity['count'])
                for entity in entity_linker.identify_entities_in_document(self.get_snippet_tokens(),
                                                                          max_token_window=4,
                                                                          min_surface_score=0.5))
        return self.snippet_entities

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


class WebFeatureGenerator:
    """
    Generates candidate features based on web search results.
    """

    def __init__(self):
        self.question_search_results = None

    @staticmethod
    def init_from_config():
        return WebFeatureGenerator()

    def generate_features(self, candidate):
        """
        Generates features for the candidate answer bases on search results documents.
        :param candidate: Candidate answer to generate features for
        :return: A dictionary from feature name to feature value.
        """
        answers = candidate.get_results_text()

        # Read search results data.
        if self.question_search_results is None:
            self.question_search_results = get_questions_serps()

        if answers is None:
            logger.error("Answers is None!")
            return dict()

        # Ignoring empty and extra long answers, these are unlikely to be correct anyway.
        if len(answers) == 0 or len(answers) > 10:
            return dict()

        question = candidate.query.original_query

        # If we don't have anything for the question...
        if question not in self.question_search_results:
            logger.warning("No documents found for question: " + question)
            return dict()

        question_entities = [entity.entity.name.lower() for entity in candidate.matched_entities]

        stopwords = globals.get_stopwords()
        question_tokens = set(filter(lambda token: token not in stopwords, get_query_text_tokens(candidate)))
        answers_lower = map(unicode.lower, answers)
        answers_tokens = [filter(lambda token: token not in stopwords, answer.split())
                          for answer in answers_lower]

        # Counters which will be used to compute features
        answer_occurances_entity = [0, ] * len(answers)
        answer_occurances_snippet_entity = [0, ] * len(answers)
        answer_occurances_snippet_text = [0, ] * len(answers)
        answer_doc_occurances_entity = [0, ] * len(answers)
        answer_doc_occurances_text = [0, ] * len(answers)
        answer_context_question_similarity = [0.0, ] * len(answers)
        question_entities_in_snippets = 0
        sliding_window_score = [0.0, ] * len(answers)
        min_distance_question_answer_token = [0.0, ] * len(answers)
        answer_neighbourhood = []
        for i in xrange(len(answers)):
            answer_neighbourhood.append(set())

        for rank, doc in enumerate(self.question_search_results[question][:globals.SEARCH_RESULTS_TOPN]):
            doc_content = doc.parsed_content()
            doc_tokens = doc.get_content_tokens_set()
            doc_entities = doc.mentioned_entities()
            seen_answers = [False, ] * len(answers)

            snippets_tokens = [token.token.lower() for token in doc.get_snippet_tokens()]
            snippet_entities = doc.get_snippet_entities()

            for question_entity in question_entities:
                if question_entity in snippet_entities:
                    question_entities_in_snippets += snippet_entities[question_entity]

            # Check which answer entities are present in the document
            for i, answer in enumerate(answers_lower):
                if doc_entities is not None and answer in doc_entities:
                    answer_occurances_entity[i] += doc_entities[answer]['count']
                    seen_answers[i] = True

                if answer in snippet_entities:
                    answer_occurances_snippet_entity[i] += snippet_entities[answer]

            for i, seen in enumerate(seen_answers):
                if seen:
                    answer_doc_occurances_entity[i] += 1

            token_to_pos, lemma_to_pos = doc.get_token_to_positions_map()
            for i, answer_tokens in enumerate(answers_tokens):
                if _answer_contains(answer_tokens, snippets_tokens) > 0.7:
                    answer_occurances_snippet_text[i] += 1

                if doc_content is not None:
                    if _answer_contains(answer_tokens, doc_tokens) > 0.7:
                        answer_doc_occurances_text[i] += 1

                    # Get a set of positions of answer tokens in the target document.
                    answer_tokens_positions = set((pos, answer_token)
                                                  for answer_token in answer_tokens
                                                  if answer_token in token_to_pos
                                                  for pos in token_to_pos[answer_token])
                    question_tokens_positions = set((pos, question_lemma)
                                                    for question_lemma in question_tokens
                                                    if question_lemma in lemma_to_pos
                                                    for pos in lemma_to_pos[question_lemma])
                    matched_positions = sorted(answer_tokens_positions.union(question_tokens_positions),
                                               key=operator.itemgetter(0))
                    score, beg, end = _compute_sliding_window_score(matched_positions,
                                                                   token_to_pos,
                                                                   lemma_to_pos,
                                                                   answer_tokens)
                    sliding_window_score[i] += score
                    min_distance_question_answer_token[i] += _compute_min_distance_question_answer(
                        question_tokens_positions, answer_tokens_positions, len(doc_content))

                    # Get a set of actual tokens in the neighbourhood of answer tokens.
                    answer_tokens_neighborhood = Counter([doc_content[neighbor][1]  # [1] is lemma
                                                          for pos, _ in answer_tokens_positions
                                                          for neighbor in range(max(0, pos - WINDOW_SIZE),
                                                                                min(len(doc_content),
                                                                                    pos + WINDOW_SIZE + 1))
                                                          if neighbor not in answer_tokens_positions])
                    answer_context_question_similarity[i] += _compute_tokenset_similarity(question_tokens,
                                                                                         answer_tokens_neighborhood)

        return {
            'web_results:answer_entity_occurances': 1.0 * sum(answer_occurances_entity) / len(answers),
            'web_results:answer_entity_doc_count': 1.0 * sum(answer_doc_occurances_entity) / len(answers),
            'web_results:answer_text_doc_count': 1.0 * sum(answer_doc_occurances_text) / len(answers),
            'web_results:sliding_window_score': 1.0 * sum(sliding_window_score) / len(answers),
            'web_results:min_distance_to_question_term_score': 1.0 * sum(min_distance_question_answer_token) / len(
                answers),
            'web_results:answer_context_question_similarity':
                1.0 * sum(answer_context_question_similarity) / len(answers),
            'web_results:snippets:answer_entity_occurances': 1.0 * sum(answer_occurances_snippet_entity) / len(answers),
            'web_results:snippets:answer_text_occurances': 1.0 * sum(answer_occurances_snippet_text) / len(answers),
            'web_results:question_entity_in_snippets': question_entities_in_snippets
        }


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s : %(levelname)s '
                               ': %(module)s : %(message)s',
                        level=logging.INFO)
    globals.read_configuration('config.cfg')
    feature_generator = WebFeatureGenerator.init_from_config()
    print feature_generator
