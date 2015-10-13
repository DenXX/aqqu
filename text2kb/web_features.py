from math import sqrt
import justext
import cPickle as pickle
import json
import logging
from math import log
import globals
import os
from query_translator.features import get_query_text_tokens
from collections import Counter

__author__ = 'dsavenk'

TOPN = 10
WINDOW_SIZE = 3
SLIDING_WINDOW_SIZE = 20

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
    return question_search_results


def _read_document_content(document_content_file):
    logger.info("Reading documents content...")
    global documents_content
    documents_content = dict()
    with open(document_content_file, 'r') as content_input:
        # Unpickle until the end of file is reached
        index = 0
        while True:
            try:
                url, content = pickle.load(content_input)
                tokens = [(token.token.lower(), token.lemma.lower()) for token in content.tokens]
                documents_content[url] = tokens
            except (EOFError, pickle.UnpicklingError):
                break
            index += 1
            if index % 1000 == 0:
                logger.info("Read " + str(index) + " documents...")
    logger.info("Reading documents content done!")
    return documents_content


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
            doc_entities[entity['name'].lower()] = entity
    logger.info("Reading documents entities done!")


def answer_contains(answer_tokens, doc_tokens_set):
    if len(answer_tokens) == 0:
        return 0.0
    return sum(1.0 for answer_token in answer_tokens if answer_token in doc_tokens_set) / len(answer_tokens)


def compute_tokenset_similarity(question_tokenset, answer_tokenset):
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


def compute_sliding_window_score(question_tokens, answer_tokens, doc_content):
    begin_index = 0
    end_index = min(len(doc_content), SLIDING_WINDOW_SIZE)

    question_lemmas = set(question_tokens)
    answer_tokens = set(answer_tokens)
    passage_tokens = dict()
    passage_lemma = dict()
    score = 0
    for i in xrange(end_index):
        token, lemma = doc_content[i]
        score = update_score_for_token(answer_tokens, score, passage_tokens, token, +1)
        score = update_score_for_token(question_lemmas, score, passage_lemma, lemma, +1)

    current_score = score
    while end_index < len(doc_content):
        to_remove_token, to_remove_lemma = doc_content[begin_index]
        current_score = update_score_for_token(answer_tokens, current_score, passage_tokens, to_remove_token, -1)
        current_score = update_score_for_token(question_lemmas, current_score, passage_lemma, to_remove_lemma, -1)

        new_token, new_lemma = doc_content[end_index]

        current_score = update_score_for_token(answer_tokens, current_score, passage_tokens, new_token, +1)
        current_score = update_score_for_token(question_lemmas, current_score, passage_lemma, new_lemma, +1)

        score = max(score, current_score)
        begin_index += 1
        end_index += 1

    return score


def update_score_for_token(target_tokens, current_score, passage_tokens, to_remove_token, delta):
    if to_remove_token not in passage_tokens:
        passage_tokens[to_remove_token] = 0

    # If the token belongs to the target set (question or answer tokens), them we update the score
    if to_remove_token in target_tokens:
        count = passage_tokens[to_remove_token]
        if count > 0:
            current_score -= log(1.0 + 1.0 / count)
        if count + delta > 0:
            current_score += log(1.0 + 1.0 / (count + delta))

    # Update the counts.
    passage_tokens[to_remove_token] += delta
    if passage_tokens[to_remove_token] == 0:
        del passage_tokens[to_remove_token]

    return current_score


def compute_min_distance_question_answer(question_tokens_positions, answer_tokens_positions, document_length):
    INF = 1000000
    diff = INF
    question_tokens_positions = sorted(question_tokens_positions)
    answer_tokens_positions = sorted(answer_tokens_positions)
    q_index = 0
    a_index = 0

    while q_index < len(question_tokens_positions) and a_index < len(answer_tokens_positions):
        q_pos = question_tokens_positions[q_index]
        a_pos = answer_tokens_positions[a_index]
        diff = min(diff, abs(q_pos - a_pos))
        if q_pos < a_pos:
            q_pos += 1
        else:
            a_pos += 1

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
        global documents_entities
        if documents_entities and self.url in documents_entities:
            return documents_entities[self.url]
        return None

    def parsed_content(self):
        """
        :return: Parsed content of the current document
        """
        if documents_content is not None and self.url in documents_content:
            return documents_content[self.url]
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

    def __init__(self, serp_files, documents_files, entities_file, content_file):
        # Reading data if needed
        if question_search_results is None:
            _read_serp_files(serp_files, documents_files)
        if documents_entities is None:
            _read_entities(entities_file)
        if documents_content is None:
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
        answers = candidate.get_results_text()

        if answers is None:
            logger.error("Answers is None!")
            return dict()

        # Ignoring empty and extra long answers, these are unlikely to be correct anyway.
        if len(answers) == 0 or len(answers) > 10:
            return dict()

        question = candidate.query.original_query

        # If we don't have anything for the question...
        if question not in question_search_results:
            logger.warning("No documents found for question: " + question)
            return dict()

        question_entities = [entity.entity.name.lower() for entity in candidate.matched_entities]

        question_tokens = set(get_query_text_tokens(candidate))
        stopwords = globals.get_stopwords()
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

        for rank, doc in enumerate(question_search_results[question][:TOPN]):
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
                if answer_contains(answer_tokens, snippets_tokens) > 0.7:
                    answer_occurances_snippet_text[i] += 1

                if doc_content is not None:
                    if answer_contains(answer_tokens, doc_tokens) > 0.7:
                        answer_doc_occurances_text[i] += 1

                    sliding_window_score[i] += compute_sliding_window_score(question_tokens, answer_tokens, doc_content)

                    # Get a set of positions of answer tokens in the target document.
                    answer_tokens_positions = set(pos
                                                  for answer_token in answer_tokens
                                                  if answer_token in token_to_pos
                                                  for pos in token_to_pos[answer_token])
                    question_tokens_positions = set(pos for question_lemma in question_tokens
                                                    if question_lemma in lemma_to_pos
                                                    for pos in lemma_to_pos[question_lemma]
                                                    if pos not in answer_tokens_positions)

                    min_distance_question_answer_token[i] += compute_min_distance_question_answer(
                        question_tokens_positions, answer_tokens_positions, len(doc_content))


                    # Get a set of actual tokens in the neighbourhood of answer tokens.
                    answer_tokens_neighborhood = Counter([doc_content[neighbor][1]  # [1] is lemma
                                                          for pos in answer_tokens_positions
                                                          for neighbor in range(max(0, pos - WINDOW_SIZE),
                                                                                min(len(doc_content),
                                                                                    pos + WINDOW_SIZE + 1))
                                                          if neighbor not in answer_tokens_positions])
                    answer_context_question_similarity[i] += compute_tokenset_similarity(question_tokens,
                                                                                         answer_tokens_neighborhood)

        return {
            'web_results:answer_entity_occurances': 1.0 * sum(answer_occurances_entity) / len(answers),
            'web_results:answer_entity_doc_count': 1.0 * sum(answer_doc_occurances_entity) / len(answers),
            'web_results:answer_text_doc_count': 1.0 * sum(answer_doc_occurances_text) / len(answers),
            'web_results:sliding_window_score': 1.0 * sum(sliding_window_score) / len(answers),
            'web_results:min_distance_to_question_term_score': 1.0 * sum(min_distance_question_answer_token) / len(answers),
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
