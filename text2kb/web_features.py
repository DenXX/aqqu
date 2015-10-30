from math import sqrt
import justext
import cPickle as pickle
import json
import logging
from math import log
import operator
from entity_linker.entity_linker import KBEntity
import globals
import os
import string
from query_translator.features import get_query_text_tokens
from collections import Counter, deque

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


# TODO(denxx): the parameter return_parsed_tokens shouldn't be here. The result is cached, so I can't call with
# different params.
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
                # if index > 1000:
                #    break
        logger.info("Reading documents content done!")
    return _documents_content


def get_documents_entities():
    """
    :return: Returns a dictionary from document url to the dictionary of mentioned entities. Each element of the list
    is a dictionary with fields name, mid, surface_score, score, positions, count.
    """
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
        _document_snippet_entities = dict()
        with open(document_snippet_entities_file) as entities_file:
            url_entities_dict = pickle.load(entities_file)
        for url, entities in url_entities_dict.iteritems():
            doc_snippet_entities = dict()
            _document_snippet_entities[url] = doc_snippet_entities
            for entity in entities:
                doc_snippet_entities[entity['name'].lower()] = entity
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
        :return: A dictionary of mentioned entities (name -> entity).
        """
        documents_entities = get_documents_entities()
        if self.url in documents_entities:
            return documents_entities[self.url]
        return dict()

    def parsed_content(self):
        """
        :return: Parsed content of the current document
        """
        documents_content = get_documents_content_dict()
        if self.url in documents_content:
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
        doc_snippet_entities = get_documents_snippet_entities()
        if self.url in doc_snippet_entities:
            return doc_snippet_entities[self.url]

        logger.warning("Didn't find cached document snippet entities.")
        # If we didn't find snippet entities in cache, use entity linker.
        if self.snippet_entities is None:
            entity_linker = globals.get_entity_linker()
            self.snippet_entities = dict(
                (entity['name'].lower(), entity)
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

    @staticmethod
    def get_answer_occurrence_by_entity(document_entities, answer_entity):
        """
        Returns a list of positions of occurences of the provided answer_entity in the document.
        :param document_entities: A dictionary (name->entity) of entities mentioned in the document.
        :param answer_entity: The name of the entity to search for.
        :return: Returns a list of positions of occurences of the provided answer_entity in the document.
        The method returns an empty list if the entity wasn't found.
        """
        return document_entities[answer_entity]['positions'] if answer_entity in document_entities else []

    @staticmethod
    def get_answer_occurrence_by_name(document_token2pos, answer_tokens, window_size=3, tokens_fraction=0.7):
        """
        Returns the list of positions of occurences of the answer entity in the document.
        :param document_token2pos: A dictionary from token to positions where it occurs in the document.
        :param answer_tokens: Tokens of the answer entity.
        :return: Returns the list of positions of occurences or an empty list.
        """
        res = []
        answer_token_positions = [(pos, index) for index, token in enumerate(answer_tokens)
                                  for pos in (document_token2pos[token]
                                              if token in document_token2pos else [])]
        answer_token_positions.sort(key=operator.itemgetter(0))
        current_window = deque()
        token_window_counts = [0, ] * len(answer_tokens)
        for pos in answer_token_positions:
            token_window_counts[pos[1]] += 1
            current_window.append(pos)
            while len(current_window) > 0 and current_window[0][0] <= pos[0] - len(answer_tokens) - window_size:
                token_window_counts[current_window[0][1]] -= 1
                current_window.popleft()
            if sum(1 for token_window_count in token_window_counts if token_window_count > 0) > \
                            tokens_fraction * len(answer_tokens):
                res.append((current_window[0][0], current_window[-1][0]))
        return res

    @staticmethod
    def get_qproximity_score(question_tokens_positions, answer_entity_positions, doc_length):
        min_scores = []
        avg_scores = []
        for pos in answer_entity_positions:
            pos = (pos[0] + pos[1]) / 2
            scores = []
            for question_token_positions in question_tokens_positions:
                # Check if the list is empty
                if not question_token_positions:
                    continue
                l = 0
                r = len(question_token_positions)
                while l < r:
                        m = (l + r) / 2
                        if question_token_positions[m] < pos:
                            l = m + 1
                        else:
                            r = m
                diffs = [abs(question_token_positions[i] - pos)
                         for i in xrange(l - 1, l + 1) if 0 <= i < len(question_token_positions)
                                                       and question_token_positions[i] != pos]
                if diffs:
                    scores.append(min(diffs))
            if scores:
                min_scores.append(min(scores))
                avg_scores.append(1.0 * sum(1.0 / score for score in scores if score > 0) / len(scores))
        return min(min_scores) if len(min_scores) > 0 else doc_length, \
               max(avg_scores) if len(avg_scores) > 0 else 0.0

    def generate_description_features(self, answers, question_tokens):
        if not answers:
            return dict()

        description_cosine = []
        is_a_description_cosine = []
        matched_terms = []
        is_a_matched_terms = []
        for answer in answers:
            descriptions = KBEntity.get_entity_descriptions_by_name(answer)
            is_a_description_part = []
            for description in descriptions:
                is_pos = description.find("is")
                if is_pos >= 0:
                    dot_pos = description.find(".", is_pos + 1)
                    if dot_pos >= 0:
                        is_a_description_part.append(description[is_pos + 3: dot_pos])
            description_tokens = map(lambda description:
                                     description.encode('utf-8').translate(
                                         string.maketrans("", ""), string.punctuation).lower().split(),
                                     descriptions)
            is_a_description_part = map(lambda description:
                                        description.encode('utf-8').translate(
                                            string.maketrans("", ""), string.punctuation).lower().split(),
                                        is_a_description_part)
            description_tokens = set(token for description in description_tokens for token in description)
            is_a_description_part_tokens = set(token for description in is_a_description_part for token in description)
            matched_terms.append(len(description_tokens.intersection(question_tokens)))
            is_a_matched_terms.append(len(is_a_description_part_tokens.intersection(question_tokens)))
            description_cosine.append(WebFeatureGenerator.cosine_similarity(description_tokens, question_tokens))
            is_a_description_cosine.append(WebFeatureGenerator.cosine_similarity(
                is_a_description_part_tokens, question_tokens))
        return {"cosine(entity_description, question)": 1.0 * sum(description_cosine) / len(answers) * log(answers),
                "matches(entity_description, question)": 1.0 * sum(matched_terms) / len(answers) * log(answers),
                "cosine(is_a_entity_description_part, question)":
                    1.0 * sum(is_a_description_cosine) / len(answers) * log(answers),
                "matches(is_a_entity_description, question)":
                    1.0 * sum(is_a_matched_terms) / len(answers) * log(answers),
                }

    def generate_features(self, candidate):
        punctuation = set(string.punctuation)
        answers = candidate.get_results_text()
        #logger.info("answers: %s", answers)  #################################

        # Skip empty and extra-long answers.
        if len(answers) == 0 or len(answers) > 10:
            return dict()

        stopwords = globals.get_stopwords()
        answers_lower = map(unicode.lower, answers)
        answers_lower_tokens = [filter(lambda token: token not in stopwords, answer.split())
                                for answer in answers_lower]
        #logger.info("answers_lower_tokens: %s", answers_lower_tokens)  #################################

        question_search_results = get_questions_serps()
        question_text = candidate.query.original_query
        #logger.info("question_text: %s", question_text)  #################################
        #logger.info(get_query_text_tokens(candidate))  ###############################
        if question_text not in question_search_results:
            logger.warning("No search results found for the question % s" % question_text)
            return dict()
        question_tokens = set(filter(lambda token: token not in stopwords and token not in punctuation and
                                                   token != 'STRTS' and token != 'ENTITY',
                                     get_query_text_tokens(candidate, lemmatize=False)))
        #logger.info("question_tokens: %s", question_tokens)  #################################
        question_entities = [entity.entity.name.lower() for entity in candidate.matched_entities]
        #logger.info("question_entities: %s", question_entities)  #################################

        answer_occurrences_text = []
        answer_occurrences_entity = []
        answer_snippet_occurrences_entity = []
        answer_doc_count_text = []
        answer_doc_count_entity = []
        answer_min_qproximity_score_text = []
        answer_min_qproximity_score_entity = []
        answer_avg_qproximity_score_text = []
        answer_avg_qproximity_score_entity = []
        question_entity_in_snippets = []

        for i in xrange(len(answers)):
            answer_occurrences_text.append([])
            answer_occurrences_entity.append([])
            answer_snippet_occurrences_entity.append([])
            answer_doc_count_text.append([])
            answer_doc_count_entity.append([])
            answer_min_qproximity_score_text.append([])
            answer_min_qproximity_score_entity.append([])
            answer_avg_qproximity_score_text.append([])
            answer_avg_qproximity_score_entity.append([])

        for document in question_search_results[question_text][:globals.SEARCH_RESULTS_TOPN]:
            doc_entities = document.mentioned_entities()
            #logger.info("doc_entities: %s", doc_entities)  #################################
            doc_token2pos, doc_lemma2pos = document.get_token_to_positions_map()
            doc_length = [pos for positions in doc_token2pos.itervalues() for pos in positions]
            doc_length = max(doc_length) if doc_length else 0
            #logger.info("doc_token2pos: %s", doc_token2pos)  #################################
            doc_snippet_entities = document.get_snippet_entities()
            #logger.info("doc_snippet_entities: %s", doc_snippet_entities)  #################################
            question_tokens_locations = [doc_token2pos[question_token] if question_token in doc_token2pos else []
                                         for question_token in question_tokens]
            #logger.info("question_tokens_locations: %s", question_tokens_locations)  #################################
            question_tokens_locations.extend([sorted(set(map(lambda position: (position[0] + position[1]) / 2,
                                                             doc_entities[question_entity]['positions'])))
                                              for question_entity in question_entities
                                              if question_entity in doc_entities])
            #logger.info("question_tokens_locations: %s", question_tokens_locations)  #################################

            question_entity_in_snippets.append(
                1.0 * sum(1 if WebFeatureGenerator.get_answer_occurrence_by_entity(doc_snippet_entities,
                                                                                   question_entity)
                          else 0 for question_entity in question_entities) / len(question_entities))

            for answer_index in xrange(len(answers)):
                answer_text_positions = WebFeatureGenerator.get_answer_occurrence_by_name(
                    doc_token2pos, answers_lower_tokens[answer_index])
                # logger.info("answer_text_positions: %s", answer_text_positions)  ###############################
                answer_entity_positions = WebFeatureGenerator.get_answer_occurrence_by_entity(
                    doc_entities, answers_lower[answer_index])
                # logger.info("answer_entity_positions: %s", answer_entity_positions)  ###############################
                answer_entity_snippet_positions = WebFeatureGenerator.get_answer_occurrence_by_entity(
                    doc_snippet_entities, answers_lower[answer_index])
                # logger.info("answer_entity_snippet_positions: %s", answer_entity_snippet_positions)

                qproximity_text = WebFeatureGenerator.get_qproximity_score(question_tokens_locations,
                                                                           answer_text_positions, doc_length)
                # logger.info("qproximity_text: %s", qproximity_text)  #######################################
                qproximity_entity = WebFeatureGenerator.get_qproximity_score(question_tokens_locations,
                                                                             answer_entity_positions, doc_length)
                # logger.info("qproximity_entity: %s", qproximity_entity)  #######################################

                answer_occurrences_text[answer_index].append(len(answer_text_positions))
                answer_occurrences_entity[answer_index].append(len(answer_entity_positions))
                answer_snippet_occurrences_entity[answer_index].append(len(answer_entity_snippet_positions))
                answer_doc_count_text[answer_index].append(1 if len(answer_text_positions) > 0 else 0)
                answer_doc_count_entity[answer_index].append(1 if len(answer_entity_positions) > 0 else 0)
                answer_min_qproximity_score_text[answer_index].append(qproximity_text[0])
                answer_min_qproximity_score_entity[answer_index].append(qproximity_entity[0])
                answer_avg_qproximity_score_text[answer_index].append(qproximity_text[1])
                answer_avg_qproximity_score_entity[answer_index].append(qproximity_entity[1])

        features = dict()
        features.update({"web_search:max_question_entity_in_snippets_doc_count": max(question_entity_in_snippets),
                         "web_search:avg_question_entity_in_snippets_doc_count":
                             1.0 * sum(question_entity_in_snippets) / len(question_entity_in_snippets)})
        features.update(self.get_doc_aggregated_features(answer_occurrences_text,
                                                         "web_search:answer_occurrences_by_text"))
        features.update(self.get_doc_aggregated_features(answer_occurrences_entity,
                                                         "web_search:answer_occurrences_by_entity"))
        features.update(self.get_doc_aggregated_features(answer_snippet_occurrences_entity,
                                                         "web_search:answer_snippet_occurrences_by_entity"))
        features.update(self.get_doc_aggregated_features(answer_doc_count_text,
                                                         "web_search:answer_doc_count_by_text"))
        features.update(self.get_doc_aggregated_features(answer_doc_count_entity,
                                                         "web_search:answer_doc_count_by_entity"))
        features.update(self.get_doc_aggregated_features(answer_min_qproximity_score_text,
                                                         "web_search:answer_min_qproximity_by_text"))
        features.update(self.get_doc_aggregated_features(answer_min_qproximity_score_entity,
                                                         "web_search:answer_min_qproximity_by_entity"))
        features.update(self.get_doc_aggregated_features(answer_min_qproximity_score_text,
                                                         "web_search:answer_avg_qproximity_by_text"))
        features.update(self.get_doc_aggregated_features(answer_min_qproximity_score_entity,
                                                         "web_search:answer_avg_qproximity_by_entity"))
        features.update(self.generate_description_features(answers, question_tokens))
        # logger.info("features: %s", features)  ########################################
        return features

    def get_doc_aggregated_features(self, answer_doc_scores, feature_prefix):
        if not answer_doc_scores:
            return dict()

        mult = log(len(answer_doc_scores) + 1) / len(answer_doc_scores)
        return {
            feature_prefix + "_top1": sum(answer[0] for answer in answer_doc_scores) * mult,
            feature_prefix + "_total": sum(sum(answer) for answer in answer_doc_scores) * mult,
            feature_prefix: sum(1.0 * sum(answer) / len(answer) for answer in answer_doc_scores) * mult,
        }

    def generate_features_old(self, candidate):
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

    @staticmethod
    def cosine_similarity(tokenset1, tokenset2):
        if len(tokenset1) == 0 or len(tokenset2) == 0:
            return 0.0
        return 1.0 * len(tokenset1.intersection(tokenset2)) / sqrt(len(tokenset1)) / sqrt(len(tokenset2))


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s : %(levelname)s '
                               ': %(module)s : %(message)s',
                        level=logging.INFO)
    globals.read_configuration('config.cfg')
    feature_generator = WebFeatureGenerator.init_from_config()
    print feature_generator
