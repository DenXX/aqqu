import cPickle as pickle
import gzip
import json
import logging
import string
from math import log, sqrt

import justext
import numpy as np
from gensim import matutils

import globals
from corenlp_parser.parser import Token
from entity_linker.entity_linker import KBEntity
from query_translator.alignment import WordEmbeddings

logger = logging.getLogger(__name__)

# Global caches for search results, document contents and list of entities mentioned in the document.
_question_search_results = None
_documents_content = None
_documents_entities = None
_document_snippet_entities = None
_term_counts = None
_entity_counts = None
_embeddings = None
WEB_DOCUMENTS_COUNT = 10000000000.0
CLUEWEB_DOCUMENTS_COUNT = 1040809705.0 + 733019372


def get_idf_function(entry_type):
    if entry_type == "entity" or entry_type == "entity_tfidf":
        return get_entity_idf
    elif entry_type == "token" or entry_type == "token_tfidf":
        return get_token_idf
    else:
        raise NotImplementedError()


def get_token_idf(token):
    """
    Get the approximation of term IDF based on Google N-Grams dataset.
    :param token: The token to lookup.
    :return: IDF of the term based on the Google N-grams corpus.
    """
    global _term_counts
    if _term_counts is None:
        _term_counts = dict()
        with gzip.open(globals.config.get('WebSearchFeatures', 'term-webcounts-file'), 'r') as input_file:
            logger.info("Reading term web counts...")
            for line in input_file:
                term, count = line.strip().split('\t')
                count = int(count)
                _term_counts[term] = count
            logger.info("Reading term web counts done!")

    if _term_counts:
        token = token.encode('utf-8')
        idf = log(max(1.0, WEB_DOCUMENTS_COUNT / (_term_counts[token]
                                                  if token in _term_counts and _term_counts[token] > 0 else 1.0)))
        return idf
    else:
        return 1.0


def get_entity_idf(entity):
    """
    Get the entity IDF based on Google's annotation of ClueWeb corpus.
    :param entity: The entity to lookup.
    :return: IDF of the entity based on ClueWeb collection.
    """
    global _entity_counts
    if _entity_counts is None:
        _entity_counts = dict()
        with gzip.open(globals.config.get('WebSearchFeatures', 'entity-webcounts-file'), 'r') as input_file:
            logger.info("Reading entity ClueWeb counts...")
            for line in input_file:
                entity, count = line.strip().split('\t')
                count = int(count)
                _entity_counts[entity] = count
            logger.info("Reading entity ClueWeb counts done!")

    if _entity_counts:
        mids = ["/" + mid.replace(".", "/") for mid in KBEntity.get_entityid_by_name(entity, keep_most_triples=True)]
        if mids:
            idf = min(log(max(1.0, CLUEWEB_DOCUMENTS_COUNT / (_entity_counts[mid]
                                                              if mid in _entity_counts and _entity_counts[mid] > 0
                                                              else 1.0)))
                      for mid in mids)
            # logger.info("IDF entity %s %.3f" % (entity, idf))
            return idf
        return 1.0
    else:
        return 0.0


def get_embeddings():
    global _embeddings
    if _embeddings is None:
        embeddings_model = globals.config.get('Alignment', 'word-embeddings')
        _embeddings = WordEmbeddings(embeddings_model)
    return _embeddings


def get_embedding(token):
    return get_embeddings()[token]


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
    global _documents_content
    if _documents_content is None:
        logger.info("Reading documents content...")
        document_content_file = globals.config.get('WebSearchFeatures', 'documents-content-file')

        _documents_content = dict()
        with open(document_content_file, 'r') as content_input:
            # Unpickle until the end of file is reached
            index = 0
            while True:
                try:
                    url, content = pickle.load(content_input)
                    if content:
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
                # denxx: Speed up startup by uncommenting the lines below. Although not all documents will be loaded.
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


def avg(lst):
    return (1.0 * sum(lst) / len(lst)) if len(lst) > 0 else 0.0


class SparseVector:
    def __init__(self, elements_dict):
        self._elems = elements_dict
        self._norm = sqrt(sum(val * val for val in self._elems.itervalues()))

    @staticmethod
    def from_2pos(elem2pos, element_calc_func=lambda elem, positions: 1.0):
        return SparseVector(dict((elem, element_calc_func(elem, positions))
                                 for elem, positions in elem2pos.iteritems()))

    @staticmethod
    def compute_tfidf_entity_elements(elem, positions):
        return log(1 + len(positions)) * get_entity_idf(elem)

    @staticmethod
    def compute_tfidf_token_elements(elem, positions):
        return log(1 + len(positions)) * get_token_idf(elem)


def tokenize(text):
    return text.encode('utf-8').translate(string.maketrans("", ""), string.punctuation).lower().decode('utf-8').split()


def merge_2pos_dicts(dict1, dict2):
    for key, positions in dict2.iteritems():
        if key not in dict1:
            dict1[key] = []
        dict1[key].extend(positions)
    return dict1


class Similarity:
    def __init__(self):
        pass

    @staticmethod
    def cosine_similarity(elem_type, vect1, answer_vector):
        res = 0.0
        if not answer_vector or not vect1:
            return res
        for key in answer_vector._elems.iterkeys():
            if key in vect1._elems:
                value = answer_vector._elems[key]
                res += value * vect1._elems[key]
        if res > 0:
            res /= vect1._norm
            res /= answer_vector._norm
        return res

    @staticmethod
    def normalized_intersection_similarity(elem_type, vect1, answer_vector):
        answer_elements = set(answer_vector._elems.keys())
        if not answer_elements:
            return 0.0
        question_elements = set(vect1._elems.keys())
        return 1.0 * len(answer_elements.intersection(question_elements)) / len(answer_elements)

    @staticmethod
    def intersection_similarity(elem_type, vect1, answer_vector):
        return 1.0 * len(set(answer_vector._elems.keys()).intersection(vect1._elems.keys()))

    @staticmethod
    def bm25_similarity(elem_type, vect1, answer_vector, k1=1.5, b=0.75):
        d = 1.0 * len(vect1._elems)
        score = 0.0
        for elem in answer_vector._elems.iterkeys():
            if elem in vect1._elems:
                tf = vect1._elems[elem]
                denominator = (tf + k1 * (1 - b + b * (d / Similarity.get_average_document_length(elem_type))))
                score += get_idf_function(elem_type)(elem) * tf * (k1 + 1.0) / denominator
        return score

    @staticmethod
    def embedding_avg_idf_similarity(elem_type, vect1, vect2):
        embeddings = get_embeddings()
        answer_token_idfs = dict((token, get_token_idf(token)) for token in vect2._elems.iterkeys()
                                 if token in embeddings.embeddings)
        sum_idf = sum(idf for idf in answer_token_idfs.itervalues())
        if answer_token_idfs:
            avg_answer_idf_embedding = matutils.unitvec(np.array(
                [embeddings[answer_token] * idf / sum_idf
                 for answer_token, idf in answer_token_idfs.iteritems()])
                                                        .sum(axis=0))
        else:
            avg_answer_idf_embedding = np.zeros(embeddings.embeddings.vector_size)

        question_token_idfs = dict((token, get_token_idf(token)) for token in vect1._elems.iterkeys()
                                   if token in embeddings.embeddings)
        sum_idf = sum(idf for idf in question_token_idfs.itervalues())

        if question_token_idfs:
            avg_question_idf_embedding = matutils.unitvec(
                np.array([embeddings[question_token] * idf / sum_idf
                          for question_token, idf in question_token_idfs.iteritems()]).sum(axis=0))
        else:
            avg_question_idf_embedding = np.zeros(embeddings.embeddings.vector_size)
        return np.dot(avg_question_idf_embedding, avg_answer_idf_embedding).item()

    @staticmethod
    def embedding_n_similarity(elem_type, vect1, vect2):
        return get_embeddings().n_similarity(set(vect1._elems.keys()), set(vect2._elems.keys()))

    @staticmethod
    def get_average_document_length(elem_type):
        if elem_type == "token" or elem_type == "token_tfidf":
            return 1000
        elif elem_type == "entity" or elem_type == "entity_tfidf":
            return 10
        else:
            raise NotImplementedError


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
        self.snippet_tokens_to_pos = None
        self.snippet_entities = None

    def mentioned_entities(self):
        """
        :return: A dictionary of mentioned entities (name -> entity).
        """
        documents_entities = get_documents_entities()
        if self.url in documents_entities:
            return documents_entities[self.url]
        return dict()

    def get_mentioned_entities_to_pos(self):
        return dict((entity_name.lower(), map(lambda pos: pos[0], entity['positions']))
                    for entity_name, entity in self.mentioned_entities().iteritems())

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

    @staticmethod
    def _get_tokens2pos_from_tokens(content):
        token_to_pos = dict()
        lemma_to_pos = dict()
        if content:
            for pos, token in enumerate(content):
                lemma = token[1].lower()
                token = token[0].lower()
                if token not in token_to_pos:
                    token_to_pos[token] = []
                if lemma not in lemma_to_pos:
                    lemma_to_pos[lemma] = []
                token_to_pos[token].append(pos)
                lemma_to_pos[lemma].append(pos)
        return token_to_pos, lemma_to_pos

    def get_token_to_positions_map(self):
        """
        :return: Returns a map from token to its position in the document. The map
        is computed the first time the method is called and then the result is cached.
        """
        if self.token_to_pos is None:
            self.token_to_pos, self.lemma_to_pos = WebSearchResult._get_tokens2pos_from_tokens(self.parsed_content())
        return self.token_to_pos, self.lemma_to_pos

    def get_snippet_tokens(self, skip_parsing=True):
        if self.snippet_tokens is None:
            if skip_parsing:
                self.snippet_tokens = [Token(token) for token in tokenize(self.title + u'\n' + self.snippet)]
                for token in self.snippet_tokens:
                    token.lemma = token.token
            else:
                # Skip parsing for performance
                parser = globals.get_parser()
                parse = parser.parse(self.title + u'\n' + self.snippet)
                if parse:
                    self.snippet_tokens = parse.tokens
        return self.snippet_tokens

    def get_snippet_token2pos(self):
        if self.snippet_tokens_to_pos is None:
            self.snippet_tokens_to_pos = WebSearchResult._get_tokens2pos_from_tokens(
                [(token.token.lower(), token.lemma.lower()) for token in self.get_snippet_tokens()])
        return self.snippet_tokens_to_pos

    def get_snippet_entities(self):
        doc_snippet_entities = get_documents_snippet_entities()
        if self.url in doc_snippet_entities:
            return doc_snippet_entities[self.url]

        logger.warning("Didn't find cached document snippet entities.")
        # If we didn't find snippet entities in cache, use entity linker.
        if self.snippet_entities is None:
            self.snippet_entities = {}
            if not doc_snippet_entities:
                entity_linker = globals.get_entity_linker()
                self.snippet_entities = dict(
                    (entity['name'].lower(), entity)
                    for entity in entity_linker.identify_entities_in_document(self.get_snippet_tokens(skip_parsing=False),
                                                                              max_token_window=4,
                                                                              min_surface_score=0.5))
        return self.snippet_entities

    def get_snippet_entities_to_pos(self):
        return dict((entity_name.lower(), entity['positions'][0])
                    for entity_name, entity in self.get_snippet_entities().iteritems())

    def content(self):
        """
        :return: Text content of the given document
        """
        try:
            from os import path
            if self.document_location and path.isfile(self.document_location):
                import codecs
                with open(self.document_location, 'r') as input_document:
                    content = input_document.read()
                    text = justext.justext(content, justext.get_stoplist("English"))
                    res = []
                    # total_length = 0
                    for paragraph in text:
                        if not paragraph.is_boilerplate:
                            res.append(paragraph.text)
                            # total_length += len(paragraph.text)
                        # if total_length > 10000:
                        #     break

                    res = '\n'.join(res)
                    return res
                    # return extract_text(content)
            else:
                logger.warning("Document not found: " + str(self.document_location))
        except Exception as exc:
            logger.warning(exc)
            raise
        return ""

    @staticmethod
    def get_best_fragment_positions(document_token2pos, document_entity2pos, question_token2pos, window_size=100):
        import operator
        from collections import deque

        question_tokens_count = len(question_token2pos.keys())
        question_tokens_list = question_token2pos.keys()
        question_token_positions = [(pos, index) for index, token in enumerate(question_tokens_list)
                                    if token in document_token2pos for pos in document_token2pos[token]]
        question_token_positions.sort(key=operator.itemgetter(0))
        current_window = deque()
        token_window_counts = [0, ] * question_tokens_count

        best_score = 0.0
        best_fragment_token2pos = dict()
        best_fragment_entity2pos = dict()

        for pos in question_token_positions:
            token_window_counts[pos[1]] += 1
            current_window.append(pos)
            while len(current_window) > 0 and current_window[0][0] <= pos[0] - window_size:
                token_window_counts[current_window[0][1]] -= 1
                current_window.popleft()

            fragment_token2pos = dict()
            fragment_entity2pos = dict()
            for token_pos in current_window:
                token = question_tokens_list[token_pos[1]]
                if token not in fragment_token2pos:
                    fragment_token2pos[token] = []
                fragment_token2pos[token].append(token_pos[0])

            for entity, positions in document_entity2pos.iteritems():
                for entity_pos in positions:
                    if current_window[0][0] <= entity_pos <= current_window[0][0] + window_size:
                        if entity not in fragment_entity2pos:
                            fragment_entity2pos[entity] = []
                        fragment_entity2pos[entity].append(entity_pos)

            current_score = Similarity.bm25_similarity("token_tfidf", SparseVector.from_2pos(fragment_token2pos),
                                                       SparseVector.from_2pos(question_token2pos))
            if current_score > best_score:
                best_score = current_score
                best_fragment_token2pos = fragment_token2pos
                best_fragment_entity2pos = fragment_entity2pos

        # logger.info("BEST FRAGMENT %s\t%s\t%.5f" % (str(best_fragment_token2pos.keys()), str(question_token2pos.keys()), best_score))
        return best_fragment_token2pos, best_fragment_entity2pos


# class WebFeatureGenerator:
#     """
#     Generates candidate features based on web search results.
#     """
#
#     def __init__(self):
#         self.question_search_results = None
#
#     @staticmethod
#     def init_from_config():
#         return WebFeatureGenerator()
#
#     @staticmethod
#     def get_answer_occurrence_by_entity(document_entities, answer_entity):
#         """
#         Returns a list of positions of occurences of the provided answer_entity in the document.
#         :param document_entities: A dictionary (name->entity) of entities mentioned in the document.
#         :param answer_entity: The name of the entity to search for.
#         :return: Returns a list of positions of occurences of the provided answer_entity in the document.
#         The method returns an empty list if the entity wasn't found.
#         """
#         return document_entities[answer_entity]['positions'] if answer_entity in document_entities else []
#
#     @staticmethod
#     def get_answer_occurrence_by_name(document_token2pos, answer_tokens, window_size=3, tokens_fraction=0.7):
#         """
#         Returns the list of positions of occurences of the answer entity in the document.
#         :param document_token2pos: A dictionary from token to positions where it occurs in the document.
#         :param answer_tokens: Tokens of the answer entity.
#         :return: Returns the list of positions of occurences or an empty list.
#         """
#         res = []
#         answer_token_positions = [(pos, index) for index, token in enumerate(answer_tokens)
#                                   for pos in (document_token2pos[token]
#                                               if token in document_token2pos else [])]
#         answer_token_positions.sort(key=operator.itemgetter(0))
#         current_window = deque()
#         token_window_counts = [0, ] * len(answer_tokens)
#         for pos in answer_token_positions:
#             token_window_counts[pos[1]] += 1
#             current_window.append(pos)
#             while len(current_window) > 0 and current_window[0][0] <= pos[0] - len(answer_tokens) - window_size:
#                 token_window_counts[current_window[0][1]] -= 1
#                 current_window.popleft()
#             if sum(1 for token_window_count in token_window_counts if token_window_count > 0) > \
#                             tokens_fraction * len(answer_tokens):
#                 res.append((current_window[0][0], current_window[-1][0]))
#         return res
#
#     @staticmethod
#     def get_qproximity_score(question_tokens_positions, answer_entity_positions, doc_length):
#         min_scores = []
#         avg_scores = []
#         for pos in answer_entity_positions:
#             pos = (pos[0] + pos[1]) / 2
#             scores = []
#             for question_token_positions in question_tokens_positions:
#                 # Check if the list is empty
#                 if not question_token_positions:
#                     continue
#                 l = 0
#                 r = len(question_token_positions)
#                 while l < r:
#                         m = (l + r) / 2
#                         if question_token_positions[m] < pos:
#                             l = m + 1
#                         else:
#                             r = m
#                 diffs = [abs(question_token_positions[i] - pos)
#                          for i in xrange(l - 1, l + 1) if 0 <= i < len(question_token_positions)
#                                                        and question_token_positions[i] != pos]
#                 if diffs:
#                     scores.append(min(diffs))
#             if scores:
#                 min_scores.append(min(scores))
#                 avg_scores.append(1.0 * sum(1.0 / score for score in scores if score > 0) / len(scores))
#         return min(min_scores) if len(min_scores) > 0 else doc_length, \
#                max(avg_scores) if len(avg_scores) > 0 else 0.0
#
#     def generate_description_features(self, answers, question_tokens):
#         if not answers:
#             return dict()
#
#         description_cosine = []
#         is_a_description_cosine = []
#         matched_terms = []
#         is_a_matched_terms = []
#         for answer in answers:
#             descriptions = KBEntity.get_entity_descriptions_by_name(answer, keep_most_triples_only=True)
#             is_a_description_part = []
#             for description in descriptions:
#                 is_pos = description.find("is")
#                 if is_pos >= 0:
#                     dot_pos = description.find(".", is_pos + 1)
#                     if dot_pos >= 0:
#                         is_a_description_part.append(description[is_pos + 3: dot_pos])
#             description_tokens = map(lambda description:
#                                      description.encode('utf-8').translate(
#                                          string.maketrans("", ""), string.punctuation).lower().split(),
#                                      descriptions)
#             is_a_description_part = map(lambda description:
#                                         description.encode('utf-8').translate(
#                                             string.maketrans("", ""), string.punctuation).lower().split(),
#                                         is_a_description_part)
#             description_tokens = set(token for description in description_tokens for token in description)
#             is_a_description_part_tokens = set(token for description in is_a_description_part for token in description)
#             matched_terms.append(len(description_tokens.intersection(question_tokens)))
#             is_a_matched_terms.append(len(is_a_description_part_tokens.intersection(question_tokens)))
#             description_cosine.append(WebFeatureGenerator.cosine_similarity(description_tokens, question_tokens))
#             is_a_description_cosine.append(WebFeatureGenerator.cosine_similarity(
#                 is_a_description_part_tokens, question_tokens))
#         return {"cosine(entity_description, question)": 1.0 * sum(description_cosine) / len(answers) * log(len(answers)),
#                 "matches(entity_description, question)": 1.0 * sum(matched_terms) / len(answers) * log(len(answers)),
#                 "cosine(is_a_entity_description_part, question)":
#                     1.0 * sum(is_a_description_cosine) / len(answers) * log(len(answers)),
#                 "matches(is_a_entity_description, question)":
#                     1.0 * sum(is_a_matched_terms) / len(answers) * log(len(answers)),
#                 }
#
#     def generate_features(self, candidate):
#         punctuation = set(string.punctuation)
#         answers = candidate.get_results_text()
#         #logger.info("answers: %s", answers)  #################################
#
#         # Skip empty and extra-long answers.
#         if len(answers) == 0:
#             return dict()
#
#         stopwords = globals.get_stopwords()
#         answers_lower = map(unicode.lower, answers)
#         answers_lower_tokens = [filter(lambda token: token not in stopwords, answer.split())
#                                 for answer in answers_lower]
#         #logger.info("answers_lower_tokens: %s", answers_lower_tokens)  #################################
#
#         question_search_results = get_questions_serps()
#         question_text = candidate.query.original_query
#         #logger.info("question_text: %s", question_text)  #################################
#         #logger.info(get_query_text_tokens(candidate))  ###############################
#         if question_text not in question_search_results:
#             logger.warning("No search results found for the question % s" % question_text)
#             return dict()
#         from query_translator.features import get_query_text_tokens
#         question_tokens = set(filter(lambda token: token not in stopwords and token not in punctuation and
#                                                    token != 'STRTS' and token != 'ENTITY',
#                                      get_query_text_tokens(candidate, lemmatize=False, replace_entity=False)))
#         #logger.info("question_tokens: %s", question_tokens)  #################################
#         question_entities = [entity.entity.name.lower() for entity in candidate.matched_entities]
#         #logger.info("question_entities: %s", question_entities)  #################################
#
#         answer_occurrences_tokens = []
#         answer_occurrences_text = []
#         answer_occurrences_entity = []
#         answer_snippet_occurrences_entity = []
#         answer_doc_count_text = []
#         answer_doc_count_entity = []
#         answer_min_qproximity_score_text = []
#         answer_min_qproximity_score_entity = []
#         answer_avg_qproximity_score_text = []
#         answer_avg_qproximity_score_entity = []
#         question_entity_in_snippets = []
#
#         for i in xrange(len(answers)):
#             answer_occurrences_tokens.append([])
#             answer_occurrences_text.append([])
#             answer_occurrences_entity.append([])
#             answer_snippet_occurrences_entity.append([])
#             answer_doc_count_text.append([])
#             answer_doc_count_entity.append([])
#             answer_min_qproximity_score_text.append([])
#             answer_min_qproximity_score_entity.append([])
#             answer_avg_qproximity_score_text.append([])
#             answer_avg_qproximity_score_entity.append([])
#
#         for document in question_search_results[question_text][:globals.SEARCH_RESULTS_TOPN]:
#             doc_entities = document.mentioned_entities()
#             #logger.info("doc_entities: %s", doc_entities)  #################################
#             doc_token2pos, doc_lemma2pos = document.get_token_to_positions_map()
#             doc_length = [pos for positions in doc_token2pos.itervalues() for pos in positions]
#             doc_length = max(doc_length) if doc_length else 0
#             #logger.info("doc_token2pos: %s", doc_token2pos)  #################################
#             doc_snippet_entities = document.get_snippet_entities()
#             #logger.info("doc_snippet_entities: %s", doc_snippet_entities)  #################################
#             question_tokens_locations = [doc_token2pos[question_token] if question_token in doc_token2pos else []
#                                          for question_token in question_tokens]
#             #logger.info("question_tokens_locations: %s", question_tokens_locations)  #################################
#             question_tokens_locations.extend([sorted(set(map(lambda position: (position[0] + position[1]) / 2,
#                                                              doc_entities[question_entity]['positions'])))
#                                               for question_entity in question_entities
#                                               if question_entity in doc_entities])
#             #logger.info("question_tokens_locations: %s", question_tokens_locations)  #################################
#
#             question_entity_in_snippets.append(
#                 1.0 * sum(1 if WebFeatureGenerator.get_answer_occurrence_by_entity(doc_snippet_entities,
#                                                                                    question_entity)
#                           else 0 for question_entity in question_entities) / len(question_entities))
#
#             for answer_index in xrange(len(answers)):
#                 answer_tokens_in_doc_count = 1.0 * len([token for token in answers_lower_tokens[answer_index]
#                                                         if token in doc_token2pos]) /\
#                                              (len(answers_lower_tokens[answer_index]) + 1)
#
#                 answer_text_positions = WebFeatureGenerator.get_answer_occurrence_by_name(
#                     doc_token2pos, answers_lower_tokens[answer_index])
#                 # logger.info("answer_text_positions: %s", answer_text_positions)  ###############################
#                 answer_entity_positions = WebFeatureGenerator.get_answer_occurrence_by_entity(
#                     doc_entities, answers_lower[answer_index])
#                 # logger.info("answer_entity_positions: %s", answer_entity_positions)  ###############################
#                 answer_entity_snippet_positions = WebFeatureGenerator.get_answer_occurrence_by_entity(
#                     doc_snippet_entities, answers_lower[answer_index])
#                 # logger.info("answer_entity_snippet_positions: %s", answer_entity_snippet_positions)
#
#                 qproximity_text = WebFeatureGenerator.get_qproximity_score(question_tokens_locations,
#                                                                            answer_text_positions, doc_length)
#                 # logger.info("qproximity_text: %s", qproximity_text)  #######################################
#                 qproximity_entity = WebFeatureGenerator.get_qproximity_score(question_tokens_locations,
#                                                                              answer_entity_positions, doc_length)
#                 # logger.info("qproximity_entity: %s", qproximity_entity)  #######################################
#
#                 answer_occurrences_tokens[answer_index].append(answer_tokens_in_doc_count)
#                 answer_occurrences_text[answer_index].append(len(answer_text_positions))
#                 answer_occurrences_entity[answer_index].append(len(answer_entity_positions))
#                 answer_snippet_occurrences_entity[answer_index].append(len(answer_entity_snippet_positions))
#                 answer_doc_count_text[answer_index].append(1 if len(answer_text_positions) > 0 else 0)
#                 answer_doc_count_entity[answer_index].append(1 if len(answer_entity_positions) > 0 else 0)
#                 answer_min_qproximity_score_text[answer_index].append(qproximity_text[0])
#                 answer_min_qproximity_score_entity[answer_index].append(qproximity_entity[0])
#                 answer_avg_qproximity_score_text[answer_index].append(qproximity_text[1])
#                 answer_avg_qproximity_score_entity[answer_index].append(qproximity_entity[1])
#
#         features = dict()
#         features.update({"web_search:max_question_entity_in_snippets_doc_count": max(question_entity_in_snippets),
#                          "web_search:avg_question_entity_in_snippets_doc_count":
#                              1.0 * sum(question_entity_in_snippets) / len(question_entity_in_snippets)})
#         features.update(self.get_doc_aggregated_features(answer_occurrences_tokens,
#                                                          "web_search:answer_occurences_tokens"))
#         features.update(self.get_doc_aggregated_features(answer_occurrences_text,
#                                                          "web_search:answer_occurrences_by_text"))
#         features.update(self.get_doc_aggregated_features(answer_occurrences_entity,
#                                                          "web_search:answer_occurrences_by_entity"))
#         features.update(self.get_doc_aggregated_features(answer_snippet_occurrences_entity,
#                                                          "web_search:answer_snippet_occurrences_by_entity"))
#         features.update(self.get_doc_aggregated_features(answer_doc_count_text,
#                                                          "web_search:answer_doc_count_by_text"))
#         features.update(self.get_doc_aggregated_features(answer_doc_count_entity,
#                                                          "web_search:answer_doc_count_by_entity"))
#         features.update(self.get_doc_aggregated_features(answer_min_qproximity_score_text,
#                                                          "web_search:answer_min_qproximity_by_text"))
#         features.update(self.get_doc_aggregated_features(answer_min_qproximity_score_entity,
#                                                          "web_search:answer_min_qproximity_by_entity"))
#         features.update(self.get_doc_aggregated_features(answer_min_qproximity_score_text,
#                                                          "web_search:answer_avg_qproximity_by_text"))
#         features.update(self.get_doc_aggregated_features(answer_min_qproximity_score_entity,
#                                                          "web_search:answer_avg_qproximity_by_entity"))
#         features.update(self.generate_description_features(answers, question_tokens))
#         # logger.info("features: %s", features)  ########################################
#         return features
#
#     def get_doc_aggregated_features(self, answer_doc_scores, feature_prefix):
#         if not answer_doc_scores:
#             return dict()
#
#         mult = log(len(answer_doc_scores) + 1) / len(answer_doc_scores)
#         return {
#             feature_prefix + "_top1": sum(answer[0] for answer in answer_doc_scores) * mult,
#             feature_prefix + "_total": sum(sum(answer) for answer in answer_doc_scores) * mult,
#             feature_prefix: sum(1.0 * sum(answer) / len(answer) for answer in answer_doc_scores) * mult,
#         }
#
#     @staticmethod
#     def cosine_similarity(tokenset1, tokenset2):
#         if len(tokenset1) == 0 or len(tokenset2) == 0:
#             return 0.0
#         return 1.0 * len(tokenset1.intersection(tokenset2)) / sqrt(len(tokenset1)) / sqrt(len(tokenset2))

if __name__ == "__main__":
    serp = get_questions_serps()
