"""
A module for simple query translation.

Copyright 2015, University of Freiburg.

Elmar Haussmann <haussmann@cs.uni-freiburg.de>

"""
import copy
import os
from abc import ABCMeta, abstractmethod
from itertools import product

from answer_type import AnswerTypeIdentifier
from pattern_matcher import QueryCandidateExtender, QueryPatternMatcher, get_content_tokens
import logging
import ranker
import time
import globals
import collections

from query_translator.features import get_n_grams_features
from text2kb.utils import avg

logger = logging.getLogger(__name__)

class Query:
    """
    A query that is to be translated.
    """

    def __init__(self, text):
        self.query_text = text.lower()
        self.original_query = self.query_text
        self.target_type = None
        self.query_tokens = None
        self.query_content_tokens = None
        self.identified_entities = None
        self.relation_oracle = None
        self.is_count_query = False
        self.transform_query(self.query_text)

    def transform_query(self, text):
        """
        For some questions we want to transform the query text
        before processing, e.g. when asking "how many ..." queries.
        """
        how_many = "how many"
        in_how_many = "in how many"
        if text.startswith(how_many):
            self.is_count_query = True
            # Replace "how many" with "what"
            self.query = "what" + text[len(how_many):]
            self.original_query = text
        elif text.startswith(in_how_many):
            self.is_count_query = True
            # Replace "how many" with "what"
            self.query = "in what" + text[len(in_how_many):]
            self.original_query = text


class CandidateGenerator(object):
    """
    Abstract class for all candidate generators.
    """
    __metaclass__ = ABCMeta

    def __init__(self, scorer, **kwargs):
        self.scorer = scorer

    @abstractmethod
    def generate_candidates(self, query_text):
        pass

    def translate_and_execute_query(self, query, n_top=200):
        """
        Generates candidate answers for the given query.
        :param query: Query to generate answer candidates for.
        :param n_top: The number of top candidates to keep. The top is determined using the provided scorer object.
        :return: A list of TranslationResult objects.
        """

        TranslationResult = collections.namedtuple('TranslationResult',
                                                   ['query_candidate',
                                                    'query_results_str'],
                                                   verbose=False)
        start_time = time.time()
        # Parse query.
        results = []
        queries_candidates = self.generate_candidates(query)
        logger.info("Ranking %s query candidates" % len(queries_candidates))
        ranker = self.get_scorer()
        ranked_candidates = ranker.rank_query_candidates(queries_candidates)
        logger.info("Fetching results for all candidates.")
        n_total_results = 0
        if len(ranked_candidates) > n_top:
            logger.info("Truncating returned candidates to %s." % n_top)
        for query_candidate in ranked_candidates[:n_top]:
            query_results_str = query_candidate.get_results_text()
            n_total_results += len(query_results_str)
            result = TranslationResult(query_candidate, query_results_str)
            results.append(result)
        logger.info("Fetched a total of %s results in %.2f ms." % (n_total_results, (time.time() - start_time)))
        logger.info("Done translating and executing: %s." % query)
        return results

    def get_scorer(self):
        return self.scorer

    def set_scorer(self, scorer):
        if scorer is not None:
            self.scorer = scorer

    @classmethod
    def get_from_config(cls, config_params):
        sparql_backend = globals.get_sparql_backend(config_params)
        query_extender = QueryCandidateExtender.init_from_config()
        entity_linker = globals.get_entity_linker()
        parser = globals.get_parser()
        scorer_obj = ranker.SimpleScoreRanker('DefaultScorer')
        ngram_notable_types_npmi_path = config_params.get('QueryCandidateExtender', 'ngram-notable-types-npmi', '')
        notable_types_npmi_threshold = float(config_params.get('QueryCandidateExtender', 'notable-types-npmi-threshold'))
        ngram_notable_types_npmi = None
        if ngram_notable_types_npmi_path and os.path.exists(ngram_notable_types_npmi_path):
            import cPickle as pickle
            try:
                with open(ngram_notable_types_npmi_path, 'rb') as inp:
                    logger.info("Loading types model from disk...")
                    ngram_notable_types_npmi = pickle.load(inp)
            except IOError as exc:
                logger.error("Error reading types model: %s" % str(exc))
                ngram_notable_types_npmi = None

        return SparqlQueryTranslator(sparql_backend, query_extender,
                                     entity_linker, parser, scorer_obj,
                                     ngram_notable_types_npmi,
                                     notable_types_npmi_threshold)


class DummyCandidateGenerator(CandidateGenerator):
    def __init__(self, scorer):
        CandidateGenerator.__init__(self, scorer)

    def generate_candidates(self, query_text):
        return []


class SparqlQueryTranslator(CandidateGenerator):

    def __init__(self, sparql_backend,
                 query_extender,
                 entity_linker,
                 parser,
                 scorer_obj,
                 ngram_notable_types_npmi=None,
                 notable_types_npmi_threshold=0.5):
        CandidateGenerator.__init__(self, scorer_obj)
        self.sparql_backend = sparql_backend
        self.query_extender = query_extender
        self.entity_linker = entity_linker
        self.parser = parser
        self.ngram_notable_types_npmi = ngram_notable_types_npmi
        self.query_extender.set_parameters(scorer_obj.get_parameters())
        self.notable_types_npmi_threshold = notable_types_npmi_threshold

    def set_scorer(self, scorer):
        """Sets the parameters of the translator.

        :type scorer: ranker.BaseRanker
        :return:
        """
        # TODO(Elmar): should this be a parameter of a function call?
        CandidateGenerator.set_scorer(self, scorer)
        self.query_extender.set_parameters(scorer.get_parameters())

    def generate_candidates(self, query_text):
        """
        Perform the actual translation.
        :param query_text:
        :param relation_oracle:
        :param entity_oracle:
        :return:
        """
        num_sparql_queries = self.sparql_backend.num_queries_executed
        sparql_query_time = self.sparql_backend.total_query_time

        # Parse query.
        logger.info("Translating query: %s." % query_text)
        start_time = time.time()
        # Parse the query.
        query = self.parse_and_identify_entities(query_text)
        # Set the relation oracle.
        query.relation_oracle = self.get_scorer().get_parameters().relation_oracle
        # Identify the target type.
        target_identifier = AnswerTypeIdentifier()
        target_identifier.identify_target(query)
        # Get content tokens of the query.
        query.query_content_tokens = get_content_tokens(query.query_tokens)
        # Match the patterns.
        pattern_matcher = QueryPatternMatcher(query,
                                              self.query_extender,
                                              self.sparql_backend)
        ert_matches = pattern_matcher.match_ERT_pattern()
        ermrt_matches = pattern_matcher.match_ERMRT_pattern()
        ermrert_matches = pattern_matcher.match_ERMRERT_pattern()
        duration = (time.time() - start_time) * 1000
        logging.info("Total translation time: %.2f ms." % duration)
        candidates = ert_matches + ermrt_matches + ermrert_matches
        # Extend existing candidates, e.g. by adding answer entity type filters.
        candidates = self.extend_candidates(candidates)

        translation_time = (self.sparql_backend.total_query_time - sparql_query_time) * 1000
        num_sparql_queries = self.sparql_backend.num_queries_executed - num_sparql_queries
        avg_query_time = translation_time / (num_sparql_queries + 0.001)
        logger.info("Translation executed %s queries in %.2f ms."
                    " Average: %.2f ms." % (num_sparql_queries,
                                            translation_time, avg_query_time))
        return candidates

    def extend_candidates(self, candidates):
        """
        Extend the set of candidates with addition query candidates, which can be based on existing candidates.
        For example, one way to extend is to add candidates containing additional filters, e.g. by notable type.
        :param candidates: A set of candidates to extend.
        :return: A new set of candidates.
        """
        extra_candidates = []
        add_types_filters = \
            globals.config.get('QueryCandidateExtender', 'add-notable-types-filter-templates', '') == "True"
        if add_types_filters and self.ngram_notable_types_npmi and candidates:
            for candidate in candidates:
                n_grams = set(get_n_grams_features(candidate))
                notable_types = set(notable_type for notable_type in candidate.get_answer_notable_types()
                                    if notable_type)
                if len(notable_types) > 1:
                    notable_type_scores = dict()
                    for n_gram, notable_type in product(n_grams, notable_types):
                        pair = (n_gram, notable_type)
                        if notable_type not in notable_type_scores:
                            notable_type_scores[notable_type] = []
                        notable_type_scores[notable_type].append((n_gram, self.ngram_notable_types_npmi[pair])
                                                                 if pair in self.ngram_notable_types_npmi
                                                                 else ("", 0.0))

                    for notable_type, ngram_scores in notable_type_scores.iteritems():
                        scores = [score for ngram, score in ngram_scores]
                        max_score = max(scores)
                        if max_score > self.notable_types_npmi_threshold:
                            avg_score = avg(scores)
                            logger.info("Extending candidate with type filter:")
                            logger.info(candidate)
                            logger.info(notable_type)
                            logger.info(ngram_scores)
                            new_query_candidate = copy.deepcopy(candidate)
                            new_query_candidate.filter_answers_by_type(notable_type,
                                                                       [max_score, avg_score])
                            extra_candidates.append(new_query_candidate)
                            logger.info(candidate.get_results_text())
                            logger.info(new_query_candidate.get_results_text())
        return candidates + extra_candidates

    def parse_and_identify_entities(self, query_text):
        """
        Parses the provided text and identifies entities.
        Returns a query object.
        :param query_text:
        :param entity_oracle:
        :return:
        """
        # Parse query.
        parse_result = self.parser.parse(query_text)
        tokens = parse_result.tokens
        # Create a query object.
        query = Query(query_text)
        query.query_tokens = tokens
        if not self.get_scorer().get_parameters().entity_oracle:
            entities = self.entity_linker.identify_entities_in_tokens(
                query.query_tokens, text=query_text)
        else:
            entity_oracle = self.get_scorer().get_parameters().entity_oracle
            entities = entity_oracle.identify_entities_in_tokens(
                query.query_tokens,
                self.entity_linker)
        query.identified_entities = entities
        return query


class WebSearchCandidateGenerator(CandidateGenerator):
    """
    Generates candidate answers by identifying entities mentioned in search results snippets.
    """
    def __init__(self, scorer):
        CandidateGenerator.__init__(self, scorer)

    def translate_and_execute_query(self, query, n_top=200):
        pass


class TranslatorParameters(object):

    """A class that holds parameters for the translator."""
    def __init__(self):
        self.entity_oracle = None
        self.relation_oracle = None
        # When generating candidates, restrict them to the
        # deterimined answer type.
        self.restrict_answer_type = True
        # When matching candidates, require that relations
        # match in some way in the question.
        self.require_relation_match = True
        # Wheather to use web search to extract candidate answers.
        self.web_search_candidates = False


def get_suffix_for_params(parameters):
    """Return a suffix string for the selected parameters.

    :type parameters TranslatorParameters
    :param parameters:
    :return:
    """
    suffix = ""
    if parameters.entity_oracle:
        suffix += "_eo"
    if not parameters.require_relation_match:
        suffix += "_arm"
    if not parameters.restrict_answer_type:
        suffix += "_atm"
    if parameters.web_search_candidates:
        suffix += "_websearch"
    return suffix


if __name__ == '__main__':
    logger.warn("No MAIN")

