"""
A module for simple query translation.

Copyright 2015, University of Freiburg.

Elmar Haussmann <haussmann@cs.uni-freiburg.de>

"""
import copy
import os
import sys
from itertools import product

from answer_type import AnswerTypeIdentifier
from pattern_matcher import QueryCandidateExtender, QueryPatternMatcher, get_content_tokens
import logging
import ranker
import time
import globals
import collections

from query_translator.features import FeatureExtractor, get_grams_feats, get_n_grams_features

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



class QueryTranslator(object):

    def __init__(self, sparql_backend,
                 query_extender,
                 entity_linker,
                 parser,
                 scorer_obj,
                 ngram_notable_types_npmi=None):
        self.sparql_backend = sparql_backend
        self.query_extender = query_extender
        self.entity_linker = entity_linker
        self.parser = parser
        self.scorer = scorer_obj
        self.ngram_notable_types_npmi = ngram_notable_types_npmi
        self.query_extender.set_parameters(scorer_obj.get_parameters())

    @staticmethod
    def init_from_config():
        config_params = globals.config
        sparql_backend = globals.get_sparql_backend(config_params)
        query_extender = QueryCandidateExtender.init_from_config()
        entity_linker = globals.get_entity_linker()
        parser = globals.get_parser()
        scorer_obj = ranker.SimpleScoreRanker('DefaultScorer')
        ngram_notable_types_npmi_path = config_params.get('QueryCandidateExtender', 'ngram-notable-types-npmi', '')
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
        return QueryTranslator(sparql_backend, query_extender,
                               entity_linker, parser, scorer_obj, ngram_notable_types_npmi)

    def set_scorer(self, scorer):
        """Sets the parameters of the translator.

        :type scorer: ranker.BaseRanker
        :return:
        """
        # TODO(Elmar): should this be a parameter of a function call?
        self.scorer = scorer
        self.query_extender.set_parameters(scorer.get_parameters())

    def get_scorer(self):
        """Returns the current parameters of the translator.
        """
        return self.scorer

    def translate_query(self, query_text):
        """
        Perform the actual translation.
        :param query_text:
        :param relation_oracle:
        :param entity_oracle:
        :return:
        """
        # Parse query.
        logger.info("Translating query: %s." % query_text)
        start_time = time.time()
        # Parse the query.
        query = self.parse_and_identify_entities(query_text)
        # Set the relation oracle.
        query.relation_oracle = self.scorer.get_parameters().relation_oracle
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
        return candidates

    def extend_candidates(self, candidates):
        """
        Extend the set of candidates with addition query candidates, which can be based on existing candidates.
        For example, one way to extend is to add candidates containing additional filters, e.g. by notable type.
        :param candidates: A set of candidates to extend.
        :return: A new set of candidates.
        """
        extra_candidates = []
        add_types_filters =\
            globals.config.get('QueryCandidateExtender', 'add-notable-types-filter-templates', '') == "True"
        if add_types_filters and self.ngram_notable_types_npmi and candidates:
            for candidate in candidates:
                n_grams = set(get_n_grams_features(candidate))
                notable_types = set(type for types in candidate.get_answer_notable_types() for type in types if type)
                if len(notable_types) > 1:
                    for n_gram, notable_type in product(n_grams, notable_types):
                        pair = (n_gram, notable_type)
                        if pair in self.ngram_notable_types_npmi and \
                            self.ngram_notable_types_npmi[pair] > globals.NPMI_THRESHOLD:
                            logger.info("Extending candidate %s with type filter: %s" % (str(candidate), notable_type))
                            logger.info(pair)
                            logger.info(self.ngram_notable_types_npmi[pair])
                            new_query_candidate = copy.deepcopy(candidate)
                            new_query_candidate.filter_answers_by_type(notable_type,
                                                                       self.ngram_notable_types_npmi[pair])
                            extra_candidates.append(new_query_candidate)
                            logger.info(candidate.get_results_text())
                            logger.info(new_query_candidate.get_results_text())
                            break
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
        if not self.scorer.get_parameters().entity_oracle:
            entities = self.entity_linker.identify_entities_in_tokens(
                query.query_tokens, text=query_text)
        else:
            entity_oracle = self.scorer.get_parameters().entity_oracle
            entities = entity_oracle.identify_entities_in_tokens(
                query.query_tokens,
                self.entity_linker)
        query.identified_entities = entities
        return query

    def translate_and_execute_query(self, query, n_top=200):
        """
        Translates the query and returns a list
        of namedtuples of type TranslationResult.
        :param query:
        :return:
        """
        TranslationResult = collections.namedtuple('TranslationResult',
                                                   ['query_candidate',
                                                    'query_results_str'],
                                                   verbose=False)
        # Parse query.
        results = []
        num_sparql_queries = self.sparql_backend.num_queries_executed
        sparql_query_time = self.sparql_backend.total_query_time
        queries_candidates = self.translate_query(query)
        translation_time = (self.sparql_backend.total_query_time - sparql_query_time) * 1000
        num_sparql_queries = self.sparql_backend.num_queries_executed - num_sparql_queries
        avg_query_time = translation_time / (num_sparql_queries + 0.001)
        logger.info("Translation executed %s queries in %.2f ms."
                    " Average: %.2f ms." % (num_sparql_queries,
                                            translation_time, avg_query_time))
        logger.info("Ranking %s query candidates" % len(queries_candidates))
        ranker = self.scorer
        ranked_candidates = ranker.rank_query_candidates(queries_candidates)
        logger.info("Fetching results for all candidates.")
        sparql_query_time = self.sparql_backend.total_query_time
        n_total_results = 0
        if len(ranked_candidates) > n_top:
            logger.info("Truncating returned candidates to %s." % n_top)
        for query_candidate in ranked_candidates[:n_top]:
            query_results_str = query_candidate.get_results_text()
            n_total_results += len(query_results_str)
            result = TranslationResult(query_candidate, query_results_str)
            results.append(result)
        # This assumes that each query candidate uses the same SPARQL backend
        # instance which should be the case at the moment.
        result_fetch_time = (self.sparql_backend.total_query_time - sparql_query_time) * 1000
        avg_result_fetch_time = result_fetch_time / (len(results) + 0.001)
        logger.info("Fetched a total of %s results in %s queries in %.2f ms."
                    " Avg per query: %.2f ms." % (n_total_results, len(results),
                                                  result_fetch_time, avg_result_fetch_time))
        logger.info("Done translating and executing: %s." % query)
        return results

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
    return suffix


if __name__ == '__main__':
    logger.warn("No MAIN")

