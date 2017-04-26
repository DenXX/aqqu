"""
A module for simple query translation.

Copyright 2015, University of Freiburg.

Elmar Haussmann <haussmann@cs.uni-freiburg.de>

"""
import copy
import json
import os
from abc import ABCMeta, abstractmethod
from itertools import product
from answer_type import AnswerTypeIdentifier
from entity_linker.entity_linker import KBEntity
from pattern_matcher import QueryCandidateExtender, QueryPatternMatcher, get_content_tokens
import globals
import logging
import time
import collections

from query_translator.answer_candidate import TextSearchAnswerCandidate, ScoredText
from query_translator.features import get_n_grams_features
from text2kb.utils import avg
from text2kb.web_search_api import SentSearchApi, BingWebSearchApi

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
        if queries_candidates is None:
            return results
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
    def get_from_config(cls, config_params, scorer_obj):
        sparql_backend = globals.get_sparql_backend(config_params)
        entity_linker = globals.get_entity_linker()
        parser = globals.get_parser()
        candidate_generators = []
        if scorer_obj.parameters.sent_search_candidates:
            logger.info("Creating sentence search candidate answer generator...")
            candidate_generators.append(SentenceSearchCandidateGenerator(scorer_obj, parser, entity_linker))
        if scorer_obj.parameters.web_search_candidates:
            logger.info("Creating web search candidate answer generator...")
            candidate_generators.append(DummyCandidateGenerator(scorer_obj))
        if scorer_obj.parameters.sparql_search_candidates:
            logger.info("Creating SPARQL candidate answer generator...")
            query_extender = QueryCandidateExtender.init_from_config()
            ngram_notable_types_npmi_path = config_params.get('QueryCandidateExtender', 'ngram-notable-types-npmi', '')
            notable_types_npmi_threshold = float(
                config_params.get('QueryCandidateExtender', 'notable-types-npmi-threshold'))
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
            candidate_generators.append(SparqlQueryTranslator(sparql_backend, query_extender,
                                         entity_linker, parser, scorer_obj,
                                         ngram_notable_types_npmi,
                                         notable_types_npmi_threshold))
        return CandidateGeneratorCombiner(scorer_obj, candidate_generators, parser, entity_linker)

    @abstractmethod
    def close(self):
        pass


class EntityBasedCandidateGenerator(CandidateGenerator):
    __metaclass__ = ABCMeta

    def __init__(self, scorer, parser, entity_linker):
        CandidateGenerator.__init__(self, scorer)
        self.parser = parser
        self.entity_linker = entity_linker

    def parse_and_identify_entities(self, query_text):
        """
        Parses the provided text and identifies entities.
        Returns a query object.
        :param query_text:
        :param entity_oracle:
        :return:
        """
        try:
            # Parse query.
            parse_result = self.parser.parse(query_text)
        except:
            return None
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

    def generate_candidates(self, query_text):
        query = self.parse_and_identify_entities(query_text)
        if query is None:
            return None
        # Set the relation oracle.
        query.relation_oracle = self.get_scorer().get_parameters().relation_oracle
        # Identify the target type.
        target_identifier = AnswerTypeIdentifier()
        target_identifier.identify_target(query)
        # Get content tokens of the query.
        query.query_content_tokens = get_content_tokens(query.query_tokens)
        return self.generate_query_candidates(query)

    @abstractmethod
    def generate_query_candidates(self, query):
        pass


class DummyCandidateGenerator(CandidateGenerator):
    def __init__(self, scorer):
        CandidateGenerator.__init__(self, scorer)

    def generate_candidates(self, query_text):
        return []

    def close(self):
        pass


class SparqlQueryTranslator(EntityBasedCandidateGenerator):

    def __init__(self, sparql_backend,
                 query_extender,
                 entity_linker,
                 parser,
                 scorer_obj,
                 ngram_notable_types_npmi=None,
                 notable_types_npmi_threshold=0.5):
        EntityBasedCandidateGenerator.__init__(self, scorer_obj, parser, entity_linker)
        self.sparql_backend = sparql_backend
        self.query_extender = query_extender
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

    def generate_query_candidates(self, query):
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
        logger.info("Translating query: %s." % query.original_query)
        start_time = time.time()
        # Match the patterns.
        pattern_matcher = QueryPatternMatcher(self.query_extender,
                                              self.sparql_backend)
        candidates = pattern_matcher.generate_pattern_candidates(query, query.identified_entities)
        duration = (time.time() - start_time) * 1000
        logging.info("Total translation time: %.2f ms." % duration)
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

    def close(self):
        pass


class SentenceSearchCandidateGenerator(EntityBasedCandidateGenerator):
    """
    Generates candidate answers by identifying entities mentioned in search results snippets.
    """
    def __init__(self, scorer, parser, entity_linker, topn=100):
        EntityBasedCandidateGenerator.__init__(self, scorer, parser, entity_linker)
        self._search = SentSearchApi()
        self._topn = topn

    def generate_query_candidates(self, query):
        candidates = []

        logger.info("Number of identified entities: %d", len(query.identified_entities))
        logger.info("Entities: %s", ", ".join(e.entity.name + "(" + str(e.surface_score) + ")"
                                              for e in sorted(query.identified_entities, key=lambda x: -x.surface_score)
                                              if isinstance(e.entity, KBEntity)))
        search_query = query.original_query
        search_start_time = time.time()
        res = json.loads(self._search.search(search_query, topn=self._topn))
        logger.info("Total search time: %.4f sec.", (time.time() - search_start_time))

        topic_entity_mids = dict()
        for entity in query.identified_entities:
            if isinstance(entity.entity, KBEntity):
                search_mid = entity.entity.id.replace(".", "/")
                if search_mid[0] == "/":
                    search_mid = search_mid[1:]
                topic_entity_mids[search_mid] = entity

        entities = dict()
        for r in res:
            # Consider only sentences, that contain at least one topic entity.
            current_sent_topic_entities = [topic_entity_mids[e['mid']]
                                           for e in r['entities'] if e['mid'] in topic_entity_mids]
            answer_entity = [e for e in r['entities'] if e['mid'] not in topic_entity_mids]
            for e in answer_entity:
                mid = e['mid'].replace("/", ".")
                if mid not in entities:
                    entities[mid] = []
                entities[mid].append((r['phrase'], r['score'], current_sent_topic_entities))
        for mid, phrases in entities.items():
            identified_entities = sorted([entity for _, _, entities in phrases for entity in entities])
            min_end = min([e.indexes[1] for e in identified_entities]) if identified_entities else 0
            for entities in self._get_nonoverlapping_entity_sets(identified_entities, 0, min_end):
                candidates.append(
                    TextSearchAnswerCandidate(query, search_query, entities,
                                              [ScoredText(score, phrase) for phrase, score, _ in phrases],
                                              [(KBEntity.get_entity_name(mid), mid), ]))
        return candidates

    def _get_nonoverlapping_entity_sets(self, identified_entities, min_start_index=0, max_start_index=-1, cur=0, selected=()):
        if cur >= len(identified_entities):
            yield selected
        else:
            e = identified_entities[cur]
            if e.indexes[0] <= min_start_index:
                for r in self._get_nonoverlapping_entity_sets(identified_entities,
                                                              min_start_index=min_start_index,
                                                              max_start_index=max_start_index,
                                                              cur=cur + 1,
                                                              selected=selected):
                    yield r
            else:
                if max_start_index == -1 or e.indexes[0] <= max_start_index:
                    # Include
                    for r in self._get_nonoverlapping_entity_sets(identified_entities,
                                                                  min_start_index=e.indexes[1],
                                                                  max_start_index=-1,
                                                                  cur=cur + 1,
                                                                  selected=selected + (e, )):
                        yield r
                    # Skip
                    for r in self._get_nonoverlapping_entity_sets(identified_entities,
                                                                  min_start_index=e.indexes[0],
                                                                  max_start_index=e.indexes[1],
                                                                  cur=cur + 1,
                                                                  selected=selected):
                        yield r

    def close(self):
        self._search.close()


class WebSearchCandidateGenerator(EntityBasedCandidateGenerator):
    """
    Generates candidate answers by identifying entities mentioned in search results snippets.
    """
    def __init__(self, api_key, entity_threshold, scorer, parser, entity_linker, topn=100):
        EntityBasedCandidateGenerator.__init__(self, scorer, parser, entity_linker)
        self._search = BingWebSearchApi(api_key)
        self._topn = topn
        self._entity_linking_score_threshold = entity_threshold

    def generate_query_candidates(self, query):
        candidates = []

        search_query = query.original_query
        search_start_time = time.time()
        res = json.loads(self._search.search(search_query, topn=self._topn))
        logger.info("Total search time: %.4f sec.", (time.time() - search_start_time))

        entities = dict()
        from entity_linking import find_entity_mentions
        for r in res['webPages']['value']:
            title_entities = find_entity_mentions(r['name'].encode("utf-8"), use_tagme=True)
            snippet_entities = find_entity_mentions(r['snippet'].encode("utf-8"), use_tagme=True)
            logger.debug("\nTitle:\t" + r['name'].encode("utf-8") + "\nSnippet:\t" + r['snippet'].encode("utf-8"))
            logger.debug(title_entities)
            logger.debug(snippet_entities)
            for e in title_entities + snippet_entities:
                if e['score'] > self._entity_linking_score_threshold:
                    if e['name'] not in entities:
                        entities[e['name']] = []
                    entities[e['name']].append(ScoredText(e['score'], r['name'] + "\n" + r['snippet']))

        for name, evidence in entities.items():
            mid = KBEntity.get_entityid_by_name(name, keep_most_triples=True)
            if mid:
                candidates.append(
                    TextSearchAnswerCandidate(query, search_query, [],
                                              evidence,
                                              [(name, mid[0]), ]))
        return candidates

    def close(self):
        self._search.close()


class CandidateGeneratorCombiner(EntityBasedCandidateGenerator):
    """
    Candidate answer generator, that combines the candidates generated by multiple candidates.
    """
    def __init__(self, scorer, candididate_generators, parser, entity_linker):
        EntityBasedCandidateGenerator.__init__(self, scorer, parser, entity_linker)
        assert all(isinstance(x, EntityBasedCandidateGenerator) for x in candididate_generators)
        self.candididate_generators = candididate_generators

    def generate_query_candidates(self, query_text):
        return [candidate for generator in self.candididate_generators
                for candidate in generator.generate_query_candidates(query_text)]

    def close(self):
        for generator in self.candididate_generators:
            generator.close()


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
        # Whether to use KB to generate candidate answers.
        self.sparql_search_candidates = True
        # Whether to use web search to extract candidate answers.
        self.web_search_candidates = False
        # Whether to use entity sentence search to extract candidate answers.
        self.sent_search_candidates = False


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
    if parameters.sent_search_candidates:
        suffix += "_sentsearch"
    return suffix


if __name__ == '__main__':
    logger.warn("No MAIN")

