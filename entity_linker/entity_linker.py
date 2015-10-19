"""
An approach to identify entities in a query. Uses a custom index for entity information.

Copyright 2015, University of Freiburg.

Elmar Haussmann <haussmann@cs.uni-freiburg.de>
"""
import itertools
import logging
import re
import operator
import time
from surface_index_memory import EntitySurfaceIndexMemory
from util import normalize_entity_name, remove_number_suffix,\
    remove_prefixes_from_name, remove_suffixes_from_name
import globals

logger = logging.getLogger(__name__)


class Entity(object):
    """An entity.

    There are different types of entities inheriting from this class, e.g.,
    knowledge base entities and values.
    """

    def __init__(self, name):
        self.name = name

    def sparql_name(self):
        """Returns an id w/o sparql prefix."""
        pass

    def prefixed_sparql_name(self, prefix):
        """Returns an id with sparql prefix."""
        pass


class KBEntity(Entity):
    """A KB entity."""

    def __init__(self, name, identifier, score, aliases):
        Entity.__init__(self, name)
        # The unique identifier used in the knowledge base.
        self.id = identifier
        # A popularity score.
        self.score = score
        # The entity's aliases.
        self.aliases = aliases

    def sparql_name(self):
        return self.id

    def prefixed_sparql_name(self, prefix):
        return "%s:%s" % (prefix, self.id)

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return self.name == other.name


class Value(Entity):
    """A value.

     Also has a name identical to its value."""

    def __init__(self, name, value):
        Entity.__init__(self, name)
        # The unique identifier used in the knowledge base.
        self.value = value

    def sparql_name(self):
        return self.value

    def prefixed_sparql_name(self, prefix):
        return "%s:%s" % (prefix, self.value)

    def __hash__(self):
        return hash(self.value)

    def __eq__(self, other):
        return self.value == other.value


class DateValue(Value):
    """A date.

    It returns a different sparql name from a value or normal entity.
    """

    def __init__(self, name, date):
        Value.__init__(self, name, date)

    def sparql_name(self):
        return self.value

    def prefixed_sparql_name(self, prefix):
        # Old version uses lowercase t in dateTime
        #return '"%s"^^xsd:dateTime' % self.value
        return '"%s"^^xsd:datetime' % self.value

    def __hash__(self):
        return hash(self.value)

    def __eq__(self, other):
        return self.value == other.value


class IdentifiedEntity():
    """An entity identified in some text."""

    def __init__(self, tokens,
                 name, entity,
                 score=0, surface_score=0,
                 perfect_match=False,
                 external_entity=False,
                 use_as_seed_entity=True,
                 external_entity_count=0):
        # A readable name to be displayed to the user.
        self.name = name
        # The tokens that matched this entity.
        self.tokens = tokens
        # A score for the match of those tokens.
        self.surface_score = surface_score
        # A popularity score of the entity.
        self.score = score
        # The identified entity object.
        self.entity = entity
        # A flag indicating whether the entity perfectly
        # matched the tokens.
        self.perfect_match = perfect_match
        # Position in tokens in the original text where
        # this entity mention was found.
        self.position = None
        # True if this entity was found not in the original text, but
        # from some extension, such as search results.
        self.external_entity = external_entity
        # The number of times this entity was found in the external data,
        # such as search results.
        self.external_entity_count = external_entity_count
        # Whether to allow to use this entity to start building a new candidate,
        # i.e. as the main entity in the question. Other entities can be used
        # as filter entities.
        self.use_as_seed_entity = use_as_seed_entity

    def as_string(self):
        t = u','.join([u"%s" % t.token
                      for t in self.tokens])
        return u"%s: tokens:%s prob:%.3f score:%s perfect_match:%s external:%s external_count:%d" % \
               (self.name, t,
                self.surface_score,
                self.score,
                self.perfect_match,
                "Yes" if self.external_entity else "No",
                self.external_entity_count)

    def overlaps(self, other):
        """Check whether the other identified entity overlaps this one."""
        return set(self.tokens) & set(other.tokens)

    def sparql_name(self):
        return self.entity.sparql_name()

    def prefixed_sparql_name(self, prefix):
        return self.entity.prefixed_sparql_name(prefix)

    def __unicode__(self):
        return self.as_string()

    def __repr__(self):
        return unicode(self).encode('utf-8')


def get_value_for_year(year):
    """Return the correct value representation for a year."""
    # Older Freebase versions do not have the long suffix.
    #return "%s-01-01T00:00:00+01:00" % (year)
    return "%s" % year


class EntityLinker:

    def __init__(self, surface_index,
                 max_entities_per_tokens=4):
        self.surface_index = surface_index
        self.max_entities_per_tokens = max_entities_per_tokens
        # Entities are a mix of nouns, adjectives and numbers and
        # a LOT of other stuff as it turns out:
        # UH, . for: hey arnold!
        # MD for: ben may library
        # PRP for: henry i
        # FW for: ?
        self.valid_entity_tag = re.compile(r'^(UH|\.|TO|PRP.?|#|FW|IN|VB.?|'
                                           r'RB|CC|NNP.?|NN.?|JJ.?|CD|DT|MD|'
                                           r'POS)+$')
        self.ignore_lemmas = {'be', 'of', 'the', 'and', 'or', 'a'}
        self.year_re = re.compile(r'[0-9]{4}')

    def get_entity_for_mid(self, mid):
        '''
        Returns the entity object for the MID or None
         if the MID is unknown. Forwards to surface index.
        :param mid:
        :return:
        '''
        return self.surface_index.get_entity_for_mid(mid)

    @staticmethod
    def init_from_config():
        """
        Return an instance with options parsed by a config parser.
        :param config_options:
        :return:
        """
        config_options = globals.config
        surface_index = EntitySurfaceIndexMemory.init_from_config()
        max_entities_p_token = int(config_options.get('EntityLinker',
                                                      'max-entites-per-tokens'))
        return EntityLinker(surface_index,
                            max_entities_per_tokens=max_entities_p_token)


    def _text_matches_main_name(self, entity, text):

        """
        Check if the entity name is a perfect match on the text.
        :param entity:
        :param text:
        :return:
        """
        text = normalize_entity_name(text)
        text = remove_prefixes_from_name(text)
        name = remove_suffixes_from_name(entity.name)
        name = normalize_entity_name(name)
        name = remove_prefixes_from_name(name)
        if name == text:
            return True
        return False

    def is_entity_occurrence(self, tokens, start, end):
        '''
        Return true if the tokens marked by start and end indices
        are a valid entity occurrence.
        :param tokens:
        :param start:
        :param end:
        :return:
        '''
        # Concatenate POS-tags
        token_list = tokens[start:end]
        pos_list = [t.pos for t in token_list]
        pos_str = ''.join(pos_list)
        # Check if all tokens are in the ignore list.
        if all((t.lemma in self.ignore_lemmas for t in token_list)):
            return False
        # For length 1 only allows nouns
        elif len(pos_list) == 1 and pos_list[0].startswith('N') or \
                                len(pos_list) > 1 and \
                        self.valid_entity_tag.match(pos_str):
            # It is not allowed to split a consecutive NNP
            # if it is a single token.
            if len(pos_list) == 1:
                if pos_list[0].startswith('NNP') and start > 0 \
                        and tokens[start - 1].pos.startswith('NNP'):
                    return False
                elif pos_list[-1].startswith('NNP') and end < len(tokens) \
                        and tokens[end].pos.startswith('NNP'):
                    return False
            return True
        return False

    def identify_dates(self, tokens):
        '''
        Identify entities representing dates in the
        tokens.
        :param tokens:
        :return:
        '''
        # Very simplistic for now.
        identified_dates = []
        for t in tokens:
            if t.pos == 'CD':
                # A simple match for years.
                if re.match(self.year_re, t.token):
                    year = t.token
                    e = DateValue(year, get_value_for_year(year))
                    ie = IdentifiedEntity([t], e.name, e, perfect_match=True)
                    identified_dates.append(ie)
        return identified_dates

    def identify_entities_in_tokens(self, tokens, text='', min_surface_score=0.1,
                                    max_token_window=-1, find_dates=True):
        '''
        Identify instances in the tokens.
        :param tokens: A list of string tokens.
        :param text: The original text, not used in this method, but can be
                     used by subclasses.
        :return: A list of tuples (i, j, e, score) for an identified entity e,
                 at token index i (inclusive) to j (exclusive)
        '''
        n_tokens = len(tokens)
        logger.debug("Starting entity identification.")
        start_time = time.time()
        # First find all candidates.
        identified_entities = []
        for start in range(n_tokens):
            for end in range(start + 1, n_tokens + 1 if max_token_window == -1 else start + max_token_window + 1):
                entity_tokens = tokens[start:end]
                if not self.is_entity_occurrence(tokens, start, end):
                    continue
                entity_str = ' '.join([t.token for t in entity_tokens])
                logger.debug(u"Checking if '{0}' is an entity.".format(entity_str))
                entities = self.surface_index.get_entities_for_surface(entity_str)
                # No suggestions.
                if len(entities) == 0:
                    continue
                for e, surface_score in entities:
                    # Ignore entities with low surface score.
                    if surface_score < min_surface_score:
                        continue
                    perfect_match = False
                    # Check if the main name of the entity exactly matches the text.
                    if self._text_matches_main_name(e, entity_str):
                        perfect_match = True
                    ie = IdentifiedEntity(tokens[start:end],
                                          e.name, e, e.score, surface_score,
                                          perfect_match)
                    ie.position = (start, end)
                    # self.boost_entity_score(ie)
                    identified_entities.append(ie)
        if find_dates:
            identified_entities.extend(self.identify_dates(tokens))
        duration = (time.time() - start_time) * 1000
        identified_entities = self._filter_identical_entities(identified_entities)
        identified_entities = EntityLinker.prune_entities(identified_entities,
                                                          max_threshold=self.max_entities_per_tokens)
        # Sort by quality
        identified_entities = sorted(identified_entities, key=lambda x: (len(x.tokens),
                                                                         x.surface_score),
                                     reverse=True)
        logging.debug("Entity identification took %.2f ms. Identified %s entities." % (duration,
                                                                                      len(identified_entities)))
        return identified_entities

    def identify_entities_in_document(self, document_content_tokens, min_surface_score=0.5, max_token_window=3):
        entities = self.identify_entities_in_tokens(document_content_tokens,
                                                    min_surface_score=min_surface_score,
                                                    max_token_window=max_token_window,
                                                    find_dates=False)
        entities = ((entity.name, entity.surface_score, entity.score,
                     entity.entity.id if isinstance(entity.entity, KBEntity) else entity.name,
                     entity.position) for entity in entities)
        res = []
        for key, values in itertools.groupby(sorted(entities,
                                                    key=operator.itemgetter(3)),
                                             key=operator.itemgetter(3)):
            max_score = 0
            max_surface_score = 0
            name = ""
            count = 0
            positions = []
            for v in values:
                name = v[0]
                max_score = max(max_score, v[2])
                max_surface_score = max(max_surface_score, v[1])
                positions.append(v[4])
                count += 1
            res.append({'mid': key,
                        'name': name,
                        'surface_score': max_surface_score,
                        'score': max_score,
                        'positions': positions,
                        'count': count})
        res.sort(key=operator.itemgetter('count'), reverse=True)
        return res


    def _filter_identical_entities(self, identified_entities):
        '''
        Some entities are identified twice, once with a prefix/suffix
          and once without.
        :param identified_entities:
        :return:
        '''
        entity_map = {}
        filtered_identifications = []
        for e in identified_entities:
            if e.entity not in entity_map:
                entity_map[e.entity] = []
            entity_map[e.entity].append(e)
        for entity, identifications in entity_map.iteritems():
            if len(identifications) > 1:
                # A list of (token_set, score) for each identification.
                token_sets = [(set(i.tokens), i.surface_score)
                              for i in identifications]
                # Remove identification if its tokens
                # are a subset of another identification
                # with higher surface_score
                while identifications:
                    ident = identifications.pop()
                    tokens = set(ident.tokens)
                    score = ident.surface_score
                    if any([tokens.issubset(x) and score < s
                            for (x, s) in token_sets if x != tokens]):
                        continue
                    filtered_identifications.append(ident)
            else:
                filtered_identifications.append(identifications[0])
        return filtered_identifications

    @staticmethod
    def prune_entities(identified_entities, max_threshold=7):
        token_map = {}
        for e in identified_entities:
            tokens = tuple(e.tokens)
            if tokens not in token_map:
                    token_map[tokens] = []
            token_map[tokens].append(e)
        remove_entities = set()
        for tokens, entities in token_map.iteritems():
            if len(entities) > max_threshold:
                sorted_entities = sorted(entities, key=lambda x: x.surface_score, reverse=True)
                # Ignore the entity if it is not in the top candidates, except, when
                # it is a perfect match.
                #for e in sorted_entities[max_threshold:]:
                #    if not e.perfect_match or e.score <= 3:
                #        remove_entities.add(e)
                remove_entities.update(sorted_entities[max_threshold:])
        filtered_entities = [e for e in identified_entities if e not in remove_entities]
        return filtered_entities

    def boost_entity_score(self, entity):
        if entity.perfect_match:
            entity.score *= 60

    @staticmethod
    def create_consistent_identification_sets(identified_entities):
        logger.info("Computing consistent entity identification sets for %s entities." % len(identified_entities))
        # For each identified entity, the ones it overlaps with
        overlapping_sets = []
        for i, e in enumerate(identified_entities):
            overlapping = set()
            for j, other in enumerate(identified_entities):
                if i == j:
                    continue
                if any([t in other.tokens for t in e.tokens]):
                    overlapping.add(j)
            overlapping_sets.append((i, overlapping))
        maximal_sets = []
        logger.info(overlapping_sets)
        EntityLinker.get_maximal_sets(0, set(), overlapping_sets, maximal_sets)
        #logger.info((maximal_sets))
        result = {frozenset(x) for x in maximal_sets}
        consistent_sets = []
        for s in result:
            consistent_set = set()
            for e_index in s:
                consistent_set.add(identified_entities[e_index])
            consistent_sets.append(consistent_set)
        logger.info("Finished computing %s consistent entity identification sets." % len(consistent_sets))
        return consistent_sets

    @staticmethod
    def get_maximal_sets(i, maximal_set, overlapping_sets, maximal_sets):
        #logger.info("i: %s" % i)
        if i == len(overlapping_sets):
            return
        maximal = True
        # Try to extend the maximal set
        for j, (e, overlapping) in enumerate(overlapping_sets[i:]):
            # The two do not overlap.
            if len(overlapping.intersection(maximal_set)) == 0 and not e in maximal_set:
                new_max_set = set(maximal_set)
                new_max_set.add(e)
                EntityLinker.get_maximal_sets(i + 1, new_max_set,
                                               overlapping_sets, maximal_sets)
                maximal = False
        if maximal:
            maximal_sets.append(maximal_set)


class WebSearchResultsExtenderEntityLinker(EntityLinker):
    # How many entities found in search results is allowed to use to build new candidate
    # queries. The rest of the entities can be used to build type-3 queries.
    TOP_ENTITIES_AS_SEEDS = 3

    def __init__(self, surface_index, max_entities_per_tokens=4, use_web_results=True, search_results=None,
                 doc_snippets_entities=None):
        if not doc_snippets_entities:
            doc_snippets_entities = dict()
        if not search_results:
            search_results = dict()
        EntityLinker.__init__(self, surface_index, max_entities_per_tokens)
        self.use_web_results = use_web_results
        if self.use_web_results:
            self.search_results = search_results
            self.doc_snippets_entities = doc_snippets_entities

    @staticmethod
    def init_from_config():
        config_options = globals.config
        surface_index = EntitySurfaceIndexMemory.init_from_config()
        max_entities_p_token = int(config_options.get('EntityLinker',
                                                      'max-entites-per-tokens'))
        use_web_results = config_options.get('EntityLinker',
                                             'use-web-results') == "True"
        question_search_results = dict()
        doc_snippets_entities = dict()
        if use_web_results:
            from text2kb.web_features import _read_serp_files, _read_document_snippet_entities
            serp_files = globals.config.get('WebSearchFeatures', 'serp-files').split(',')
            documents_files = globals.config.get('WebSearchFeatures', 'documents-files').split(',')
            document_snippet_entities_file = globals.config.get('WebSearchFeatures', 'document-snippet-entities')
            question_search_results = _read_serp_files(serp_files, documents_files)
            doc_snippets_entities = _read_document_snippet_entities(document_snippet_entities_file)
        return WebSearchResultsExtenderEntityLinker(surface_index,
                                                    max_entities_p_token,
                                                    use_web_results,
                                                    search_results=question_search_results,
                                                    doc_snippets_entities=doc_snippets_entities)

    def identify_entities_in_tokens(self, tokens, text='', min_surface_score=0.1,
                                    max_token_window=-1, find_dates=True):
        """
        Overrides method that finds entity occurances in the list of tokens and extends the list of tokens with
        entities derived from web search results for the question.
        :param tokens:
        :param text: The text of the original question, which is used to lookup search results and use entities
                     from the snippets to extend the list of entities detected in the question.
        :param min_surface_score: Minimum surface score of entity match.
        :param max_token_window: The maximum width of the token window, -1 considers all possible windows.
        :param find_dates: Whether to find dates.
        :return: A list of detected entities.
        """
        entities = EntityLinker.identify_entities_in_tokens(self, tokens, text=text,
                                                            min_surface_score=min_surface_score,
                                                            max_token_window=max_token_window,
                                                            find_dates=find_dates)
        # Extend with entities found in search results if needed.
        if self.use_web_results:
            entities.extend(self._get_search_results_snippets_entities(text, entities))
        return entities

    def _get_search_results_snippets_entities(self, question, identified_entities):
        """
        Returns a list of entities that are found in snippets of search results when question is issued
        as a query to a search engine.
        :param question: Original question.
        :param identified_entities: Entities already identified from the given question.
        :return: A list of entities found in snippets.
        """
        entities = []
        identified_entity_mids = dict((entity.entity.id, entity) for entity in identified_entities
                                      if isinstance(entity.entity, KBEntity))

        if question not in self.search_results:
            logger.warning("Question '%s' not found in SERPs dictionary!" % question)
            return entities

        for doc in self.search_results[question][:globals.SEARCH_RESULTS_TOPN]:
            if doc.url not in self.doc_snippets_entities:
                logger.warning("Document %s not found in document snippets entities dictionary!" % doc.url)
                continue
            for index, entity in enumerate(self.doc_snippets_entities[doc.url]):
                if entity['mid'] not in identified_entity_mids:
                    kb_entity = KBEntity(entity['name'], entity['mid'], entity['score'], None)
                    perfect_match = self._text_matches_main_name(
                        kb_entity, ' '.join(token.token for token in entity['matches'][0]))
                    is_seed = index < WebSearchResultsExtenderEntityLinker.TOP_ENTITIES_AS_SEEDS
                    # TODO(denxx): At the moment I only take the first match. This might be ok, but need to check.
                    ie = IdentifiedEntity(entity['matches'][0],
                                          kb_entity.name, kb_entity, kb_entity.score,
                                          entity['surface_score'], perfect_match=perfect_match,
                                          external_entity=True,
                                          use_as_seed_entity=is_seed,   # we only
                                          external_entity_count=entity['count'])
                    entities.append(ie)
                else:
                    # Update the external entity count.
                    identified_entity_mids[entity['mid']].external_entity_count += entity['count']
        return entities


if __name__ == '__main__':
    pass
