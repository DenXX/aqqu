"""
An approach to identify entities in a query. Uses a custom index for entity information.

Copyright 2015, University of Freiburg.

Elmar Haussmann <haussmann@cs.uni-freiburg.de>
"""
import atexit
import itertools
import logging
import re
import operator
import shelve
import time
from Levenshtein._levenshtein import jaro_winkler

from surface_index_memory import EntitySurfaceIndexMemory
from util import normalize_entity_name, remove_number_suffix, \
    remove_prefixes_from_name, remove_suffixes_from_name
import globals

logger = logging.getLogger(__name__)

STOPWORDS = {"a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't", "as",
             "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can't",
             "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down",
             "during", "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have", "haven't",
             "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself",
             "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's",
             "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off",
             "on", "once", "only", "or", "other", "ought", "our", "ours	ourselves", "out", "over", "own", "same",
             "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such", "than", "that",
             "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they",
             "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up",
             "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's",
             "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with",
             "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself",
             "yourselves"}

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
    _entity_descriptions = None
    _entity_ids = None
    _entity_names = None
    _entity_counts = None
    _notable_types = None

    def __init__(self, name, identifier, score, aliases):
        Entity.__init__(self, name)
        # The unique identifier used in the knowledge base.
        self.id = identifier
        # A popularity score.
        self.score = score
        # The entity's aliases.
        self.aliases = aliases

    def get_description(self):
        """
        Returns the text of the description of the entity.
        :return: The text of the description of the entity.
        """
        KBEntity.get_entity_description(self.id)

    @staticmethod
    def get_entity_description(entity_id):
        if KBEntity._entity_descriptions is None:
            KBEntity._read_descriptions()
        return KBEntity._entity_descriptions[entity_id] \
            if entity_id in KBEntity._entity_descriptions else ""

    @staticmethod
    def get_entityid_by_name(name, keep_most_triples=False):
        if KBEntity._entity_ids is None:
            KBEntity._read_names()
        name = name.lower()
        if isinstance(name, unicode):
            name = name.encode("utf-8")
        if name in KBEntity._entity_ids:
            if keep_most_triples:
                mids = sorted([(mid, KBEntity.get_entity_triples_count(mid)) for mid in KBEntity._entity_ids[name]],
                              key=operator.itemgetter(1), reverse=True)
                if mids:
                    return [mids[0][0], ]
                return []
            else:
                return KBEntity._entity_ids[name]
        else:
            return []

    @staticmethod
    def get_entity_descriptions_by_name(name, keep_most_triples_only=False):
        name = name.lower()
        if isinstance(name, unicode):
            name = name.encode("utf-8")
        if KBEntity._entity_ids is None:
            KBEntity._read_names()
        return filter(lambda x: x, [KBEntity.get_entity_description(entity_id)
                                    for entity_id in KBEntity.get_entityid_by_name(name,
                                                                                   keep_most_triples=keep_most_triples_only)])

    def get_main_name(self):
        return KBEntity.get_entity_name(self.id)

    @staticmethod
    def get_entity_name(mid):
        if KBEntity._entity_names is None:
            KBEntity._read_names()
        if isinstance(mid, unicode):
            mid = mid.encode("utf-8")
        if mid in KBEntity._entity_names:
            return KBEntity._entity_names[mid]
        return ""

    @staticmethod
    def get_notable_type(mid):
        if KBEntity._notable_types is None:
            KBEntity._read_notable_types()
        if mid in KBEntity._notable_types:
            return KBEntity._notable_types[mid]
        return ""

    @staticmethod
    def get_notable_type_by_name(name):
        mid = KBEntity.get_entityid_by_name(name, keep_most_triples=True)
        if mid:
            return KBEntity.get_notable_type(mid[0])
        return ""

    @staticmethod
    def close():
        if KBEntity._entity_ids:
            KBEntity._entity_ids.close()
        if KBEntity._entity_names:
            KBEntity._entity_names.close()

    def sparql_name(self):
        return self.id

    def prefixed_sparql_name(self, prefix):
        return "%s:%s" % (prefix, self.id)

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return self.name == other.name

    @staticmethod
    def _read_descriptions():
        if KBEntity._entity_descriptions is None:
            import globals
            import gzip
            descriptions_file = globals.config.get('EntityLinker', 'entity-descriptions-file')
            logger.info("Reading entity descriptions...")
            with gzip.open(descriptions_file, 'r') as input_file:
                KBEntity._entity_descriptions = dict()
                for index, line in enumerate(input_file):
                    triple = KBEntity.parse_freebase_string_triple(line)
                    if triple is not None:
                        KBEntity._entity_descriptions[triple[0]] = triple[2]
            logger.info("Done reading entity descriptions.")

    @staticmethod
    def _read_names():
        if KBEntity._entity_ids is None:
            import globals
            import gzip
            KBEntity._entity_ids = shelve.open(globals.config.get('EntityLinker', 'entity-names-cache-file'))
            KBEntity._entity_names = shelve.open(globals.config.get('EntityLinker', 'entity-ids-cache-file'))
            if len(KBEntity._entity_ids) == 0:
                names_file = globals.config.get('EntityLinker', 'entity-names-file')
                logger.info("Reading entity names...")
                with gzip.open(names_file, 'r') as input_file:
                    for index, line in enumerate(input_file):
                        triple = KBEntity.parse_freebase_string_triple(line)
                        if triple is not None:
                            name = triple[2]
                            name_lower = name.lower().encode('utf-8')
                            if name_lower not in KBEntity._entity_ids:
                                KBEntity._entity_ids[name_lower] = []
                            KBEntity._entity_ids[name_lower].append(triple[0])
                            KBEntity._entity_names[triple[0].encode("utf-8")] = name
                logger.info("Done reading entity names.")

    @staticmethod
    def _read_entity_counts():
        if KBEntity._entity_counts is None:
            import globals
            import gzip
            counts_file = globals.config.get('EntityLinker', 'entity-counts-file')
            KBEntity._entity_counts = dict()
            logger.info("Reading entity counts...")
            with gzip.open(counts_file, 'r') as input_file:
                for index, line in enumerate(input_file):
                    fields = line.strip().split()
                    mid = fields[1].split('/')[-1][:-1]
                    count = int(fields[0])
                    KBEntity._entity_counts[mid] = count

    @staticmethod
    def _read_notable_types():
        if KBEntity._notable_types is None:
            import globals
            import gzip
            types_file = globals.config.get('EntityLinker', 'notable-types-file')
            KBEntity._notable_types = dict()
            logger.info("Reading notable types...")
            with gzip.open(types_file, 'r') as input_file:
                for index, line in enumerate(input_file):
                    triple = KBEntity.parse_freebase_triple(line)
                    if triple is not None:
                        mid = triple[0]
                        notable_type = triple[2]
                        if mid not in KBEntity._notable_types:
                            KBEntity._notable_types[mid] = notable_type

    @staticmethod
    def parse_freebase_triple(triple_string):
        line = triple_string.decode('utf-8').strip().split('\t')
        if len(line) > 2:
            mid = line[0].split('/')[-1][:-1]
            predicate = line[1]
            obj = line[2].split('/')[-1][:-1]
            return mid, predicate, obj
        return None

    @staticmethod
    def parse_freebase_string_triple(triple_string):
        line = triple_string.decode('utf-8').strip().split('\t')
        if len(line) > 2:
            mid = line[0].split('/')[-1][:-1]
            predicate = line[1]
            obj = line[2]
            pos_left = obj.find("\"")
            pos_right = obj.rfind("\"")
            if pos_left != -1 and pos_right != -1:
                obj = obj[pos_left + 1: pos_right]
            return mid, predicate, obj
        return None

    @staticmethod
    def get_entity_triples_count(mid):
        if KBEntity._entity_counts is None:
            KBEntity._read_entity_counts()

        if mid in KBEntity._entity_counts:
            return KBEntity._entity_counts[mid]
        else:
            return 0


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
        return self.prefixed_sparql_name("")
        # return self.value

    def prefixed_sparql_name(self, prefix):
        # Old version uses lowercase t in dateTime
        # return '"%s"^^xsd:dateTime' % self.value
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

    def is_external_entity(self):
        if hasattr(self, 'external_entity'):
            return self.external_entity
        return False

    def get_external_entity_count(self):
        if hasattr(self, 'external_entity_count'):
            return self.external_entity_count
        return 0

    def can_use_as_seed_entity(self):
        if hasattr(self, 'use_as_seed_entity'):
            return self.use_as_seed_entity
        return True

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
    # return "%s-01-01T00:00:00+01:00" % (year)
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
                entity_occurence = self.is_entity_occurrence(tokens, start, end)
                logger.debug(
                    "TOKENS: " + ' '.join(token.token + "/" + token.pos for token in entity_tokens) + "\t" + str(
                        entity_occurence))
                if not entity_occurence:
                    continue
                entity_str = ' '.join([t.token for t in entity_tokens])
                logger.debug(u"Checking if '{0}' is an entity.".format(entity_str))
                entities = self.surface_index.get_entities_for_surface(entity_str)
                logger.debug(u"Found {0} entities.".format(len(entities)))
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

    def identify_entities_in_document(self, document_content_tokens,
                                      min_surface_score=0.5,
                                      max_token_window=3,
                                      get_main_name=False):
        entities = self.identify_entities_in_tokens(document_content_tokens,
                                                    min_surface_score=min_surface_score,
                                                    max_token_window=max_token_window,
                                                    find_dates=False)
        entities = ((entity.entity.get_main_name() if get_main_name else entity.name, entity.surface_score, entity.score,
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
                # for e in sorted_entities[max_threshold:]:
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
        # logger.info((maximal_sets))
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
        # logger.info("i: %s" % i)
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

    def __init__(self, surface_index, max_entities_per_tokens=4, use_web_results=True, search_results=None,
                 doc_snippets_entities=None, topn_entities=3):
        self.topn_entities = topn_entities
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
        topn_entities = int(config_options.get('EntityLinker',
                                              'topn-external-entities'))
        question_search_results = dict()
        doc_snippets_entities = dict()
        if use_web_results:
            from text2kb.utils import get_documents_snippet_entities
            from text2kb.utils import get_questions_serps
            question_search_results = get_questions_serps()
            doc_snippets_entities = get_documents_snippet_entities()
        return WebSearchResultsExtenderEntityLinker(surface_index,
                                                    max_entities_p_token,
                                                    use_web_results,
                                                    topn_entities=topn_entities,
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
            entities.extend(self._get_search_results_snippets_entities(text, tokens, entities))
        return sorted(entities, key=lambda x: (len(x.tokens), x.surface_score))

    def _get_search_results_snippets_entities(self, question, question_tokens, identified_entities):
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

        search_results_entities = dict()  # mid -> identified entity
        for doc in self.search_results[question][:globals.SEARCH_RESULTS_TOPN]:
            if doc.url not in self.doc_snippets_entities:
                logger.warning("Document %s not found in document snippets entities dictionary!" % doc.url)
                continue

            for snippet_entity in self.doc_snippets_entities[doc.url].itervalues():
                if snippet_entity['mid'] not in identified_entity_mids:
                    if snippet_entity['mid'] in search_results_entities:
                        search_results_entities[snippet_entity['mid']].external_entity_count += snippet_entity['count']
                    else:
                        if keep_entity(snippet_entity['matches'][0], snippet_entity['name'], question_tokens):
                            kb_entity = KBEntity(snippet_entity['name'], snippet_entity['mid'], snippet_entity['score'], None)
                            perfect_match = self._text_matches_main_name(
                                kb_entity, ' '.join(token.token for token in snippet_entity['matches'][0]))
                            ie = IdentifiedEntity(snippet_entity['matches'][0],
                                                  kb_entity.name, kb_entity, kb_entity.score,
                                                  snippet_entity['surface_score'],
                                                  perfect_match=perfect_match,
                                                  external_entity=True,
                                                  use_as_seed_entity=True,  # we only
                                                  external_entity_count=snippet_entity['count'])
                            search_results_entities[snippet_entity['mid']] = ie

                else:
                    # Update the external entity count.
                    identified_entity_mids[snippet_entity['mid']].external_entity_count += snippet_entity['count']

        search_results_entities =\
            sorted(search_results_entities.values(), key=lambda entity: entity.external_entity_count, reverse=True)
        entities.extend(search_results_entities[:self.topn_entities])
        return entities


def keep_entity(matched_tokens, entity_name, question_tokens):
    if "Wikipedia" in matched_tokens:
        return True

    entity_name_tokens = entity_name.lower().split()
    for entity_name_token in entity_name_tokens:
        if entity_name_token in STOPWORDS:
            continue
        for question_token in question_tokens:
            question_token = question_token.token.lower()
            if question_token in STOPWORDS:
                continue
            if get_string_distance(entity_name_token, question_token) > 0.8:
                return True
    return False


def get_string_distance(string1, string2):
    return jaro_winkler(string1, string2)


atexit.register(KBEntity.close)

if __name__ == '__main__':
    entity = KBEntity("Daniil Kharms", u"m.03lp80", 1.0, None)
    print entity.get_description()
