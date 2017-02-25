"""
Base class to represent answer candidate. QueryCandidate now inherits from AnswerCandidate. This was needed
to support candidates generated from other data sources than KB.

Copyright 2016, Emory University

Denis Savenkov <denis.savenkov@emory.edu>
"""
import logging
import re
from abc import ABCMeta, abstractmethod
from entity_linker.entity_linker import KBEntity

logger = logging.getLogger(__name__)

_year_pattern = re.compile("[0-9]+")


class EntityMatch:
    """
    Describes a match of an entity in the tokens.
    """

    def __init__(self, entity):
        self.entity = entity
        self.score = None

    def __deepcopy__(self, memo):
        # No need to copy the identified entity.
        m = EntityMatch(self.entity)
        m.score = self.score
        return m

    def as_string(self):
        return self.entity.as_string()

    def __repr__(self):
        return self.as_string()


# Note: might be a place for ABC
class RelationMatch:
    """
    Describes a match of a relation.
    Supports iterative construction by first instantiating
    and then adding tokens that matched in different parts
    with different scores.
    """

    def __init__(self, relation):
        # The tokens that matched the relation
        self.tokens = set()
        # The name of the relation that matched
        # In the case of a mediated relation, this can be a tuple
        # (rel_a, rel_b) for the structure "rel_a -> M -> rel_b"
        self.relation = relation
        # The match objects, lazily instantiated
        # when adding matches.
        self.words_match = None
        self.name_match = None
        self.name_weak_match = None
        self.count_match = None
        self.derivation_match = None
        # For target relations we track the cardinality
        # of a relation. This is only set for target relations.
        self.cardinality = -1

    def is_empty(self):
        """
        Returns False if the match matched any tokens,
        else returns True.
        :return:
        """
        return len(self.tokens) == 0

    def add_count_match(self, count):
        """
        Add a count for the relation.
        :param token:
        :param score:
        :return:
        """
        self.count_match = CountMatch(count)

    def add_relation_words_match(self, token, score):
        """
        Add a token and score that matched the relation words.
        :param token:
        :param score:
        :return:
        """
        self.tokens.add(token)
        if not self.words_match:
            self.words_match = WordsMatch([(token, score)])
        else:
            self.words_match.add_match((token, score))

    def add_relation_name_match(self, token, name):
        """
        Add a token and score that matched the relation words.
        :param token:
        :param score:
        :return:
        """
        self.tokens.add(token)
        if not self.name_match:
            self.name_match = NameMatch([(token, name)])
        else:
            self.name_match.add_match((token, name))

    def add_derivation_match(self, token, name):
        """Add a token and and the name it matched via some derivation.
        :param token:
        :param score:
        :return:
        """
        self.tokens.add(token)
        if not self.derivation_match:
            self.derivation_match = DerivationMatch([(token, name)])
        else:
            self.derivation_match.add_match((token, name))

    def add_relation_name_weak_match(self, token, name, score):
        """
        Add a token and score that matched the relation words.
        :param token:
        :param score:
        :return:
        """
        self.tokens.add(token)
        if not self.name_weak_match:
            self.name_weak_match = NameWeakMatch([(token, name, score)])
        else:
            self.name_weak_match.add_match((token, name, score))

    def __deepcopy__(self, memo):
        # No need to copy the matches. They remain unchanged after
        # RelationMatch was created.
        m = RelationMatch(self.relation)
        m.words_match = self.words_match
        m.name_match = self.name_match
        m.name_weak_match = self.name_weak_match
        return m

    def as_string(self):
        result = []
        if self.name_match:
            result.append(self.name_match.as_string())
        if self.derivation_match:
            result.append(self.derivation_match.as_string())
        if self.words_match:
            result.append(self.words_match.as_string())
        if self.name_weak_match:
            result.append(self.name_weak_match.as_string())
        if self.count_match:
            result.append(self.count_match.as_string())
        indent = "\n  "
        s = indent.join(result)
        relation_name = self.relation
        if isinstance(self.relation, tuple):
            relation_name = ' -> '.join(self.relation)
        return "%s:%s%s" % (relation_name, indent, s)


class WordsMatch:
    """
    Describes a match against a list of tokens.
    It has two lists: a list of matching tokens, and a list
    with a score for each token indicating how well the corresponding
    token matched.
    """

    def __init__(self, token_scores=[]):
        # A list of tuples (word, score)
        self.token_scores = token_scores

    def add_match(self, token_score):
        self.token_scores.append(token_score)

    def as_string(self):
        s = ','.join(["%s:%.4f" % (t.lemma, s)
                      for t, s in self.token_scores])
        return "RelationContext: %s" % s


class DerivationMatch:
    """A match against a derived word.

    """

    def __init__(self, token_names):
        # A list of token, name tuples
        # where token matched name via some derivation.
        self.token_names = token_names

    def add_match(self, token_name):
        self.token_names.append(token_name)

    def as_string(self):
        s = ','.join(["%s=%s" % (t.lemma, n)
                      for t, n in self.token_names])
        return "DerivationMatch: %s" % s


class CountMatch:
    """
    Describes a match using only the relation count of freebase.
    """

    def __init__(self, count):
        # A list of tuples (word, score)
        self.count = count

    def as_string(self):
        return "Count: %s" % self.count


class NameMatch:
    """
    Describes a match of tokens in the relation name.
    """

    def __init__(self, token_names):
        # A list of token, name tuples
        # where token matched name.
        self.token_names = token_names

    def add_match(self, token_name):
        self.token_names.append(token_name)

    def as_string(self):
        s = ','.join(["%s=%s" % (t.lemma, n)
                      for t, n in self.token_names])
        return "RelationName: %s" % s


class NameWeakMatch:
    """
    Describes a match of tokens in the relation name with a weak
    synonym.
    """

    def __init__(self, token_name_scores=[]):
        # A list of tuples (token, name, score)
        # where token matched in name with score.
        self.token_name_scores = token_name_scores

    def add_match(self, token_name_score):
        self.token_name_scores.append(token_name_score)

    def as_string(self):
        s = ','.join(["%s=%s:%.2f" % (t.lemma, n, s)
                      for t, n, s in self.token_name_scores])
        return "RelationNameSynonym: %s" % s


class AnswerCandidate:
    __metaclass__ = ABCMeta

    def __init__(self, query):
        self.query = query
        # Result of the given query
        self.query_results = None
        # Mids of the query results.
        self.query_results_mids = None
        # Notable types for the answers
        self.answer_notable_types = None
        # A score computed for this candidate.
        self.rank_score = None
        # An indicator whether the candidate matches the answer type
        self.matches_answer_type = None
        # A set of EntityMatches.
        self.matched_entities = set()
        # A set of RelationMatches.
        self.matched_relations = set()
        # Sets of matched and unmatched tokens so far.
        self.matched_tokens = set()
        self.unmatched_tokens = set(query.query_tokens)

    def get_query(self):
        """
        Returns the query this candidate was generated for.
        :return:
        """
        return self.query

    def get_rank_score(self):
        """
        Returns the current rank score value.
        :return:
        """
        return self.rank_score

    def set_rank_score(self, score):
        """
        Sets the new value for the rank_score.
        :param score:
        :return:
        """
        from query_translator.ranker import RankScore
        from query_translator.ranker import LiteralRankerFeatures
        # For some reason LiteralRanker puts features in score.
        assert isinstance(score, RankScore) or isinstance(score, LiteralRankerFeatures)
        self.rank_score = score

    def get_matched_entities(self):
        """
        Returns the list of entities matched in the query.
        :return: A set of EntityMatch objects.
        """
        return self.matched_entities

    def get_entity_names(self):
        return sorted([me.entity.name for me in self.get_matched_entities()])

    def get_entity_scores(self):
        entities = sorted([me.entity for me in self.get_matched_entities()])
        return [e.score for e in entities]

    def add_entity_match(self, entity_match):
        self.matched_entities.add(entity_match)
        self.matched_tokens.update(entity_match.entity.tokens)
        self.unmatched_tokens = self.unmatched_tokens - set(entity_match.entity.tokens)

    def add_relation_match(self, relation_match, allow_new_match=False):
        """
        Adds a new relation match to the answer candidate.
        :param relation_match:
        :param allow_new_match:
        :return:
        """
        self.matched_relations.add(relation_match)
        self.matched_tokens.update(relation_match.tokens)
        if not allow_new_match:
            self.unmatched_tokens = self.unmatched_tokens - self.matched_tokens

    def get_matched_relations(self):
        """
        Returns matched relations.
        :return: A set of
        """
        return self.matched_relations

    def get_covered_tokens(self):
        """
        Return the set of tokens covered by this candidate.
                :return
        """
        return self.matched_tokens

    @abstractmethod
    def is_match_answer_type(self):
        """
        Returns a boolean of whether the candidate answer matches the type of the question.
        :return:
        """
        pass

    @abstractmethod
    def get_relation_names(self):
        """
        Returns the names of matched relations.
        :return:
        """
        pass

    def get_answer_notable_types(self):
        """
        Returns a list of notable types of each of the result entities.
        :return:
        """
        if self.answer_notable_types is None:
            self.answer_notable_types = []
            for mid, answer in zip(self.get_results_mids(), self.get_results_text()):
                if _year_pattern.match(answer) is not None:
                    continue
                self.answer_notable_types.append(KBEntity.get_notable_type(mid))
        return self.answer_notable_types

    @abstractmethod
    def get_results_mids(self):
        """
        Returns a list with entity ids.
        :return: A list with strings with answer entity ids.
        """
        pass

    @abstractmethod
    def get_results_text(self):
        """
        Returns a list of answer entity names.
        :return: A list of strings with answer entity names.
        """
        pass

    @abstractmethod
    def get_result_count(self):
        """
        Returns the number of entities in the current answer candidate.
        :return: An integer number of entities in the answer.
        """
        pass

    @abstractmethod
    def get_candidate_query(self, **kwargs):
        """
        Returns the candidate query used to generate the current candidate. For example, SPARQL query for
        KB-based candidates.
        :return:
        """
        pass

    @abstractmethod
    def __str__(self):
        pass

class ScoredText:
    def __init__(self, text, score):
        self.text = text
        self.score = score


class TextSearchAnswerCandidate(AnswerCandidate):
    """
    A candidate answer, generated by searching an index over sentences and entity mentions.
    """
    def __init__(self, query, candidate_query, identified_entities, source_scored_texts, answer_entities):
        AnswerCandidate.__init__(self, query)
        assert all(isinstance(source_scored_text, ScoredText) for source_scored_text in source_scored_texts)
        self.source_scored_texts = source_scored_texts
        self.answer = answer_entities
        self.candidate_query = candidate_query
        for entity in identified_entities:
            self.add_entity_match(EntityMatch(entity))

    def get_results_mids(self):
        return [mid for _, mid in self.answer]

    def is_match_answer_type(self):
        return True

    def get_results_text(self):
        return [name for name, _ in self.answer]

    def get_result_count(self):
        return len(self.answer)

    def get_relation_names(self):
        return set()

    def get_source_scored_text(self):
        return self.source_scored_texts

    def get_candidate_query(self, **kwargs):
        return self.candidate_query

    def __str__(self):
        # return "\n".join([("%s %.2f" % (sentence, score)).encode("utf-8") for sentence, score in self.sentences])
        return ", ".join([name.encode("utf-8") for name, _ in self.answer])

if __name__ == "__main__":
    pass