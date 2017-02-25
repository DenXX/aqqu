import json, shelve, operator, logging

import globals
from query_translator.answer_candidate import TextSearchAnswerCandidate

from query_translator.evaluation import EvaluationCandidate
from query_translator.ranker import Ranker, MLModel
from text2kb.web_search_api import BingWebSearchApi
from entity_linking import find_entity_mentions

__author__ = 'dsavenk'


logging.basicConfig(format='%(asctime)s : %(levelname)s '
                           ': %(module)s : %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class QuaseWebAnswerer(Ranker):
    """Ranks based on a simple score of relation and entity matches."""

    def __init__(self, name, **kwargs):
        Ranker.__init__(self, name, **kwargs)
        answers_file = globals.config.get('WebSearchAnswers', 'websearch-answers')
        self.answers = dict()

        def object_decoder(q):
            rank = int(q['id'].split("-")[1])
            return q['utterance'], q['result'], rank

        self.answers = dict()
        for question, answer, rank in json.load(open(answers_file, 'r'), object_hook=object_decoder, encoding='utf-8'):
            if question not in self.answers:
                self.answers[question] = []
            self.answers[question].append((answer, rank))

        import operator
        for question, answers in self.answers.iteritems():
            self.answers[question] = [answer for answer, rank in sorted(answers, key=operator.itemgetter(1))]


    def rank_query_candidates(self, query_candidates, key=lambda x: x, utterance=""):
        """
        Returns the candidate generated from search results. This methods doesn't look into the
         existing candidates, but rather creates a new one based on search results.
        :param query_candidates: List of EvaluationCandidate objects. This answerer don't actually use them.
        """

        # If I don't have an answer to the question, return an empty list
        if utterance not in self.answers:
            return []

        return [EvaluationCandidate(None, "", answer) for answer in self.answers[utterance]]


class BingWebAnswerer(Ranker):
    """Ranks based on a simple score of relation and entity matches."""

    def __init__(self, name, entity_link_min_score=0.3, topn=50, use_search_cache=True, use_answers_cache=True, **kwargs):
        Ranker.__init__(self, name, **kwargs)
        answers_cache_file = globals.config.get('WebSearchAnswers', 'websearch-answers-cache')
        self._answers_cache = shelve.open(answers_cache_file) if use_answers_cache else dict()
        self._searcher = BingWebSearchApi(globals.config.get('WebSearchAnswers', 'bing-api-key'), use_search_cache)
        self._topn = topn
        self._entity_linking_score_threshold = entity_link_min_score
        self.parameters.web_search_candidates = True

    def rank_query_candidates(self, query_candidates, key=lambda x: x, utterance=""):
        """
        Returns the candidate generated from search results. This methods doesn't look into the
         existing candidates, but rather creates a new one based on search results.
        :param query_candidates: List of EvaluationCandidate objects. This answerer don't actually use them.
        """
        if isinstance(utterance, unicode):
            utterance = utterance.encode('utf-8')
        if utterance in self._answers_cache:
            return self._answers_cache[utterance]
        logger.debug("-------------------------------------\nQUESTION: " + utterance)
        question_entities = set([e['name'] for e in find_entity_mentions(utterance.encode("utf-8"), use_tagme=True)])
        logger.debug("Question entities: " + str(find_entity_mentions(utterance.encode("utf-8"), use_tagme=True)))
        res = self._searcher.search(utterance, topn=self._topn)
        res = json.loads(res)
        entities = dict()
        for r in res['webPages']['value']:
            title_entities = find_entity_mentions(r['name'].encode("utf-8"), use_tagme=True)
            snippet_entities = find_entity_mentions(r['snippet'].encode("utf-8"), use_tagme=True)
            logger.debug("\nTitle:\t" + r['name'].encode("utf-8") + "\nSnippet:\t" + r['snippet'].encode("utf-8"))
            logger.debug(title_entities)
            logger.debug(snippet_entities)
            for e in title_entities + snippet_entities:
                if e['score'] > self._entity_linking_score_threshold:
                    if e['name'] not in entities:
                        entities[e['name']] = 0
                    entities[e['name']] += e['score']

        answers = sorted(entities.items(), key=operator.itemgetter(1), reverse=True)
        logger.debug("Answer:\t" + str(answers))
        answers = [answer[0] for answer in answers if answer[0] not in question_entities]
        answers = [EvaluationCandidate(None, "", [answer, ]) for answer in answers]
        self._answers_cache[utterance] = answers
        return answers

    def close(self):
        self._searcher.close()
        if not isinstance(self._answers_cache, dict):
            self._answers_cache.close()


class TextAnswerer(MLModel, Ranker):
    """Ranks based on a simple score of relation and entity matches."""

    def __init__(self, name, train_dataset, use_sparql=True, use_web_search=False, use_sent_search=True, **kwargs):
        Ranker.__init__(self, name, **kwargs)
        MLModel.__init__(self, name, train_dataset)
        self.parameters.web_search_candidates = use_web_search
        self.parameters.sent_search_candidates = use_sent_search
        self.parameters.sparql_search_candidates = use_sparql

    def rank_query_candidates(self, query_candidates, key=lambda x: x, utterance=""):
        """
        Returns the candidate generated from search results. This methods doesn't look into the
         existing candidates, but rather creates a new one based on search results.
        :param query_candidates: List of EvaluationCandidate objects. This answerer don't actually use them.
        """
        return sorted(query_candidates,
                      key=lambda x: sum(scoredText.score for scoredText in x.query_candidate.get_source_scored_text())
                      if isinstance(x.query_candidate, TextSearchAnswerCandidate) else 0.0,
                      reverse=True)

    def learn_model(self, train_queries):
        pass

    def store_model(self):
        pass

    def score(self, candidate):
        pass

    def print_model(self):
        pass

    def load_model(self):
        pass


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--config',
                        default='config.cfg',
                        help='The configuration file to use.')
    args = parser.parse_args()
    globals.read_configuration(args.config)

    answerer = BingWebAnswerer("AskMsr+")
    print answerer.rank_query_candidates([], utterance="What animal has human-like fingerprints?")