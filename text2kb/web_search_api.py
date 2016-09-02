import httplib, json, logging, shelve, sys, operator, urllib

import globals
# from entity_linking import find_entity_mentions

logger = logging.getLogger(__name__)

class BingWebSearchApi:
    def __init__(self, api_key, use_search_cache=True):
        self.headers = {
            # Request headers
            'Ocp-Apim-Subscription-Key': api_key,
        }
        if use_search_cache:
            search_cache_file = globals.config.get('WebSearchAnswers', 'websearch-cache')
            self._web_search_cache = shelve.open(search_cache_file)
        else:
            self._web_search_cache = dict()

    def search(self, query, topn=50, offset=0):
        params = urllib.urlencode({
            # Request parameters
            'q': query,
            'count': topn,
            'offset': offset,
            'mkt': 'en-us',
            'safesearch': 'Moderate',
        })

        if params in self._web_search_cache:
            logger.debug("Returning cached search results")
            return self._web_search_cache[params]

        data = None
        try:
            logger.debug("Calling Bing Search API...")
            conn = httplib.HTTPSConnection('api.cognitive.microsoft.com')
            conn.request("GET", "/bing/v5.0/search?%s" % params, "{body}", self.headers)
            response = conn.getresponse()
            data = response.read()
            conn.close()
        except Exception as e:
            print("[Errno {0}] {1}".format(e.errno, e.strerror))
        finally:
            self._web_search_cache[params] = data
            return data

    def close(self):
        if not isinstance(self._web_search_cache, dict):
            self._web_search_cache.close()


class SentSearchApi:
    def __init__(self):
        search_cache_file = globals.config.get('WebSearchAnswers', 'sentsearch-cache')
        self._sent_search_host, self._sent_search_port = globals.config.get('WebSearchAnswers', 'sentsearch-url').split(":")
        self._search_cache = shelve.open(search_cache_file)

    def search(self, query, mids=None, topn=100):
        if query in self._search_cache:
            return self._search_cache[query]

        params = [('phrase', query), ('topn', topn)]
        if mids is not None:
            for mid in mids:
                params.append(("mid", mid))

        params = urllib.urlencode(params)
        data = None
        try:
            conn = httplib.HTTPConnection(self._sent_search_host, int(self._sent_search_port))
            conn.request("GET", "/search?%s" % params)
            response = conn.getresponse()
            data = response.read()
            conn.close()
        except Exception as e:
            print("[Errno {0}] {1}".format(e.errno, e.strerror))
        finally:
            self._search_cache[params] = data
            return data

    def close(self):
        self._search_cache.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--config',
                        default='config.cfg',
                        help='The configuration file to use.')
    args = parser.parse_args()
    globals.read_configuration(args.config)
    searcher = SentSearchApi()

    while True:
        print "Type question: "
        question = sys.stdin.readline()
        if len(question.strip()) == 0:
            break
        question, mids, topn = question.split("\t")
        mids = mids.split()
        print searcher.search(question, mids, topn)


    searcher.close()