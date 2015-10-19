"""
Copyright 2015, University of Freiburg.

Elmar Haussmann <haussmann@cs.uni-freiburg.de>
"""
import codecs
from ConfigParser import SafeConfigParser
from itertools import tee, izip
import logging

logger = logging.getLogger(__name__)

FREEBASE_NS_PREFIX = "http://rdf.freebase.com/ns/"
FREEBASE_SPARQL_PREFIX = "fb"
FREEBASE_NAME_RELATION = "type.object.name"
FREEBASE_KEY_PREFIX = "http://rdf.freebase.com/key/"

SEARCH_RESULTS_TOPN = 10

sparql_backend = None
# When a configuration is read, store it here to make it accessible.
config = None

# A set of stopwords.
_stopwords = None

_entity_linker = None
_parser = None

# TODO(elmar): move this somewhere else.
def get_sparql_backend(config_options):
    global sparql_backend
    from sparql_backend.backend import  SPARQLHTTPBackend
    if not sparql_backend:
        sparql_backend = SPARQLHTTPBackend.init_from_config(config_options)
    return sparql_backend


def get_stopwords():
    """
    :return: Returns a set of stopwords.
    """
    global _stopwords
    if _stopwords is not None:
        return _stopwords
    _stopwords = set()
    stopwords_file = config.get('WebSearchFeatures', 'stopwords-file')
    with codecs.open(stopwords_file, 'r', 'utf-8') as stopwords_file_input:
        for line in stopwords_file_input:
            _stopwords.add(line.strip())
    return _stopwords


def get_entity_linker():
    global _entity_linker
    if _entity_linker is not None:
        return _entity_linker
    from entity_linker.entity_linker import EntityLinker, WebSearchResultsExtenderEntityLinker
    _entity_linker = WebSearchResultsExtenderEntityLinker.init_from_config()
    return _entity_linker


def get_parser():
    global _parser
    if _parser is not None:
        return _parser
    from corenlp_parser.parser import CoreNLPParser
    _parser = CoreNLPParser.init_from_config()
    return _parser


def get_prefixed_qualified_mid(mid, prefix):
    return "%s:%s" % (prefix, mid)


def read_configuration(configfile):
    """Read configuration and set variables.

    :return:
    """
    global config
    logger.info("Reading configuration from: " + configfile)
    parser = SafeConfigParser()
    parser.read(configfile)
    config = parser
    return parser

def get_qualified_mid(mid):
    '''
    Returns a fully qualified MID, with NS
    prefix and brackets.
    :param mid:
    :return:
    '''
    return "<%s%s>" % (FREEBASE_NS_PREFIX, mid)

def get_mid_from_qualified_string(qualified_str):
    '''
    Returns a fully qualified MID, with NS
    prefix and brackets.
    :param mid:
    :return:
    '''
    if qualified_str.startswith('<') and qualified_str.endswith('>'):
        qualified_str = qualified_str[1:-1]
    return remove_freebase_ns(qualified_str)

def remove_freebase_ns(mid):
    '''
    Returns a fully qualified MID, with NS
    prefix and brackets.
    :param mid:
    :return:
    '''
    if mid.startswith(FREEBASE_NS_PREFIX):
        return mid[len(FREEBASE_NS_PREFIX):]
    return mid

def pairwise(iterable):
    a, b = tee(iterable)
    next(b, None)
    return izip(a, b)


def get_freebase_easy_name_for_freebase_relation_suffix(fb_suffix):
    '''
    Return the Freebase easy name for a freebase relation suffix,
    e.g. people.person.born_in -> people/person/born_in
    :param fb_suffix:
    :return:
    '''
    fb_suffix = fb_suffix.replace('.', '/')
    return fb_suffix
