import justext
import cPickle as pickle
import json
import logging
import globals

__author__ = 'dsavenk'

TOPN = 10

logger = logging.getLogger(__name__)

# Global caches for search results, document contents and list of entities mentioned in the document.
question_search_results = None
documents_content = None
documents_entities = None


def _read_serp_files(serp_files, document_files):
    global question_search_results

    logger.info("Reading search results...")
    question_search_results = dict()
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
        question_search_results[question] = question_serp
        for result in search_results['results']['elems']:
            question_serp.append(
                WebSearchResult(result['url'],
                                result['title'],
                                result['snippet'],
                                url2doc[result['url']]))
    logger.info("Reading search results done!")


def _read_document_content(document_content_file):
    logger.info("Reading documents content...")
    global documents_content
    documents_content = dict()
    with open(document_content_file, 'r') as content_input:
        # Unpickle until the end of file is reached
        while True:
            try:
                url, content = pickle.load(content_input)
                tokens = ((token.token.lower(), token.lemma.lower()) for token in content.tokens)
                documents_content[url] = tokens
            except (EOFError, pickle.UnpicklingError):
                break
    logger.info("Reading documents content done!")


def _read_entities(document_entities_file):
    logger.info("Reading documents entities...")
    global documents_entities
    with open(document_entities_file) as entities_file:
        documents_entities = pickle.load(entities_file)
    logger.info("Reading documents entities done!")


def entity_names_equal(entity1, entity2):
    return entity1.lower() == entity2.lower()


def answer_contains(answer_tokens, doc_tokens_set):
    return sum(1.0 for answer_token in answer_tokens if answer_token in doc_tokens_set) / len(answer_tokens)


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

    def mentioned_entities(self):
        """
        :return: A list of mentioned entities.
        """
        global documents_entities
        if documents_entities and self.url in documents_entities:
            return documents_entities[self.url]
        return None

    def parsed_content(self):
        """
        :return: Parsed content of the current document
        """
        global documents_content
        if documents_content and self.url in documents_content:
            return documents_content[self.url]
        # TODO(denxx): Here we need to parse, but to parse we need parser. Should make a global parser.
        return None

    def content(self):
        """
        :return: Text content of the given document
        """
        try:
            if os.path.isfile(self.document_location):
                import codecs
                with codecs.open(self.document_location, 'r', 'utf-8') as input_document:
                    content = input_document.read()
                    text = justext.justext(content, justext.get_stoplist("English"))
                    res = []
                    for paragraph in text:
                        if not paragraph.is_boilerplate:
                            res.append(paragraph.text)
                    return '\n'.join(res)
                    # return extract_text(content)
            else:
                logger.warning("Document not found: " + str(self.document_location))
        except Exception as exc:
            logger.warning(exc)
        return ""


class WebFeatureGenerator:
    """
    Generates candidate features based on web search results.
    """

    def __init__(self, serp_files, documents_files, entities_file, content_file):
        # Reading data if needed
        if not question_search_results:
            _read_serp_files(serp_files, documents_files)
        if not documents_entities:
            _read_entities(entities_file)
        if not documents_content:
            _read_document_content(content_file)

    @staticmethod
    def init_from_config():
        config_options = globals.config
        serp_files = config_options.get('WebSearchFeatures', 'serp-files').split(',')
        documents_files = config_options.get('WebSearchFeatures', 'documents-files').split(',')
        entities_file = config_options.get('WebSearchFeatures', 'documents-entities-file')
        content_file = config_options.get('WebSearchFeatures', 'documents-content-file')
        return WebFeatureGenerator(serp_files, documents_files, entities_file, content_file)

    def compute_features(self, candidate):
        question = candidate.query.original_query
        answers = candidate.query_results

        # If we don't have anything for the question...
        if question not in question_search_results:
            logger.warning("No documents found for question: " + question)
            return dict()

        answer_occurances_entity = [0, ] * len(answers)
        answer_doc_occurances_entity = [0, ] * len(answers)
        answer_doc_occurances_text = [0, ] * len(answers)
        answer_neighbourhood = []
        for i in xrange(len(answers)):
            answer_neighbourhood.append(set())

        for rank, doc in enumerate(question_search_results[question][:TOPN]):
            doc_content = doc.parsed_content()
            doc_entities = doc.mentioned_entities()
            seen_answers = [False, ] * len(answers)
            if doc_entities:
                for entity in doc_entities:
                    for i, answer in enumerate(answers):
                        if entity_names_equal(entity['name'], answer):
                            answer_occurances_entity[i] += entity['count']
                            seen_answers[i] = True
            for i, seen in seen_answers:
                if seen:
                    answer_doc_occurances_entity[i] += 1

            if doc_content:
                doc_tokens = set(token[0] for token in doc_content)
                # token_to_pos = {(token, pos) for pos, token in enumerate(doc_content)}
                for i, answer in enumerate(answers):
                    answer_tokens = [token.lower() for token in answer.split()]
                    if answer_contains(answer_tokens, doc_tokens):
                        answer_doc_occurances_text[i] += 1

                    # answer_positions = set(
                    #     token_to_pos[answer_token] for answer_token in answer_tokens if answer_token in token_to_pos)
                    # for pos in answer_positions:
                    #     for around in xrange(max(0, pos - 3), min(len(doc_content), pos + 4)):
                    #         answer_neighbourhood[i] += doc_content[around][0]



        return {
            'web_results:answer_entity_occurances': 1.0 * sum(answer_occurances_entity) / len(answers),
            'web_results:answer_entity_doc_occurances': 1.0 * sum(answer_doc_occurances_entity) / len(answers),
            'web_results:answer_text_doc_occurances': 1.0 * sum(answer_doc_occurances_text) / len(answers),
        }


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s : %(levelname)s '
                               ': %(module)s : %(message)s',
                        level=logging.INFO)
    globals.read_configuration('config.cfg')
    feature_generator = WebFeatureGenerator.init_from_config()
    print feature_generator
