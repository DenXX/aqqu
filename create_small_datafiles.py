
import globals
import scorer_globals
import cPickle as pickle

from entity_linker.entity_linker import KBEntity
from query_translator import translator
from query_translator.learner import get_evaluated_queries
from text2kb.web_features import get_questions_serps, create_document_vectors_cache, generate_document_vectors
from text2kb.utils import tokenize, get_questions_serps


def filter_documents_content_file(urls_to_keep):
    document_content_file = globals.config.get('WebSearchFeatures', 'documents-content-file')
    with open(document_content_file + '_small', 'w') as out, open(document_content_file, 'r') as content_input:
        # Unpickle until the end of file is reached
        while True:
            try:
                url, content = pickle.load(content_input)
                if url in urls_to_keep:
                    pickle.dump((url, content), out)
            except (EOFError, pickle.UnpicklingError):
                break


def filter_documents_entities(urls):
    document_entities_file = globals.config.get('WebSearchFeatures', 'documents-entities-file')
    with open(document_entities_file) as entities_file:
        url_entities = pickle.load(entities_file)

    filtered_url_entities = {url: url_entities[url] for url in urls if url in url_entities}
    with open(document_entities_file + '_small', 'w') as out:
        pickle.dump(filtered_url_entities, out)


def filter_entity_names(names):
    import gzip
    mids = set()
    entities_file = globals.config.get('EntityLinker', 'entity-names-file')
    with gzip.open(entities_file, 'r') as input_file, gzip.open(entities_file + '_small', 'w') as out:
        for index, line in enumerate(input_file):
            triple = KBEntity.parse_freebase_string_triple(line)
            name = triple[2].lower()
            if name in names:
                mids.add(triple[0])
                print >> out, line.strip()
    return mids


def filter_entity_descriptions(mids):
    import gzip
    descriptions_file = globals.config.get('EntityLinker', 'entity-descriptions-file')
    with gzip.open(descriptions_file, 'r') as input_file, gzip.open(descriptions_file + '_small', 'w') as out:
        for index, line in enumerate(input_file):
            triple = KBEntity.parse_freebase_string_triple(line)
            if triple[0] in mids:
                print >> out, line.strip()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Console based translation.")
    parser.add_argument("--config",
                        default="config.cfg",
                        help="The configuration file to use.")
    args = parser.parse_args()
    globals.read_configuration(args.config)
    scorer_globals.init()

    question_serps = get_questions_serps()

    parameters = translator.TranslatorParameters()
    parameters.require_relation_match = False
    parameters.restrict_answer_type = False
    queries = get_evaluated_queries("webquestions_small_train", True, parameters)
    urls = set()
    entity_mids = set()
    entity_names = set()
    for query in queries:
        entity_names.update(query.target_result)
        for candidate in query.eval_candidates:
            entity_names.update(candidate.prediction)
            for entity in candidate.query_candidate.matched_entities:
                entity_mids.add(entity.entity.entity.id)
                entity_names.add(entity.entity.name)

        # Go through search results.
        question = query.utterance
        for document in question_serps[question][:globals.SEARCH_RESULTS_TOPN]:
            urls.add(document.url)
            #for entity_name in document.mentioned_entities().iterkeys():
                # entity_mids.add(entity['mid'])
            #    entity_names.add(entity_name)

    print("Filtering document content...")
    filter_documents_content_file(urls)

    print("Filtering document entitites...")
    filter_documents_entities(urls)

    print("Printing entity names...")
    entity_names = map(unicode.lower, entity_names)

    mids = filter_entity_names(entity_names)
    filter_entity_descriptions(mids)

    create_document_vectors_cache(question_serps.keys())