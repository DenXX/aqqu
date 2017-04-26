import cPickle as pickle
import logging

import globals
from entity_linker.entity_linker import KBEntity
from text2kb.utils import tokenize, merge_2pos_dicts, Similarity, avg, get_questions_serps, SparseVector, \
    WebSearchResult

__author__ = 'dsavenk'

logger = logging.getLogger(__name__)

_documents_vectors_cache = dict()


def generate_text_based_features(candidate):
    # Get candidate answers
    answers = map(unicode.lower, candidate.get_results_text())
    # Skip empty and extra-long answers.
    if len(answers) == 0:
        return dict()
    # Get answers descriptions.
    answers_descriptions = ['\n'.join(KBEntity.get_entity_descriptions_by_name(answer, keep_most_triples_only=True))
                            for answer in answers]

    # Get question text.
    question_text = candidate.query.original_query
    question_tokens2pos = dict((token, [1, ]) for token in tokenize(question_text))
    question_token_tfidf = SparseVector.from_2pos(question_tokens2pos,
                                                  element_calc_func=SparseVector.compute_tfidf_token_elements)

    # Get question entities
    question_entities2pos = dict((entity.entity.name.lower(), [1, ]) for entity in candidate.matched_entities)
    question_entitytoken2pos = dict((token, [1, ])
                                    for entity in candidate.matched_entities
                                    for token in tokenize(entity.entity.name))
    question_entity_tfidf = SparseVector.from_2pos(question_entitytoken2pos,
                                                   element_calc_func=SparseVector.compute_tfidf_token_elements)

    # Get search results and check that they aren't empty
    questions_search_results = get_questions_serps()

    documents_vectors = []
    snippets_vectors = []
    fragment_vectors = []
    combined_documents_vector = dict()
    combined_document_snippets_vector = dict()

    representations = ["entity_tfidf",
                       "token_tfidf",
                       # "entity",
                       # "token",
                      ]
    for r in representations:
        combined_documents_vector[r] = dict()
        combined_document_snippets_vector[r] = dict()

    if question_text not in questions_search_results:
        logger.warning("No search results found for the question %s" % question_text)
    else:
        documents_vectors, snippets_vectors, fragment_vectors, combined_documents_vector,\
            combined_document_snippets_vector = generate_document_vectors(question_text,
                                                                          question_tokens2pos,
                                                                          questions_search_results)

    answer_entity2pos = dict((answer_entity, [1, ]) for answer_entity in answers)
    answer_token2pos = dict((answer_token, [1, ]) for answer_entity in answers
                            for answer_token in tokenize(answer_entity))
    answers_vectors = {
        "token_tfidf": SparseVector.from_2pos(answer_token2pos,
                                              element_calc_func=SparseVector.compute_tfidf_token_elements),
        "entity_tfidf": SparseVector.from_2pos(answer_entity2pos,
                                               element_calc_func=SparseVector.compute_tfidf_entity_elements),
        # "entity": SparseVector.from_2pos(answer_entity2pos),
        # "token": SparseVector.from_2pos(answer_token2pos),
    }

    answer_descriptions_token2pos = dict((token, [1, ]) for description in answers_descriptions
                                         for token in tokenize(description))
    answer_description_vectors = {
        "token_tfidf": SparseVector.from_2pos(answer_descriptions_token2pos,
                                              element_calc_func=SparseVector.compute_tfidf_token_elements),
        "entity_tfidf": SparseVector(dict()),
        # Keeping only tf-idf similarities. This seems to be enough.
        # "token": SparseVector.from_2pos(answer_descriptions_token2pos),
        # "entity": SparseVector(dict()),
    }

    similarity_functions = [
        ("cosine", Similarity.cosine_similarity),
        # ("itersection", Similarity.intersection_similarity),
        # ("normalized_intersection", Similarity.normalized_intersection_similarity),
        # ("bm25", Similarity.bm25_similarity),
    ]
    features = dict()

    for similarity_name, similarity in similarity_functions:
        # Computing document-answer similarities for each representation.
        document_answer_similarities = {}
        for representation in representations:
            if representation not in document_answer_similarities:
                document_answer_similarities[representation] = []
            for doc_vector in documents_vectors:
                document_answer_similarities[representation].append(similarity(representation,
                                                                               doc_vector[representation],
                                                                               answers_vectors[representation]))
        for representation in representations:
            features.update({
                "text_features:avg_document_answer_%s_%s" % (representation, similarity_name):
                    avg(document_answer_similarities[representation]),
                "text_features:max_document_answer_%s_%s" % (representation, similarity_name):
                    max(document_answer_similarities[representation]) if document_answer_similarities[representation]
                    else 0.0,
            })

        # logger.info("Snippet-answer similarity...")
        # Computing snippet-answer similarities for each representation.
        snippet_answer_similarities = {}
        for representation in representations:
            if representation not in snippet_answer_similarities:
                snippet_answer_similarities[representation] = []

            for snippet_vector in snippets_vectors:
                snippet_answer_similarities[representation].append(similarity(representation,
                                                                              snippet_vector[representation],
                                                                              answers_vectors[representation]))

        for representation in representations:
            features.update({
                "text_features:avg_snippet_answer_%s_%s" % (representation, similarity_name):
                    avg(snippet_answer_similarities[representation]),
                "text_features:max_snippet_answer_%s_%s" % (representation, similarity_name):
                    max(snippet_answer_similarities[representation]) if snippet_answer_similarities[representation] else 0.0,
            })

        # logger.info("Fragment-answer similarity...")
        # Best BM25 fragment-answer similarities.
        # Weren't very efficient and therefore I remove this features. There is a chance that there is a bug in the features.

        # fragment_answer_similarities = {}
        # for fragment_vector in fragment_vectors:
        #     for representation in representations:
        #         if representation not in fragment_answer_similarities:
        #             fragment_answer_similarities[representation] = []
        #         fragment_answer_similarities[representation].append(similarity(representation,
        #                                                                        fragment_vector[representation],
        #                                                                        answers_vectors[representation]))
        #
        # for representation in representations:
        #     features.update({
        #         "text_features:avg_fragment_answer_%s_%s" % (representation, similarity_name):
        #             avg(fragment_answer_similarities[representation]),
        #         "text_features:max_fragment_answer_%s_%s" % (representation, similarity_name):
        #             max(fragment_answer_similarities[representation]) if fragment_answer_similarities[representation] else 0.0,
        #     })

        # logger.info("Combined document-answer similarity...")
        # Combined documents answer similarity
        for representation in representations:
            combineddoc_answer_similarity = similarity(representation,
                                                       combined_documents_vector[representation],
                                                       answers_vectors[representation])
            features.update({
                "text_features:combdocument_answer_%s_%s" % (representation, similarity_name):
                    combineddoc_answer_similarity,
            })

        # logger.info("Combined snippet-answer similarity...")
        for representation in representations:
            combineddocsnippet_answer_similarity = similarity(representation,
                                                              combined_document_snippets_vector[representation],
                                                              answers_vectors[representation])
            features.update({
                "text_features:combdocument_snippet_answer_%s_%s" % (representation, similarity_name):
                    combineddocsnippet_answer_similarity,
            })

        # logger.info("Description-question similarity...")
        # These features aren't very efficient either. The next candidate for removal.
        description_question_entity_similarity = similarity("token_tfidf", question_entity_tfidf,
                                                            answer_description_vectors["token_tfidf"])
        description_question_token_similarity = similarity("token_tfidf", question_token_tfidf,
                                                           answer_description_vectors["token_tfidf"])
        features.update({
            "text_features:description_question_entitytoken_%s" % similarity_name:
                description_question_entity_similarity,
            "text_features:description_question_token_%s" % similarity_name: description_question_token_similarity,
        })

    # Description - question embedding similarity.
    description_question_token_embedding_avg_similarity = Similarity.embedding_avg_idf_similarity(
        "token_tfidf", question_token_tfidf, answer_description_vectors["token_tfidf"])
    description_question_token_embedding_n_similarity = Similarity.embedding_avg_idf_similarity(
        "token_tfidf", question_token_tfidf, answer_description_vectors["token_tfidf"])
    features.update({
        "text_features:description_question_token_avg_idf_embeddings":
            description_question_token_embedding_avg_similarity,
        "text_features:description_question_token_n_embeddings":
            description_question_token_embedding_n_similarity,
    })

    # Remove features with 0 score.
    features = dict((feature, value) for feature, value in features.iteritems() if value != 0.0)
    return features


def generate_document_vectors(question_text, question_tokens2pos, questions_search_results):
    if not _documents_vectors_cache:
        import os
        cache_file = globals.config.get('WebSearchFeatures', 'document-vectors')
        if os.path.isfile(cache_file):
            logger.info("Reading cached document vectors...")
            with open(cache_file, 'r') as inp:
                # Unpickle until the end of file is reached
                index = 0
                while True:
                    try:
                        question, vectors = pickle.load(inp)
                        _documents_vectors_cache[question] = vectors
                    except (EOFError, pickle.UnpicklingError):
                        break
                    index += 1
                    if index % 1000 == 0:
                        logger.info("Read " + str(index) + " document vectors...")
            logger.info("Reading cached document vectors done!")

    if question_text in _documents_vectors_cache:
        return _documents_vectors_cache[question_text]

    documents_vectors = []
    snippets_vectors = []
    fragment_vectors = []
    combined_doc_token2pos = dict()
    combined_doc_entity2pos = dict()
    combined_doc_snippet_token2pos = dict()
    combined_doc_snippet_entity2pos = dict()
    for document in questions_search_results[question_text][:globals.SEARCH_RESULTS_TOPN]:
        # Whole document
        doc_entity2pos = document.get_mentioned_entities_to_pos()
        doc_token2pos, doc_lemma2pos = document.get_token_to_positions_map()
        documents_vectors.append({
            "entity": SparseVector.from_2pos(doc_entity2pos),
            "entity_tfidf": SparseVector.from_2pos(doc_entity2pos,
                                                   element_calc_func=SparseVector.compute_tfidf_entity_elements),
            "token": SparseVector.from_2pos(doc_token2pos),
            "token_tfidf": SparseVector.from_2pos(doc_token2pos,
                                                  element_calc_func=SparseVector.compute_tfidf_token_elements),
        })
        merge_2pos_dicts(combined_doc_token2pos, doc_token2pos)
        merge_2pos_dicts(combined_doc_entity2pos, doc_entity2pos)

        # Snippet
        doc_snippet_entity2pos = document.get_snippet_entities_to_pos()
        doc_snippet_token2pos, doc_snippet_lemma2pos = document.get_snippet_token2pos()
        snippets_vectors.append({
            "entity": SparseVector.from_2pos(doc_snippet_entity2pos),
            "entity_tfidf": SparseVector.from_2pos(doc_snippet_entity2pos,
                                                   element_calc_func=SparseVector.compute_tfidf_entity_elements),
            "token": SparseVector.from_2pos(doc_snippet_token2pos),
            "token_tfidf": SparseVector.from_2pos(doc_snippet_token2pos,
                                                  element_calc_func=SparseVector.compute_tfidf_token_elements),
        })
        merge_2pos_dicts(combined_doc_snippet_token2pos, doc_snippet_token2pos)
        merge_2pos_dicts(combined_doc_snippet_entity2pos, doc_snippet_entity2pos)

        # Best fragment
        fragment_token2pos, fragment_entity2pos = \
            WebSearchResult.get_best_fragment_positions(doc_token2pos, doc_entity2pos, question_tokens2pos)
        fragment_vectors.append({
            "entity": SparseVector.from_2pos(fragment_entity2pos),
            "entity_tfidf": SparseVector.from_2pos(fragment_entity2pos,
                                                   element_calc_func=SparseVector.compute_tfidf_entity_elements),
            "token": SparseVector.from_2pos(fragment_token2pos),
            "token_tfidf": SparseVector.from_2pos(fragment_token2pos,
                                                  element_calc_func=SparseVector.compute_tfidf_token_elements),
        })
    combined_documents_vector = {
        "entity": SparseVector.from_2pos(combined_doc_entity2pos),
        "entity_tfidf": SparseVector.from_2pos(combined_doc_entity2pos,
                                               element_calc_func=SparseVector.compute_tfidf_entity_elements),
        "token": SparseVector.from_2pos(combined_doc_token2pos),
        "token_tfidf": SparseVector.from_2pos(combined_doc_token2pos,
                                              element_calc_func=SparseVector.compute_tfidf_token_elements),
    }
    combined_document_snippets_vector = {
        "entity": SparseVector.from_2pos(combined_doc_snippet_entity2pos),
        "entity_tfidf": SparseVector.from_2pos(combined_doc_snippet_entity2pos,
                                               element_calc_func=SparseVector.compute_tfidf_entity_elements),
        "token": SparseVector.from_2pos(combined_doc_snippet_token2pos),
        "token_tfidf": SparseVector.from_2pos(combined_doc_snippet_token2pos,
                                              element_calc_func=SparseVector.compute_tfidf_token_elements),
    }

    # Cache the computed vectors.
    _documents_vectors_cache[question_text] = (documents_vectors, snippets_vectors, fragment_vectors,
                                               combined_documents_vector, combined_document_snippets_vector)

    return documents_vectors, snippets_vectors, fragment_vectors, combined_documents_vector, combined_document_snippets_vector


def create_document_vectors_cache(questions):
    cache_file = globals.config.get('WebSearchFeatures', 'document-vectors')
    logger.info("Caching document vectors...")
    with open(cache_file, 'wx') as out:
        for index, question in enumerate(questions):
            question_token2pos = dict((token, [1, ]) for token in tokenize(question))
            generate_document_vectors(question, question_token2pos, get_questions_serps())
            pickle.dump((question, _documents_vectors_cache[question]), out)
            if index % 100 == 0:
                logger.info("Cached document vectors for %d questions" % index)


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s : %(levelname)s '
                               ': %(module)s : %(message)s',
                        level=logging.INFO)
    globals.read_configuration('config_yahoofactoid.cfg')
    serps = get_questions_serps()
    create_document_vectors_cache(serps.keys())
