"""
Copyright 2015, University of Freiburg.

Elmar Haussmann <haussmann@cs.uni-freiburg.de>
"""
from query_translator import ranker
from collections import OrderedDict

from text2kb.web_answerer import QuaseWebAnswerer, BingWebAnswerer, TextAnswerer

free917_entities = "evaluation-data/free917_entities.txt"

scorer_list = None
scorers_dict = None
DATASETS = None


def init():
    global scorer_list
    global scorers_dict
    global DATASETS
    # The scorers that can be selected.
    scorer_list = [ranker.AccuModel('F917_Ranker',
                                     "free917train",
                                     top_ngram_percentile=2,
                                     rel_regularization_C=0.002),
                   ranker.AccuModel('F917_Ranker_entity_oracle',
                                     "free917train",
                                     entity_oracle_file=free917_entities,
                                     top_ngram_percentile=2,
                                     rel_regularization_C=1.0),

                   # Main models
                   ranker.AccuModel('WQ_Ranker_RF100_TypeFilter05_Dev',
                                     "webquestions_split_train_typefilter05",
                                     top_ngram_percentile=5,
                                     ranking_algorithm='random_forest',
                                     ranking_n_estimators=100,
                                     rel_regularization_C=1.0,
                                     use_type_model=False,
                                     extract_text_features_pruning=False,
                                     extract_text_features_ranking=False,
                                     use_pruning=True),
                   ranker.AccuModel('WQ_Ranker_RF100_TypeFilter07_Dev',
                                     "webquestions_split_train_typefilter07",
                                     top_ngram_percentile=5,
                                     ranking_algorithm='random_forest',
                                     ranking_n_estimators=100,
                                     rel_regularization_C=1.0,
                                     use_type_model=False,
                                     extract_text_features_pruning=False,
                                     extract_text_features_ranking=False,
                                     use_pruning=True),

                   ranker.AccuModel('WQ_Ranker',
                                    "webquestionstrain",
                                    ranking_n_estimators=100,
                                    top_ngram_percentile=5,
                                    rel_regularization_C=1.0,
                                    use_type_model=False,
                                    extract_text_features_pruning=False,
                                    extract_text_features_ranking=False,
                                    extract_cqa_features_pruning=False,
                                    extract_cqa_features_ranking=False,
                                    extract_clueweb_features_pruning=False,
                                    extract_clueweb_features_ranking=False,
                                    use_pruning=True),

                   ranker.AccuModel('WQ_Ranker_ExtEnt_WebCqaClue_Dates_Types_TypeModel',
                                    "webquestions_train_extent_dates_types",
                                    top_ngram_percentile=5,
                                    ranking_algorithm='random_forest',
                                    ranking_n_estimators=100,
                                    rel_regularization_C=1.0,
                                    use_type_model=True,
                                    extract_text_features_pruning=False,
                                    extract_text_features_ranking=True,
                                    extract_cqa_features_pruning=False,
                                    extract_cqa_features_ranking=True,
                                    extract_clueweb_features_pruning=False,
                                    extract_clueweb_features_ranking=True,
                                    use_pruning=True),
                   ranker.AccuModel('WQ_Ranker_ExtEnt_WebCqaClue_Dates_Types_TypeModel_RF300',
                                    "webquestions_train_extent_dates_types",
                                    top_ngram_percentile=5,
                                    ranking_algorithm='random_forest',
                                    ranking_n_estimators=300,
                                    rel_regularization_C=1.0,
                                    use_type_model=True,
                                    extract_text_features_pruning=False,
                                    extract_text_features_ranking=True,
                                    extract_cqa_features_pruning=False,
                                    extract_cqa_features_ranking=True,
                                    extract_clueweb_features_pruning=False,
                                    extract_clueweb_features_ranking=True,
                                    use_pruning=True),


                   ranker.AccuModel('WQ_Ranker_WebCqaClue',
                                    "webquestionstrain",
                                    top_ngram_percentile=5,
                                    ranking_algorithm='random_forest',
                                    ranking_n_estimators=100,
                                    rel_regularization_C=1.0,
                                    use_type_model=False,
                                    extract_text_features_pruning=False,
                                    extract_text_features_ranking=True,
                                    extract_cqa_features_pruning=False,
                                    extract_cqa_features_ranking=True,
                                    extract_clueweb_features_pruning=False,
                                    extract_clueweb_features_ranking=True,
                                    use_pruning=True),

                   ranker.AccuModel('WQ_Ranker_ExtEnt_WebCqaClue_Dates_TypeModel',
                                    "webquestions_train_extent_dates",
                                    top_ngram_percentile=5,
                                    ranking_algorithm='random_forest',
                                    ranking_n_estimators=100,
                                    rel_regularization_C=1.0,
                                    use_type_model=True,
                                    extract_text_features_pruning=False,
                                    extract_text_features_ranking=True,
                                    extract_cqa_features_pruning=False,
                                    extract_cqa_features_ranking=True,
                                    extract_clueweb_features_pruning=False,
                                    extract_clueweb_features_ranking=True,
                                    use_pruning=True),
                   ranker.AccuModel('WQ_Ranker_Wiki_ExtEnt_WebCqaClue_Dates_TypeModel',
                                    "webquestions_train_extent_dates",
                                    top_ngram_percentile=5,
                                    ranking_algorithm='random_forest',
                                    ranking_n_estimators=100,
                                    rel_regularization_C=1.0,
                                    use_type_model=True,
                                    extract_text_features_pruning=False,
                                    extract_text_features_ranking=True,
                                    extract_cqa_features_pruning=False,
                                    extract_cqa_features_ranking=True,
                                    extract_clueweb_features_pruning=False,
                                    extract_clueweb_features_ranking=True,
                                    use_pruning=True),

                   ranker.AccuModel('WQ_Ranker_ExtEnt_WebCqaClue_Dates_TypeModel_RF300',
                                    "webquestions_train_extent_dates",
                                    top_ngram_percentile=5,
                                    ranking_algorithm='random_forest',
                                    ranking_n_estimators=300,
                                    rel_regularization_C=1.0,
                                    use_type_model=True,
                                    extract_text_features_pruning=False,
                                    extract_text_features_ranking=True,
                                    extract_cqa_features_pruning=False,
                                    extract_cqa_features_ranking=True,
                                    extract_clueweb_features_pruning=False,
                                    extract_clueweb_features_ranking=True,
                                    use_pruning=True),
                   ranker.AccuModel('WQ_Ranker_ExtEnt_WebCqaClue_Dates_TypeModel_RF1000',
                                    "webquestions_train_extent_dates",
                                    top_ngram_percentile=5,
                                    ranking_algorithm='random_forest',
                                    ranking_n_estimators=1000,
                                    rel_regularization_C=1.0,
                                    use_type_model=True,
                                    extract_text_features_pruning=False,
                                    extract_text_features_ranking=True,
                                    extract_cqa_features_pruning=False,
                                    extract_cqa_features_ranking=True,
                                    extract_clueweb_features_pruning=False,
                                    extract_clueweb_features_ranking=True,
                                    use_pruning=True),

                   ranker.AccuModel('WQ_Ranker_WebCqaClue_TypeModel',
                                    "webquestionstrain",
                                    top_ngram_percentile=5,
                                    ranking_algorithm='random_forest',
                                    ranking_n_estimators=100,
                                    rel_regularization_C=1.0,
                                    use_type_model=True,
                                    extract_text_features_pruning=False,
                                    extract_text_features_ranking=True,
                                    extract_cqa_features_pruning=False,
                                    extract_cqa_features_ranking=True,
                                    extract_clueweb_features_pruning=False,
                                    extract_clueweb_features_ranking=True,
                                    use_pruning=True),
                   ranker.AccuModel('WQ_Ranker_TypeModel',
                                    "webquestionstrain",
                                    top_ngram_percentile=5,
                                    ranking_algorithm='random_forest',
                                    ranking_n_estimators=100,
                                    rel_regularization_C=1.0,
                                    use_type_model=True,
                                    extract_text_features_pruning=False,
                                    extract_text_features_ranking=False,
                                    extract_cqa_features_pruning=False,
                                    extract_cqa_features_ranking=False,
                                    extract_clueweb_features_pruning=False,
                                    extract_clueweb_features_ranking=False,
                                    use_pruning=True),
                   ranker.AccuModel('WQ_Ranker_TypeModel_Dates',
                                    "webquestions_train_dates",
                                    top_ngram_percentile=5,
                                    ranking_algorithm='random_forest',
                                    ranking_n_estimators=100,
                                    rel_regularization_C=1.0,
                                    use_type_model=True,
                                    extract_text_features_pruning=False,
                                    extract_text_features_ranking=False,
                                    extract_cqa_features_pruning=False,
                                    extract_cqa_features_ranking=False,
                                    extract_clueweb_features_pruning=False,
                                    extract_clueweb_features_ranking=False,
                                    use_pruning=True),
                   ranker.AccuModel('WQ_Ranker_WebCqaClue_TypeModel_Dates',
                                    "webquestions_train_dates",
                                    top_ngram_percentile=5,
                                    ranking_algorithm='random_forest',
                                    ranking_n_estimators=100,
                                    rel_regularization_C=1.0,
                                    use_type_model=True,
                                    extract_text_features_pruning=False,
                                    extract_text_features_ranking=True,
                                    extract_cqa_features_pruning=False,
                                    extract_cqa_features_ranking=True,
                                    extract_clueweb_features_pruning=False,
                                    extract_clueweb_features_ranking=True,
                                    use_pruning=True),
                   ranker.AccuModel('WQ_Ranker_Wiki_WebCqaClue_TypeModel_Dates',
                                    "webquestions_train_dates",
                                    top_ngram_percentile=5,
                                    ranking_algorithm='random_forest',
                                    ranking_n_estimators=100,
                                    rel_regularization_C=1.0,
                                    use_type_model=True,
                                    extract_text_features_pruning=False,
                                    extract_text_features_ranking=True,
                                    extract_cqa_features_pruning=False,
                                    extract_cqa_features_ranking=True,
                                    extract_clueweb_features_pruning=False,
                                    extract_clueweb_features_ranking=True,
                                    use_pruning=True),

                   ranker.AccuModel('WQ_Ranker_ExtEnt_WebCqa_Dates_TypeModel',
                                    "webquestions_train_extent_dates",
                                    top_ngram_percentile=5,
                                    ranking_algorithm='random_forest',
                                    ranking_n_estimators=100,
                                    rel_regularization_C=1.0,
                                    use_type_model=True,
                                    extract_text_features_pruning=False,
                                    extract_text_features_ranking=True,
                                    extract_cqa_features_pruning=False,
                                    extract_cqa_features_ranking=True,
                                    extract_clueweb_features_pruning=False,
                                    extract_clueweb_features_ranking=False,
                                    use_pruning=True),
                   ranker.AccuModel('WQ_Ranker_ExtEnt_WebClue_Dates_TypeModel',
                                    "webquestions_train_extent_dates",
                                    top_ngram_percentile=5,
                                    ranking_algorithm='random_forest',
                                    ranking_n_estimators=100,
                                    rel_regularization_C=1.0,
                                    use_type_model=True,
                                    extract_text_features_pruning=False,
                                    extract_text_features_ranking=True,
                                    extract_cqa_features_pruning=False,
                                    extract_cqa_features_ranking=False,
                                    extract_clueweb_features_pruning=False,
                                    extract_clueweb_features_ranking=True,
                                    use_pruning=True),
                   ranker.AccuModel('WQ_Ranker_ExtEnt_CqaClue_Dates_TypeModel',
                                    "webquestions_train_extent_dates",
                                    top_ngram_percentile=5,
                                    ranking_algorithm='random_forest',
                                    ranking_n_estimators=100,
                                    rel_regularization_C=1.0,
                                    use_type_model=True,
                                    extract_text_features_pruning=False,
                                    extract_text_features_ranking=False,
                                    extract_cqa_features_pruning=False,
                                    extract_cqa_features_ranking=True,
                                    extract_clueweb_features_pruning=False,
                                    extract_clueweb_features_ranking=True,
                                    use_pruning=True),

                   ranker.AccuModel('WQ_Ranker_ExtEnt_Web_Dates_TypeModel',
                                    "webquestions_train_extent_dates",
                                    top_ngram_percentile=5,
                                    ranking_algorithm='random_forest',
                                    ranking_n_estimators=100,
                                    rel_regularization_C=1.0,
                                    use_type_model=True,
                                    extract_text_features_pruning=False,
                                    extract_text_features_ranking=True,
                                    extract_cqa_features_pruning=False,
                                    extract_cqa_features_ranking=False,
                                    extract_clueweb_features_pruning=False,
                                    extract_clueweb_features_ranking=False,
                                    use_pruning=True),
                   ranker.AccuModel('WQ_Ranker_ExtEnt_Clue_Dates_TypeModel',
                                    "webquestions_train_extent_dates",
                                    top_ngram_percentile=5,
                                    ranking_algorithm='random_forest',
                                    ranking_n_estimators=100,
                                    rel_regularization_C=1.0,
                                    use_type_model=True,
                                    extract_text_features_pruning=False,
                                    extract_text_features_ranking=False,
                                    extract_cqa_features_pruning=False,
                                    extract_cqa_features_ranking=False,
                                    extract_clueweb_features_pruning=False,
                                    extract_clueweb_features_ranking=True,
                                    use_pruning=True),
                   ranker.AccuModel('WQ_Ranker_ExtEnt_Cqa_Dates_TypeModel',
                                    "webquestions_train_extent_dates",
                                    top_ngram_percentile=5,
                                    ranking_algorithm='random_forest',
                                    ranking_n_estimators=100,
                                    rel_regularization_C=1.0,
                                    use_type_model=True,
                                    extract_text_features_pruning=False,
                                    extract_text_features_ranking=False,
                                    extract_cqa_features_pruning=False,
                                    extract_cqa_features_ranking=True,
                                    extract_clueweb_features_pruning=False,
                                    extract_clueweb_features_ranking=False,
                                    use_pruning=True),


                   ranker.AccuModel('WQ_Ranker_ExtEnt_Dates_TypeModel',
                                    "webquestions_train_extent_dates",
                                    top_ngram_percentile=5,
                                    ranking_algorithm='random_forest',
                                    ranking_n_estimators=100,
                                    rel_regularization_C=1.0,
                                    use_type_model=True,
                                    extract_text_features_pruning=False,
                                    extract_text_features_ranking=False,
                                    extract_cqa_features_pruning=False,
                                    extract_cqa_features_ranking=False,
                                    extract_clueweb_features_pruning=False,
                                    extract_clueweb_features_ranking=False,
                                    use_pruning=True),

                   ranker.AccuModel('WQ_Ranker_small',
                                    "webquestions_small_test",
                                    top_ngram_percentile=5,
                                    rel_regularization_C=1.0,
                                    extract_text_features_ranking=False,
                                    extract_text_features_pruning=False,
                                    extract_cqa_features_pruning=False,
                                    extract_cqa_features_ranking=False,
                                    extract_clueweb_features_pruning=False,
                                    extract_clueweb_features_ranking=False,
                                    use_pruning=True),
                   ranker.SimpleScoreRanker('SimpleRanker'),
                   ranker.SimpleScoreRanker('SimpleRanker_entity_oracle',
                                             entity_oracle_file=free917_entities),
                   ranker.LiteralRanker('LiteralRanker'),
                   ranker.LiteralRanker('LiteralRanker_entity_oracle',
                                         entity_oracle_file=free917_entities),

                   # TREC QA experiments
                   ranker.AccuModel('Aqqu_on_TRECQA_Ranker',
                                    "trecqa_train",
                                    ranking_n_estimators=100,
                                    top_ngram_percentile=5,
                                    rel_regularization_C=1.0,
                                    use_type_model=False,
                                    extract_text_features_pruning=False,
                                    extract_text_features_ranking=False,
                                    extract_cqa_features_pruning=False,
                                    extract_cqa_features_ranking=False,
                                    extract_clueweb_features_pruning=False,
                                    extract_clueweb_features_ranking=False,
                                    use_pruning=False),

                   ranker.AccuModel('Aqqu_on_Yahoo_Ranker',
                                    "yahoo_train_full",
                                    ranking_n_estimators=100,
                                    top_ngram_percentile=5,
                                    rel_regularization_C=1.0,
                                    use_type_model=False,
                                    extract_text_features_pruning=False,
                                    extract_text_features_ranking=False,
                                    extract_cqa_features_pruning=False,
                                    extract_cqa_features_ranking=False,
                                    extract_clueweb_features_pruning=False,
                                    extract_clueweb_features_ranking=False,
                                    use_pruning=True),

                   QuaseWebAnswerer('AskMSR'),
                   BingWebAnswerer('BingSearchCount', entity_link_min_score=0.1, use_answers_cache=False),
                   TextAnswerer('SentSearchCount', "trecqa_train"),
                   ]

    # A dictionary used for lookup via scorer name.
    scorers_dict = OrderedDict(
        [(s.name, s) for s in scorer_list]
    )

    # A dict of dataset name and file.
    DATASETS = OrderedDict(
        [('free917train',
          'evaluation-data/'
          'free917.train.json'),
         ('webquestionstrain',
          'evaluation-data/'
          'webquestions.train.json'),
         ('free917train_1of2',
          'evaluation-data/'
          'free917.train_1of2.json'),
         ('free917train_2of2',
          'evaluation-data/'
          'free917.train_2of2.json'),
         ('webquestionstrain_1of2',
          'evaluation-data/'
          'webquestions.train_1of2.json'),
         ('webquestionstrain_1of2_1of2',
          'evaluation-data/'
          'webquestions.train_1of2_1of2.json'),
         ('webquestionstrain_1of2_2of2',
          'evaluation-data/'
          'webquestions.train_1of2_2of2.json'),
         ('webquestionstrain_2of2',
          'evaluation-data/'
          'webquestions.train_2of2.json'),
         ('free917test',
          'evaluation-data/'
          'free917.test.json'),
         ('webquestionstest',
          'evaluation-data/'
          'webquestions.test.json'),
         ('free917test_graphparser',
          'evaluation-data/'
          'free917.test_graphparser.json'),
         ('webquestionstest_graphparser',
          'evaluation-data/'
          'webquestions.test_graphparser.json'),
         ('webquestionstrain_graphparser',
          'evaluation-data/'
          'webquestions.train_graphparser.json'),
         ('webquestions_small_train',
          'evaluation-data/'
          'webquestions.train_small.json'),
         ('webquestions_small_test',
          'evaluation-data/'
          'webquestions.test_small.json'),
         ('webquestions_split_train',
          'evaluation-data/'
          'webquestions.split.train.json'),
         ('webquestions_split_dev',
          'evaluation-data/'
          'webquestions.split.dev.json'),
         ('webquestions_train_externalentities',
          'evaluation-data/'
          'webquestions.train.extent.json'),
         ('webquestions_test_externalentities',
          'evaluation-data/'
          'webquestions.test.extent.json'),
         ('webquestions_train_externalentities3',
          'evaluation-data/'
          'webquestions.train.extent3.json'),
         ('webquestions_test_externalentities3',
          'evaluation-data/'
          'webquestions.test.extent3.json'),

         ('webquestions_split_train_externalentities',
          'evaluation-data/'
          'webquestions.split.train.extent.json'),
         ('webquestions_split_dev_externalentities',
          'evaluation-data/'
          'webquestions.split.dev.extent.json'),


         ('webquestions_split_train_externalentities3',
          'evaluation-data/'
          'webquestions.split.train.extent3.json'),
         ('webquestions_split_dev_externalentities3',
          'evaluation-data/'
          'webquestions.split.dev.extent3.json'),

         ('webquestions_split_train_externalentities_all',
          'evaluation-data/'
          'webquestions.split.train.extent_all.json'),
         ('webquestions_split_dev_externalentities_all',
          'evaluation-data/'
          'webquestions.split.dev.extent_all.json'),
         ('webquestions_train_externalentities_all',
          'evaluation-data/'
          'webquestions.train.extent_all.json'),
         ('webquestions_test_externalentities_all',
          'evaluation-data/'
          'webquestions.test.extent_all.json'),

         # Train and test with external entities and extra date range templates
         ('webquestions_train_extent_dates',
          'evaluation-data/'
          'webquestions.train.extent.dates.json'),
         ('webquestions_test_extent_dates',
          'evaluation-data/'
          'webquestions.test.extent.dates.json'),
         ('webquestions_train_extent_dates_types',
          'evaluation-data/'
          'webquestions.train.extent.dates.types.json'),
         ('webquestions_test_extent_dates_types',
          'evaluation-data/'
          'webquestions.test.extent.dates.types.json'),
         ('webquestions_train_dates',
          'evaluation-data/'
          'webquestions.train.dates.json'),
         ('webquestions_test_dates',
          'evaluation-data/'
          'webquestions.test.dates.json'),



         ('webquestions_split_train_externalentities_all_daterange',
          'evaluation-data/'
          'webquestions.split.train.extent_all_daterange.json'),
         ('webquestions_split_dev_externalentities_all_daterange',
          'evaluation-data/'
          'webquestions.split.dev.extent_all_daterange.json'),



         ('webquestions_split_train_daterange',
          'evaluation-data/'
          'webquestions.split.train.daterange.json'),
         ('webquestions_split_dev_daterange',
          'evaluation-data/'
          'webquestions.split.dev.daterange.json'),

         ('webquestions_split_train_typefilter05',
          'evaluation-data/'
          'webquestions.split.train.typefilter05.json'),
         ('webquestions_split_dev_typefilter05',
          'evaluation-data/'
          'webquestions.split.dev.typefilter05.json'),
         ('webquestions_split_train_typefilter07',
          'evaluation-data/'
          'webquestions.split.train.typefilter07.json'),
         ('webquestions_split_dev_typefilter07',
          'evaluation-data/'
          'webquestions.split.dev.typefilter07.json'),


         ('webquestions_test_filter',
          'evaluation-data/'
          'webquestions.test_for_filter.json'),

         # TREC QA datasets
         ('trecqa_train',
          'evaluation-data/'
          'trecqa.train.json'),
         ('trecqa_test',
          'evaluation-data/'
          'trecqa.test.json'),
         ('trecqa_test_small',
          'evaluation-data/'
          'trecqa.test_small.json'),

        # YAHOO FACTOID
         ('yahoo_train_full',
          'evaluation-data/'
          'yahoofactoid.train_dev.json'),
         ('yahoo_train',
          'evaluation-data/'
          'yahoofactoid.train.json'),
         ('yahoo_dev',
          'evaluation-data/'
          'yahoofactoid.dev.json'),
         ('yahoofactoid_test',
          'evaluation-data/'
          'yahoofactoid.test.json'),

         ]
    )
