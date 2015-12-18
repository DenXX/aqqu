"""
Copyright 2015, University of Freiburg.

Elmar Haussmann <haussmann@cs.uni-freiburg.de>
"""
from query_translator import ranker
from collections import OrderedDict

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
                   ranker.AccuModel('WQ_Ranker',
                                     "webquestionstrain",
                                     top_ngram_percentile=5,
                                     rel_regularization_C=1.0),
                   ranker.AccuModel('WQ_Ranker_Dev',
                                     "webquestions_split_train",
                                     top_ngram_percentile=5,
                                     rel_regularization_C=1.0),
                   ranker.AccuModel('WQ_Ranker_NoPruning_Dev',
                                     "webquestions_split_train",
                                     top_ngram_percentile=5,
                                     rel_regularization_C=1.0,
                                     use_pruning=False),
                   ranker.AccuModel('WQ_Ranker_ExternalEntities',
                                     "webquestionstrain_externalentities",
                                     top_ngram_percentile=5,
                                     rel_regularization_C=1.0),
                   ranker.AccuModel('WQ_Ranker_WithTextRank',
                                     "webquestionstrain",
                                     top_ngram_percentile=5,
                                     rel_regularization_C=1.0,
                                     extract_text_features_pruning=False,
                                     extract_text_features_ranking=True,
                                     use_pruning=True),
                   ranker.AccuModel('WQ_Ranker_WithTextPruneRank',
                                     "webquestionstrain",
                                     top_ngram_percentile=5,
                                     rel_regularization_C=1.0,
                                     extract_text_features_pruning=True,
                                     extract_text_features_ranking=True,
                                     use_pruning=True),

                   # Baseline models with more iterations
                   ranker.AccuModel('WQ_Ranker_RF100_Dev',
                                     "webquestions_split_train",
                                     top_ngram_percentile=5,
                                     ranking_algorithm='random_forest',
                                     rel_regularization_C=1.0,
                                     extract_text_features_pruning=False,
                                     extract_text_features_ranking=False,
                                     use_pruning=True),
                   ranker.AccuModel('WQ_Ranker_RF500_Dev',
                                     "webquestions_split_train",
                                     top_ngram_percentile=5,
                                     ranking_algorithm='random_forest',
                                     ranking_n_estimators=500,
                                     rel_regularization_C=1.0,
                                     extract_text_features_pruning=False,
                                     extract_text_features_ranking=False,
                                     use_pruning=True),
                   ranker.AccuModel('WQ_Ranker_RF1000_Dev',
                                     "webquestions_split_train",
                                     top_ngram_percentile=5,
                                     ranking_algorithm='random_forest',
                                     ranking_n_estimators=1000,
                                     rel_regularization_C=1.0,
                                     extract_text_features_pruning=False,
                                     extract_text_features_ranking=False,
                                     use_pruning=True),

                   # Baselines with external entities
                   ranker.AccuModel('WQ_Ranker_ExtEnt_RF100_Dev',
                                     "webquestions_split_train_externalentities",
                                     top_ngram_percentile=5,
                                     ranking_algorithm='random_forest',
                                     rel_regularization_C=1.0,
                                     extract_text_features_pruning=False,
                                     extract_text_features_ranking=False,
                                     use_pruning=True),
                   ranker.AccuModel('WQ_Ranker_ExtEnt3_RF100_Dev',
                                     "webquestions_split_train_externalentities3",
                                     top_ngram_percentile=5,
                                     ranking_algorithm='random_forest',
                                     rel_regularization_C=1.0,
                                     extract_text_features_pruning=False,
                                     extract_text_features_ranking=False,
                                     use_pruning=True),
                   ranker.AccuModel('WQ_Ranker_ExtEnt_RF100',
                                     "webquestions_train_externalentities",
                                     top_ngram_percentile=5,
                                     ranking_algorithm='random_forest',
                                     rel_regularization_C=1.0,
                                     extract_text_features_pruning=False,
                                     extract_text_features_ranking=False,
                                     use_pruning=True),
                   ranker.AccuModel('WQ_Ranker_ExtEnt3_RF100',
                                     "webquestions_train_externalentities3",
                                     top_ngram_percentile=5,
                                     ranking_algorithm='random_forest',
                                     rel_regularization_C=1.0,
                                     extract_text_features_pruning=False,
                                     extract_text_features_ranking=False,
                                     use_pruning=True),

                   # Text models with more iterations
                   ranker.AccuModel('WQ_Ranker_WithTextRank_RF100_Dev',
                                     "webquestions_split_train",
                                     top_ngram_percentile=5,
                                     ranking_algorithm='random_forest',
                                     ranking_n_estimators=100,
                                     rel_regularization_C=1.0,
                                     extract_text_features_pruning=False,
                                     extract_text_features_ranking=True,
                                     use_pruning=True),
                   ranker.AccuModel('WQ_Ranker_WithTextRank_RF1000_Dev',
                                     "webquestions_split_train",
                                     top_ngram_percentile=5,
                                     ranking_algorithm='random_forest',
                                     ranking_n_estimators=1000,
                                     rel_regularization_C=1.0,
                                     extract_text_features_pruning=False,
                                     extract_text_features_ranking=True,
                                     use_pruning=True),
                   ranker.AccuModel('WQ_Ranker_WithTextRankPrune_RF100_Dev',
                                     "webquestions_split_train",
                                     top_ngram_percentile=5,
                                     ranking_algorithm='random_forest',
                                     ranking_n_estimators=100,
                                     rel_regularization_C=1.0,
                                     extract_text_features_pruning=True,
                                     extract_text_features_ranking=True,
                                     use_pruning=True),
                   ranker.AccuModel('WQ_Ranker_WithTextRankPrune_RF1000_Dev',
                                     "webquestions_split_train",
                                     top_ngram_percentile=5,
                                     ranking_algorithm='random_forest',
                                     ranking_n_estimators=1000,
                                     rel_regularization_C=1.0,
                                     extract_text_features_pruning=True,
                                     extract_text_features_ranking=True,
                                     use_pruning=True),
                  ranker.AccuModel('WQ_Ranker_WithTextRankNoPrune_RF1000_Dev',
                                     "webquestions_split_train",
                                     top_ngram_percentile=5,
                                     ranking_algorithm='random_forest',
                                     ranking_n_estimators=1000,
                                     rel_regularization_C=1.0,
                                     extract_text_features_pruning=True,
                                     extract_text_features_ranking=True,
                                     use_pruning=False),



                   # -------------------------------------------------------------
                   # Here is my main model for now!
                   ranker.AccuModel('WQ_Ranker_WithTextPruneRank_Dev',
                                     "webquestions_split_train",
                                     top_ngram_percentile=5,
                                     rel_regularization_C=1.0,
                                     extract_text_features_pruning=True,
                                     extract_text_features_ranking=True,
                                     use_pruning=True),
                   ranker.AccuModel('WQ_Ranker_WithTextRank_Dev',
                                     "webquestions_split_train",
                                     top_ngram_percentile=5,
                                     rel_regularization_C=1.0,
                                     extract_text_features_pruning=False,
                                     extract_text_features_ranking=True,
                                     use_pruning=True),
                   ranker.AccuModel('WQ_Ranker_WithTextRank_RF1000_Dev',
                                     "webquestions_split_train",
                                     top_ngram_percentile=5,
                                     rel_regularization_C=1.0,
                                     ranking_n_estimators=1000,
                                     extract_text_features_pruning=False,
                                     extract_text_features_ranking=True,
                                     use_pruning=True),
                   ranker.AccuModel('WQ_Ranker_WithTextRank_GBT1000_Dev',
                                     "webquestions_split_train",
                                     top_ngram_percentile=5,
                                     rel_regularization_C=1.0,
                                     ranking_algorithm='gbt',
                                     ranking_n_estimators=1000,
                                     extract_text_features_pruning=False,
                                     extract_text_features_ranking=True,
                                     use_pruning=True),






                   ranker.AccuModel('WQ_Ranker_WithTextPruneRank_ExternalEntities',
                                     "webquestionstrain_externalentities",
                                     top_ngram_percentile=5,
                                     rel_regularization_C=1.0,
                                     extract_text_features_pruning=True,
                                     extract_text_features_ranking=True,
                                     use_pruning=True),
                   ranker.AccuModel('WQ_Ranker_WithText_NoPruning',
                                     "webquestionstrain",
                                     top_ngram_percentile=5,
                                     rel_regularization_C=1.0,
                                     extract_text_features_pruning=False,
                                     extract_text_features_ranking=True,
                                     use_pruning=False),
                   ranker.AccuModel('WQ_Ranker_WithText_small',
                                     "webquestions_small_train",
                                     top_ngram_percentile=5,
                                     rel_regularization_C=1.0,
                                     extract_text_features_ranking=True,
                                     extract_text_features_pruning=True,
                                     use_pruning=True),
                   ranker.SimpleScoreRanker('SimpleRanker'),
                   ranker.SimpleScoreRanker('SimpleRanker_entity_oracle',
                                             entity_oracle_file=free917_entities),
                   ranker.LiteralRanker('LiteralRanker'),
                   ranker.LiteralRanker('LiteralRanker_entity_oracle',
                                         entity_oracle_file=free917_entities),
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
         ]
    )
