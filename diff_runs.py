
import cPickle as pickle

import sys
from scipy.stats import ttest_rel
from sys import argv

from query_translator.evaluation import evaluate


def read_queries(filename):
    print >> sys.stderr, "Reading queries from %s" % filename
    with open(filename, 'rb') as inp:
        return pickle.load(inp)

if __name__ == "__main__":
    baseline_queries = read_queries(argv[1])

    results = dict()
    answers = dict()
    id2questionanswer = dict()
    for query in baseline_queries:
        id2questionanswer[query.id] = (query.utterance, query.target_result)
        if query.id not in results:
            results[query.id] = {"avg_precision": [0, 0], "avg_recall": [0, 0], "avg_f1": [0, 0]}
        results[query.id]["avg_precision"][0] = query.precision
        results[query.id]["avg_recall"][0] = query.recall
        results[query.id]["avg_f1"][0] = query.f1
        answers[query.id] = [query.eval_candidates[0] if query.eval_candidates else None, None]

    test_queries = read_queries(argv[2])
    for query in test_queries:
        results[query.id]["avg_precision"][1] = query.precision
        results[query.id]["avg_recall"][1] = query.recall
        results[query.id]["avg_f1"][1] = query.f1

        assert query.id in answers
        answers[query.id][1] = query.eval_candidates[0] if query.eval_candidates else None

    diff_count = 0
    win_count = 0
    loose_count = 0
    for id, answers in answers.iteritems():
        baseline_answer, test_answer = answers
        if (baseline_answer is None and test_answer is not None) or \
            (baseline_answer is not None and test_answer is None) or \
                (baseline_answer is not None and test_answer is not None and
                         baseline_answer.prediction != test_answer.prediction):
            baseline_f1 = baseline_answer.evaluation_result.f1 if baseline_answer is not None else 0.0
            system_f1 = test_answer.evaluation_result.f1 if test_answer is not None else 0.0
            win = baseline_f1 < system_f1
            loose = baseline_f1 > system_f1

            # Increasing win, loose and total counters.
            if win:
                win_count += 1
            elif loose:
                loose_count += 1
            diff_count += 1

            # Printing answers
            reset = "\x1B[m"
            red_text = "\x1B[31m%s\x1B[0m"
            green_text = "\x1B[32m%s\x1B[0m"

            print id2questionanswer[id][0], green_text % "WIN" if win else (red_text % "LOOSE" if loose else "")
            print "Correct answer: ", id2questionanswer[id][1]
            print "Baseline: ", (baseline_answer.prediction, baseline_answer.query_candidate) \
                if baseline_answer is not None else "None"
            print "System: ", (test_answer.prediction, test_answer.query_candidate)\
                if test_answer is not None else "None"
            print "--------"

    print "Diff: %d (wins: %d; loses: %d)" % (diff_count, win_count, loose_count)
    baseline_results, _ = evaluate(baseline_queries)
    test_results, _ = evaluate(test_queries)
    print "Baseline results:\n", baseline_results
    print "System results:\n", test_results

    r = dict()
    for metric in ["avg_precision", "avg_recall", "avg_f1"]:
        r[metric] = [(res[metric][0], res[metric][1]) for res in results.itervalues()]
        print metric, ttest_rel([res[0] for res in r[metric]], [res[1] for res in r[metric]])