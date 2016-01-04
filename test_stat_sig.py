
import cPickle as pickle

from scipy.stats import ttest_rel
from sys import argv


def read_queries(filename):
    with open(filename, 'rb') as inp:
        return pickle.load(inp)

if __name__ == "__main__":
    baseline_queries = read_queries(argv[1])

    results = dict()
    for query in baseline_queries:
        if query.id not in results:
            results[query.id] = {"avg_precision": [0, 0], "avg_recall": [0, 0], "avg_f1": [0, 0]}
        results[query.id]["avg_precision"][0] = query.precision
        results[query.id]["avg_recall"][0] = query.recall
        results[query.id]["avg_f1"][0] = query.f1

    test_queries = read_queries(argv[2])
    for query in test_queries:
        results[query.id]["avg_precision"][1] = query.precision
        results[query.id]["avg_recall"][1] = query.recall
        results[query.id]["avg_f1"][1] = query.f1

    r = dict()
    for metric in ["avg_precision", "avg_recall", "avg_f1"]:
        r[metric] = [(res[metric][0], res[metric][1]) for res in results.itervalues()]
        print metric, ttest_rel([res[0] for res in r[metric]], [res[1] for res in r[metric]])
