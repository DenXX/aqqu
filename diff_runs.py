
import cPickle as pickle
import json
import sys

import operator
from scipy.stats import ttest_rel
from sys import argv

from query_translator.evaluation import evaluate


questions_filter = set(["in which state was the battle of antietam fought?", "what all does google now do?", "what are aristotle's contributions to science?", "what are bigos?", "what are republicans views on health care?", "what are the official languages of the eu?", "what are the school colors for harvard university?", "what are the supreme court cases?", "what awards did marilyn monroe won?", "what channel is the usa pageant on?", "what colleges did albert einstein teach at?", "what company does nike own?", "what counties in florida have the lowest property taxes?", "what countries does the arctic circle run through?", "what countries don need a visa for usa?", "what country does alaska belong to?", "what country does rafael nadal play for?", "what country was slovakia?", "what did african americans do during the revolutionary war?", "what did elliot stabler do?", "what did john hancock do for the american revolution?", "what did thomas hobbes do?", "what discovery did galileo make?", "what do christians believe about heaven hell and purgatory?", "what does bolivia border?", "what does david beckham play?", "what does egfr african american mean on a blood test?", "what does god shiva represent?", "what fma stands for?", "what genre of art is the mona lisa?", "what honor did agatha christie receive in 1971?", "what is berkshire hathaway invested in?", "what is john edwards indicted for?", "what is monta ellis career high points?", "what is newcastle metro?", "what is real name of santa claus?", "what is the capital of australia victoria state?", "what is the government of france for 2010?", "what is the new movie john carter about?", "what is the theme of scarlet letter by nathaniel hawthorne?", "what is the university of georgia known for?", "what kind of currency does mexico use?", "what led to the split of the republican party in 1912?", "what money to take to turkey?", "what movies did joan crawford play in?", "what movies did morgan freeman star in?", "what objects did galileo see with his telescope?", "what position did stanley matthews play?", "what products and\/or services does google offer customers?", "what radio station is npr on in nyc?", "what team did david beckham play for before la galaxy?", "what team does mike fisher play for?", "what team is raul ibanez on?", "what time does registration open portland state?", "what type of guitar does johnny depp play?", "what vegetables can i plant in november in southern california?", "what war did the us lose the most soldiers?", "what was firefox programmed in?", "what was francis bacon contributions?", "what was john deere famous for?", "what was scottie pippen known for?", "what works of art did leonardo da vinci produce?", "what year did super mario bros 2 come out?", "what year was kenya moore crowned miss usa?", "when did kelly slater go pro?", "when did liverpool fc last win the champions league?", "when did mark mcgwire retired?", "when did michael vick start playing for the eagles?", "when did the burma cyclone happen?", "when did we start war with iraq?", "when did william mckinley died?", "when do world war ii end?", "when is nrl grand final day?", "when is saint george day celebrated?", "when is venus brightest?", "when tupac was shot?", "when was john paul ii?", "when was taylor swift fearless tour?", "when was the last tsunami in the atlantic ocean?", "when was the most recent earthquake in haiti?", "where can you go on eco holidays in the uk?", "where did andy murray started playing tennis?", "where did kennedy's inaugural address take place?", "where did kobe earthquake happen?", "where did starbucks get their logo?", "where does airtran airways fly?", "where does toronto get its water from?", "where do ireland play rugby union?", "where do they speak tibetan?", "where is chesapeake bay bridge?", "where was the ottoman empire based?", "where were the great pyramids of giza built?", "which country does greenland belong to?", "who are the st louis cardinals coaches?", "who did armie hammer play in the social network?", "who did george w. bush run against for the second term?", "who did the chargers draft in 2011?", "who does lee clark manager?", "who influenced samuel taylor coleridge?", "who invented islamic religion?", "who is in the american league in baseball?", "who is the coach of the sf giants?", "who is the head coach of inter milan?", "who led the mexican army at the battle of the alamo?", "who made the laws in canada?", "who played denver in four christmases?", "who plays ponyboy in that was then this is now?", "who was with president lincoln when he was assassinated?", "who will plaxico burress play for in 2011?", "who wrote st trinians?", "who wrote the gospel according to john?", "who wrote the jana gana mana?"])

def read_queries(filename):
    print >> sys.stderr, "Reading queries from %s" % filename
    with open(filename, 'rb') as inp:
        return pickle.load(inp)

def diff_pickles(baseline_path, system_path):
    baseline_queries = read_queries(baseline_path)

    results = dict()
    answers = dict()
    id2questionanswer = dict()
    for query in baseline_queries:
        # if query.utterance not in questions_filter:
        #     continue

        id2questionanswer[query.id] = (query.utterance, query.target_result)
        if query.id not in results:
            results[query.id] = {"avg_precision": [0, 0], "avg_recall": [0, 0], "avg_f1": [0, 0]}
        results[query.id]["avg_precision"][0] = query.precision
        results[query.id]["avg_recall"][0] = query.recall
        results[query.id]["avg_f1"][0] = query.f1
        answers[query.id] = [query.eval_candidates[0] if query.eval_candidates else None, None]

    test_queries = read_queries(system_path)
    for query in test_queries:
        # if query.utterance not in questions_filter:
        #     continue

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


def compute_f1(answer, correct_answer):
    answer = set(answer)
    correct_answer = set(correct_answer)
    intersection = len(answer.intersection(correct_answer))
    precision = (1.0 * intersection / len(answer)) if len(answer) > 0 else 1
    recall = (1.0 * intersection / len(correct_answer)) if len(correct_answer) > 0 else 0
    return (2.0 * precision * recall / (precision + recall)) if precision > 0 or recall > 0 else 0.0


def merge_answers(baseline_answer, system_answer):
    baseline_answer = set(baseline_answer)
    system_answer = set(system_answer)
    if len(baseline_answer) == 0:
        return system_answer
    if len(system_answer) == 0:
        return baseline_answer
    return system_answer if len(baseline_answer) > len(system_answer) else baseline_answer


def diff_text_pickle(text_results, pickle_results):
    answers = dict()
    with open(text_results, 'r') as inputfile:
        for line in inputfile:
            question, answer = line.strip().split("\t")
            answer = json.loads(answer)
            answers[question] = answer
    system_queries = read_queries(pickle_results)
    diff = []
    system_average_f1 = []
    baseline_average_f1 = []
    average_combined_f1 = []
    for query in system_queries:
        if query.utterance not in questions_filter:
            continue

        question = query.utterance
        answer = query.eval_candidates[0].prediction if query.eval_candidates else []
        baseline_f1 = compute_f1(answers[question], query.target_result)
        system_f1 = compute_f1(answer, query.target_result)
        combined_answer = list(merge_answers(answers[question], answer))
        # combined_answer = answers[question] if baseline_f1 > system_f1 else answer
        print "\t".join(map(str, [question, answers[question], baseline_f1, answer, system_f1,
                                  query.target_result, baseline_f1 - system_f1, combined_answer]))
        # print question + "\t" + json.dumps(combined_answer)
        combined_f1 = compute_f1(combined_answer, query.target_result)
        diff.append(baseline_f1 - system_f1)
        system_average_f1.append(system_f1)
        baseline_average_f1.append(baseline_f1)
        average_combined_f1.append(combined_f1)
    print >> sys.stderr, sorted(diff)
    print >> sys.stderr, sum(baseline_average_f1) / len(baseline_average_f1),\
        sum(system_average_f1) / len(system_average_f1), sum(average_combined_f1) / len(average_combined_f1), ttest_rel(baseline_average_f1, system_average_f1), len(baseline_average_f1)


def pickle_to_text(results_file):
    queries = read_queries(results_file)
    for query in queries:
        answer = query.eval_candidates[0].prediction if query.eval_candidates else []
        print query.utterance.encode("utf-8") + "\t" + json.dumps(answer)


def print_all_non_perfect(system_queries):
    queries = read_queries(system_queries)
    data = []
    for query in queries:
        if query.f1 < 1.0:
            answer = query.eval_candidates[0].prediction if query.eval_candidates else []
            query_candidate = query.eval_candidates[0].query_candidate if query.eval_candidates else ""
            correct_candidate = ""
            correct_f1 = 0
            if query.oracle_position > 0:
                correct_candidate = query.eval_candidates[query.oracle_position - 1].query_candidate
                correct_f1 = query.eval_candidates[query.oracle_position - 1].evaluation_result.f1
            data.append((query.utterance, query.target_result, answer, query_candidate, query.f1, correct_candidate, correct_f1))
    data.sort(key=operator.itemgetter(4))
    for rec in data:
        print "\t".join(map(str, rec))


if __name__ == "__main__":
    diff_pickles(argv[1], argv[2])
    # diff_text_pickle(argv[1], argv[2])
    # pickle_to_text(argv[1])
    # print_all_non_perfect(argv[1])
