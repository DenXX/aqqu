"""
Provides a console based translation interface.

Copyright 2015, University of Freiburg.

Elmar Haussmann <haussmann@cs.uni-freiburg.de>

"""
import logging
import globals
import scorer_globals
import sys
from query_translator.translator import SparqlQueryTranslator, CandidateGenerator

logging.basicConfig(format="%(asctime)s : %(levelname)s "
                           ": %(module)s : %(message)s",
                    level=logging.INFO)

logger = logging.getLogger(__name__)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Console based translation.")
    parser.add_argument("ranker_name",
                        default="WQ_Ranker",
                        help="The ranker to use.")
    parser.add_argument("--config",
                        default="config.cfg",
                        help="The configuration file to use.")
    args = parser.parse_args()
    globals.read_configuration(args.config)
    scorer_globals.init()
    if args.ranker_name not in scorer_globals.scorers_dict:
        logger.error("%s is not a valid ranker" % args.ranker_name)
        logger.error("Valid rankers are: %s " % (" ".join(scorer_globals.scorers_dict.keys())))
    logger.info("Using ranker %s" % args.ranker_name)
    ranker = scorer_globals.scorers_dict[args.ranker_name]
    translator = CandidateGenerator.get_from_config(globals.config)
    translator.set_scorer(ranker)
    while True:
        try:
            sys.stdout.write("enter question> ")
            sys.stdout.flush()
            query = sys.stdin.readline().strip()
            logger.info("Translating query: %s" % query)
            results = translator.translate_and_execute_query(query)
            logger.info("Done translating query: %s" % query)
            logger.info("#candidates: %s" % len(results))
            logger.info("------------------- Candidate features ------------------")
            for rank, result in enumerate(results[:10]):
                logger.info("RANK " + str(rank))
                logger.info(result.query_candidate.relations)
                logger.info(result.query_candidate.get_results_text())
                if result.query_candidate.features:
                    logger.info("Features: " + str(result.query_candidate.features))
            logger.info("---------------------------------------------------------")
            if len(results) > 0:
                best_candidate = results[0].query_candidate
                sparql_query = best_candidate.get_candidate_query()
                result_rows = results[0].query_result_rows
                result = []
                # Usually we get a name + mid.
                for r in result_rows:
                    if len(r) > 1:
                        result.append("%s (%s)" % (r[1], r[0]))
                    else:
                        result.append("%s" % r[0])
                logger.info("SPARQL query: %s" % sparql_query)
                logger.info("Result: %s " % " ".join(result))
        except Exception as e:
            logger.error(e.message)


if __name__ == "__main__":
    main()
