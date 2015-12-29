import globals
import sys

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Dump qa entity pairs.")
    parser.add_argument("--config",
                        default="config.cfg",
                        help="The configuration file to use.")
    args = parser.parse_args()
    globals.read_configuration(args.config)
    sparql_backend = globals.get_sparql_backend(globals.config)
    while True:
        print "Please enter query: "
        query_str = ""
        while True:
            query_str_line = sys.stdin.readline()
            if query_str_line.startswith("END"):
                break
            query_str += query_str_line
        print sparql_backend.query(query_str)