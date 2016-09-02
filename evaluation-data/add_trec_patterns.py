
import json
from sys import argv

if __name__ == "__main__":
    input_patterns = argv[1]
    input_json = argv[2]

    patterns = dict()
    with open(input_patterns, 'r') as inp:
        for line in inp:
            line = line.strip().split()
            id = int(line[0])
            pattern = ' '.join(line[1:])
            if id not in patterns:
                patterns[id] = []
            patterns[id].append(pattern)

    questions = json.load(open(input_json, 'r'))
    for q in questions:
        id = int(q['id'])
        q['patterns_raw'] = patterns[id]
    json.dump(questions, open(argv[3], 'w'), indent=2)