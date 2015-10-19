import json
import sys
import codecs
import math
from random import shuffle

split_ratio=(8, 2)

if __name__ == "__main__":
    with codecs.open(sys.argv[1], 'r') as input_file:
        questions = json.load(input_file)
    shuffle(questions)
    split_point = int(math.ceil(1.0 * len(questions) * split_ratio[0] / sum(split_ratio)))
    train = questions[:split_point]
    dev = questions[split_point:]
    with codecs.open(sys.argv[2], 'w', 'utf-8') as out:
        json.dump(train, out)
    with codecs.open(sys.argv[3], 'w', 'utf-8') as out:
        json.dump(dev, out)
