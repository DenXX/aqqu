from __future__ import print_function

import cPickle as pickle
import gzip

from sys import argv


if __name__ == "__main__":
    features_file = argv[1]
    labels = []
    features = []
    with gzip.open(features_file, 'r') as inp:
        try:
            index = 0
            while True:
                label, feature = pickle.load(inp)
                labels.append(label)
                features.append(feature)
                index += 1
                if index % 1000 == 0:
                    print("%d instances read" % index)
        except pickle.PickleError:
            pass
