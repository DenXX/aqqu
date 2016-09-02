"""
Base class to represent answer candidate. QueryCandidate now inherits from AnswerCandidate. This was needed
to support candidates generated from other data sources than KB.

Copyright 2016, Emory University

Denis Savenkov <denis.savenkov@emory.edu>
"""
import logging
from abc import ABCMeta, abstractmethod

logger = logging.getLogger(__name__)


class AnswerCandidate:
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @abstractmethod
    def get_results_text(self):
        pass

if __name__ == "__main__":
    pass