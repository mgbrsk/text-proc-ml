import sys
import numpy as np


def parse_array(s):
    return np.array([int(s.strip()) for s in s.strip().split(" ")])


def read_array():
    return parse_array(sys.stdin.readline())


def calculate_pmi(a, b):
    def get_single_proba(x):
        return sum(x) / len(x)

    def get_combine_proba(x, y):
        assert len(a) == len(b)
        return sum([x and y for x, y in zip(a, b)]) / len(a)

    pmi = np.log(get_combine_proba(a, b) / (get_single_proba(a) * get_single_proba(b)))

    return float(round(pmi, 6))
