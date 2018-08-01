from math import sqrt
import sys


def euclid_dist(loc1, loc2):
    return sqrt((loc1[0] - loc2[0]) ** 2 + (loc1[1] - loc2[1]) ** 2)


def print_progress(n):
    sys.stdout.write(str(n))
    sys.stdout.write('\r')
    sys.stdout.flush()
