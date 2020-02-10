import os
import sys
import math


def createFolder(path):
    if not os.path.isdir(path):
        try:
            os.mkdir(path)
        except OSError:
            sys.exit("\nCreation of the directory %s failed" % path)
        else:
            print("\nSuccessfully created the directory %s " % path)
    else:
        print("\nDirectory %s already existing." % path)


def linear_euclidean_distance(a, b):
    return abs(a - b)


def D2_euclidean_distance(a, b):
    return math.sqrt(((a[0] - b[0]) ** 2) + ((a[1] - b[1]) ** 2))
