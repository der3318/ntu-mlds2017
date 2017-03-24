#!/usr/bin/python3
from __future__ import print_function

import sys
import numpy

labels = ["a", "b", "c", "d", "e"]
with open(sys.argv[1], "r") as fin1, open(sys.argv[2], "r") as fin2, open(sys.argv[3], "w") as fout:
    forward = []
    backward = []
    for line in fin1:
        sim = list(map(float, line.split()))
        forward.append(sim)
    for line in fin2:
        sim = list(map(float, line.split()))
        backward.append(sim)
    forward = numpy.array(forward)
    backward = numpy.array(backward)
    total = forward + backward
    fout.write("id,answer\n")
    for i in range(total.shape[0]):
        label = labels[numpy.argmax(total[i, :])]
        fout.write(str(i + 1) + "," + label + "\n")

