from datetime import datetime
from os import path
from os import walk
import time
import random

import numpy as np
import pandas as pd
import xarray as xr

from problem1 import Problem_1
from problem2works import Problem_2
# from problem2noworking import Problem_2

# returns message of the elapsed time on 2nd call
def timeit():
    if timeit.start == 0:
        timeit.start = time.monotonic()
        return

    # Determine if elapsed time is in seconds or minutes
    h = 0
    m, s = divmod((time.monotonic() - timeit.start), 60)
    if m > 0:
        h, m = divmod(m, 60)

    m = int(m)
    h = int(h)
    time.start = 0

    if h == 0:
        if m == 0:
            msg = '{:.2f} seconds'.format(s)
        else:
            msg = '{}:{:.2f} minutes'.format(m, s)
    else:
        msg = '{}:{}:{:.2f} hours'.format(h, m, s)

    return msg


setattr(timeit, 'start', 0)


def random_subset( iterator, K ):
    result = []
    N = 0

    for item in iterator:
        N += 1
        if len(result) < K:
            result.append(item)
        else:
            s = int(random.random() * N)
            if s < K:
                result[s] = item

    return result


def load_data2():
    # Problem 2:  Exercise 6.2 from http://luthuli.cs.uiuc.edu/~daf/courses/AML-18/learning-book-15-Jan.pdf
    print('Loading data for Problem 2...')

    skip1stdir = True
    # Remove _MODEL folders as they appear to be subsets of the main data folders
    excludeDirs = ['Climb_stairs_MODEL', 'Pour_water_MODEL', 'Drink_glass_MODEL', 'Getup_bed_MODEL',
                   'Sitdown_chair_MODEL', 'Standup_chair_MODEL', 'Walk_MODEL']

    # Iterate through all data sets
    columns = ['HMP', 'ActivityTS', 'Gender', 'GenderID', 'ActivityArray']
    arrayColumns = ['x', 'y', 'z']
    ds = xr.Dataset()
    coords = []
    for root, dirs, files in walk(".\data\ADL_HMP_Dataset", topdown=True):
        dirs[:] = [d for d in dirs if d not in excludeDirs]
        if skip1stdir:  # skip first directory level
            skip1stdir = False
            continue

        hmp = path.basename(root)
        coordList = []

        for name in files:
            coord = {}
            # Gather HMP data
            HMPdata = name.split('-', 1)[1].split('-')
            coord['ActivityTS'] = datetime(int(HMPdata[0]), int(HMPdata[1]), int(HMPdata[2]), int(HMPdata[3]),
                                           int(HMPdata[4]), int(HMPdata[5]))
            coord['Gender'] = HMPdata[7][0]
            coord['GenderID'] = HMPdata[7][1]

            # Load data in file to array
            fp = path.join(root, name)
            coord['Activities'] = pd.read_table(fp, delim_whitespace=True, names=arrayColumns)
            coordList.append(coord)

        coords.append((hmp, coordList))

    trcords = []
    for hmp in coords:
        training = []
        sample = round(len(hmp[1]) * .2)
        samples = random_subset(list(range(0,len(hmp[1]))), sample)
        samples.sort(reverse=True)
        [training.append(hmp[1].pop(idx)) for idx in samples]
        trcords.append((hmp[0], training))

    print('Problem 2 data load complete!')
    return coords, trcords


def load_data():
    """
    Loads the data from:
        Problem 1: Outlined on webpage: http://lib.stat.cmu.edu/DASL/Datafiles/EuropeanJobs.html
        Problem 2: https://archive.ics.uci.edu/ml/datasets/Dataset+for+ADL+Recognition+with+Wrist-worn+Accelerometer
    """
    print("Loading Data...")

    fnpath = "data\\"
    fnlabels = 'variablenames.csv'
    fndata = 'europeanjobs.csv'

    labels = pd.read_csv(path.join(fnpath, fnlabels), names=['Industries', 'Description'])
    data = pd.read_csv(path.join(fnpath, fndata))
    return labels, data


# Begin Processing
# Problem 1
np.set_printoptions(precision=5, suppress=True)  # suppress scientific float notation
labelData, jobData = load_data()
p1 = Problem_1(jobData.values[:, 0])
# p1.process(jobData.values[:, 1:])

#Problem 2
data, train = load_data2()
p2 = Problem_2()
p2.process(data, train)

print("Debug")