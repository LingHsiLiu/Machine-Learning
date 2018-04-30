import os
import time

import numpy as np
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
import pylab
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.vq import kmeans, vq
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

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


class Problem_2:
    """ From the course book lesson 6.2
    Obtain the activities of daily life dataset from the UC Irvine machine learning website (Data provided
    by Barbara Bruno, Fulvio Mastrogiovanni and Antonio Sgorbissa):

        https://archive.ics.uci.edu/ml/datasets/Dataset+for+ADL+Recognition+with+Wrist-worn+Accelerometer

    (a) Build a classifier that classifies sequences into one of the 14 activities provided. To make features, you
        should vector quantize, then use a histogram of cluster centers (as described in the subsection; this gives a
        pretty explicit set of steps to follow). You will find it helpful to use hierarchical k-means to vector quantize.
        You may use whatever multi-class classifier you wish, though I’d start with R’s decision forest, because it’s easy
        to use and effective. You should report (a) the total error rate and (b) the class confusion matrix of your
        classifier.
    (b) Now see if you can improve your classifier by (a) modifying the number of cluster centers in your hierarchical
        k-means and (b) modifying the size of the fixed length samples that you use.
    """
    def __init__(self):
        self.seed = np.random.RandomState(seed=3)

    def create_seg(self, values, slicepoint, segsize):
        seg = np.asarray(values[slicepoint:slicepoint + segsize, :])
        return seg.astype(dtype=np.float)

    def prepareActivity(self, act):
        result = []
        segList = [self.create_seg(act['Activities'].values, slicepoint, self.segsize)
                   for slicepoint in range(0, len(act['Activities']), self.segsize)]
        if len(segList[-1]) < self.segsize:
            segList = segList[:len(segList) - 1]

        if len(segList) >= self.samplesize:
            result = [seg.reshape(1, self.segsize * 3) for seg in segList]
        return result

    def calcKMeans(self, data):
        if len(data) == 0: return None
        kmeans = KMeans(n_clusters=self.samplesize)
        centoids = {}
        for key, val in data.items():
            kmeans.fit(np.squeeze(val))
            centoids[key] = kmeans.labels_

        return centoids

    def histogram(self, centoids):
        histData = {}
        for key, val in centoids.items():
            cnts, _ = np.histogram(val, self.samplesize)
            histData[key] = cnts
        return histData

    def prepareData(self, hmps):
        accData = {}
        for hmp in hmps:
            actdata = []
            for act in hmp[1]:
                if len(act['Activities']) / 32 < self.samplesize: continue
                [actdata.append(seg) for seg in self.prepareActivity(act)]
                if len(actdata) == 0:
                    continue
                if len(actdata[-1]) == 0:
                    actdata.remove(-1)

            accData[hmp[0]] = actdata

        return accData

    def naivebayes(self, trCNTS, dataCNTS):
        nb = GaussianNB()
        nb.fit(list(trCNTS.values()), list(trCNTS.keys()))

        cls = []
        for hmp in list( dataCNTS.values() ):
            for lst in hmp:
                for act in list( lst.values() ):
                    act = np.append(act, list( lst.keys() )[0])
                    cls.append(act)

        print("Debug")

        cls = np.array(cls)
        y_pred = nb.predict(cls[:, :-1].astype(int))
        y_true = cls[:, -1:]
        cnf_mat = confusion_matrix(y_true, y_pred)
        score = nb.score(cls[:, :-1].astype(int), cls[:, -1:])
        print("debug")

    def randomForest(self, X_train, y_train, X_Test, y_Test):

        clf = RandomForestClassifier(n_estimators=250, max_leaf_nodes=16, n_jobs=-1)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_Test)

        # y_true = cls[:, -1:]
        cnf_mat = confusion_matrix(y_Test, y_pred)
        score = clf.score(X_Test, y_Test)
        print("Debug")

    def process(self, data, trdata):
        self.samplesize = 10  # it appears kmeans is doing sampling at the segment level which is 32x3 means a max of 32 samples
        self.segsize = 8

        accData = self.prepareData(trdata)
        centoids = self.calcKMeans(accData)
        trCNTS = self.histogram(centoids)

        dataCENTOIDS = {}
        for hmp in data:
            actCENTOIDS = []
            for act in hmp[1]:
                if len(act['Activities']) / 32 < self.samplesize: continue
                centoids = self.calcKMeans({hmp[0]: self.prepareActivity(act)})
                actCENTOIDS.append(np.array(list(centoids.values())).T)
            dataCENTOIDS[hmp[0]] = actCENTOIDS

        print("Debug")
        #Part A
        self.naivebayes(trCNTS, self.histogram(dataCENTOIDS))

        #Part B
        cls = []
        y_labels = []
        for key, hmp in list(dataCENTOIDS.items()):
            for act in hmp:
                y_labels.append(key)
                cls.append(act)
        y_test = np.array(cls)

        self.randomForest(list(trCNTS.values()), list(trCNTS.keys()), y_test, y_labels)

        print("Debug")
    #END of process()
