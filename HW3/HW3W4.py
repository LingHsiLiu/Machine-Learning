from os import path
from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn.metrics import euclidean_distances
from sklearn import manifold

import pickle
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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


def unpickle(file) -> dict:
    with open(file, 'rb') as fo:
        d = pickle.load(fo, encoding='bytes')

    result = defaultdict(list)
    for key in d:
        result["".join(map(chr, key))] = d[key]

    return result


def load_data():
    """
    Loads the CIFAR-10-batches-py data
    to global variables labelNames, testBatch and
    the list of batchData.
    """
    global labelNames
    print("Loading Data...")

    fnpath = "rawdata\\cifar-10-batches-py"
    fnprefix = 'data_batch_'
    fnlblnames = 'batches.meta'
    fntstbatch = 'test_batch'

    labelNames = unpickle(path.join(fnpath, fnlblnames))
    label_names = []
    for label in labelNames['label_names']:
        label_names.append("".join(map(chr, label)))
    labelNames['label_names'] = label_names

    CIFAR_Data.append(unpickle(path.join(fnpath, fntstbatch)))
    for n in range(1, 6):
        CIFAR_Data.append(unpickle(path.join(fnpath, fnprefix + str(n))))


def categorize_data(files: list, labels: dict):
    print("Categorizing data...")
    keys = labels['label_names']
    temp = dict((key, []) for key in labels['label_names'])

    for images in files:
        imageCNT = len(images['labels'])
        for i in range(imageCNT):
            key = keys[images['labels'][i]]
            temp[key].append(images['data'][i])

    result = defaultdict(np.ndarray)
    for k, v in temp.items():
        result[k] = np.array(v)

    return result


def rotate(X, angle):
    theta = np.deg2rad(angle)
    R = [[np.cos(theta), np.sin(theta)],
         [-np.sin(theta), np.cos(theta)]]
    return np.dot(X, R)


def scatterPlot(x, y, labels):
    plt.scatter(x, y)
    for label, x, y in zip(labels, x, y):
        plt.annotate(label, xy=(x, y), xytext=(-1, 1),
            textcoords='offset points', ha='right', va='bottom')


def Part_1(catData: dict):
    # Original Instructions:
    #   resulting from representing the images of each category using the first 20 principal components
    #   against the category.

    # @551 Revised Instructions:
    #   The error you want to find is either of the following two figures. They should be about the same value,
    #   but one or the other might be easier for you to compute, depending on what functions you're using:

    #   Option 1: The sum of the unused eigenvalues for a given category (the ones not corresponding to the 20
    #             major principal components). This assumes that your PCA calculated all of the eigenvalues,
    #             not just the major ones.

    #   Option 2: First reconstruct low-dimensional representations for all of the images in a given category. Then,
    #             for each image, find the per-pixel differences between the reconstruction and its original image.
    #             Square those differences individually and sum them over the whole image. Finally, calculate the
    #             mean of this value over the whole category. (Notice that this technique doesn't use the eigenvalues
    #             at all, just the eigenvectors. Also you'll need to do something like this again in Part 3 anyway.)

    print("Begin Part 1 PCA analysis...")
    timeit()
    fname = "part_1.dat"
    errorList = defaultdict(float)
    results = defaultdict()

    if path.exists(fname):
        with open(fname, 'rb') as fo:
            results = pickle.load(fo, encoding='bytes')
        for key, X in results.items():
            errorList[key] = X.explained_variance_[20:].sum()
    else:
        for key, data in catData.items():
            X = PCA().fit(data)
            results[key] = X
            errorList[key] = X.explained_variance_[20:].sum()

        with open(fname, 'wb') as fo:
            pickle.dump(results, fo)

    print("Analysis processing: " + timeit())

    if Part1PlotEnabled:
        xpos = np.arange(10)  # the x locations
        y = errorList.values()

        plt.figure(figsize=(10, 5))
        plt.bar(xpos, y, align='center', alpha=0.5)
        plt.xticks(xpos, errorList.keys())
        plt.xlabel('Classes')
        plt.ylabel('Error Variance')
        plt.title("Variance of non principal components")

        plt.show()

    return errorList, results


def Part_2(catData: dict):
    # Original Instructions:
    #   Compute the distances between mean images for each pair of classes. Use principal coordinate analysis
    #   to make a 2D map of the means of each categories. For this exercise, compute distances by thinking of
    #   the images as vectors.

    # @551 Revised Instructions:
    #   [Update Feb 16: We would ask that you please include the 10x10 distance matrix with your submission. This
    #                   will aid grading in case your plotting package does something unusual.]

    #   You are performing MDS on the mean images for this part. The mean images are the same as you calculated
    #   in part 1. Your scatter plot should look similar to Figure 4.16 (upper-right of page 83 in the textbook).
    #   Please label the points or make a legend so you can tell which category point is which.

    #   About calculating MDS: There are libraries that can find this for you based on a distance measurement or
    #   distance matrix you specify. A distance matrix is a square matrix of pairwise distances. See also:
    #       https://en.wikipedia.org/wiki/Distance_matrix

    #   The distance measurement for Part 2 should be the Euclidean distance (that is, between two image vectors,
    #   you find the squared root of the sum of squared differences); this is the same as the L2 norm.

    #   If you're calculating MDS manually, see Procedure 4.3 on page 82. It has a typo; it should have
    #   W = −1/2(AD(2)A^T. This W is the same as the M described in section 4.4.2. See also the mathematical
    #   discussion at @452. The matrix A is the centering matrix; it's defined on page 80. See also:
    #       https://en.wikipedia.org/wiki/Centering_matrix

    #   Your MDS plots should be pretty consistent between runs, but some libraries might randomize parts of the
    #   calculation when performing PCA, and furthermore solving the low-dimensional representation may result in
    #   rotation or distortion sometimes. If you want to regulate the output to compare different methods better, you
    #   could try this with some linear algebra tricks [optional]: Offset the results so that category 1 is at the
    #   origin; then rotate and scale the results so that category 2 is at (1,0); then, if necessary, scale the
    #   results so that category 3 has a positive x coordinate. Even if you don't do this, you should be able to
    #   turn your head or flip the image and see consistent results.
    print("Begin Part 2 MDS analysis...")
    timeit()

    seed = np.random.RandomState(seed=3)
    X_mean = defaultdict(np.ndarray)
    XLabels = list(catData.keys())
    fig = plt.figure(figsize=(8, 3))

    # Calculate the category mean
    for key, X in catData.items():
        X = X.mean(axis=0)
        X_mean[key] = X

    # convert to numpy array
    X_array = np.array(list(X_mean.values()))

    D = euclidean_distances(X_array)

    mds = manifold.MDS(n_components=10, random_state=seed, dissimilarity="precomputed")
    pos = mds.fit(D).embedding_
    pca = PCA(n_components=10).fit(pos)
    pca2D = pca.transform(pos)

    print("First fit processing time: " + timeit())

    if Part2PlotEnabled:
        scatterPlot(pca2D[:, 0], pca2D[:, 1], XLabels)
        plt.show()


def cache_calcs(key: str, data: np.ndarray):
    """
    :param key: str
    :param data: np.ndarray
    :return: (pca, mean, centered)
    """
    if key not in cached_calcs:
        mean_of_key = data.mean(axis=0)
        centered = data - mean_of_key
        pca = PCA(n_components=20).fit(data)
        cached_calcs[key] = (pca, mean_of_key, centered)

    return cached_calcs[key]


def pca_of_A_using_B(pca_of_A: PCA, A_centered: np.ndarray, key_of_B: str, B: np.ndarray):
    """
    Images of A using mean of A and first 20 principal components of B
    Formula: x_hat = mean(A) + sum( B_pc_eigenvectors * (A_of_i - mean(A)) * U_j
    New Formula
    :return:
    """
    pca_of_B, mean_of_B, B_centered = cache_calcs(key_of_B, B)
    results = []
    for i in np.arange(0, 20):
        results.append(pca_of_A.components_[i] * A_centered * pca_of_B.components_[i])

    return sum(results)


def similarity(A: str, B: str, data: dict, Acentered: np.ndarray, pca_of_A: PCA):
    """
    Calculate 1/2 * (E(A|B) + E(B|A)
    :param A: str
    :param B: str
    :param data: dict
    :param Acentered:
    :param pca_of_A:
    :return: sim: float
    """
    # Calculate E(A|B)
    A_by_B = pca_of_A_using_B(pca_of_A, Acentered, B, data[B])
    A_sqrd = np.power(A_by_B - data[A], 2)
    EAB = A_sqrd.mean(axis=0)

    # Calculate E(B|A)
    pca_of_B, mean_of_B, B_centered = cache_calcs(B, data[B])
    B_by_A = pca_of_A_using_B(pca_of_B, B_centered, A, data[A])
    B_sqrd = np.power(B_by_A - data[B], 2)
    EBA = B_sqrd.mean(axis=0)

    # Calculate similarity
    sim = 1 / 2 * (EAB + EBA)
    return sim


def Part_3(catData: dict):
    """
        Original Instructions:
        Here is another measure of the similarity of two classes. For class A and class B, define E(A | B) to be
        the average error obtained by representing all the images of class A using the mean of class A and the
        first 20 principal components of class B. Now define the similarity between classes to
        be (1/2)(E(A | B) + E(B | A)). If A and B are very similar, then this error should be small,
        because A's principal components should be good at representing B. But if they are very different, then
        A's principal components should represent B poorly. In turn, the similarity measure should be big. Use
        principal coordinate analysis to make a 2D map of the classes. Compare this map to the map in the
        previous exercise? are they different? why?

        [Update Feb 16: We would ask that you please include the 10x10 similarity distance matrix with your
        submission. This will aid grading in case your plotting package does something unusual.]

        This time we'll do MDS with a different kind of distance measurement. Again, please label the points
        in your scatterplot so we can see which is which.

        We use a measurement of similarity that comes from swapping principal components between categories,
        and it's defined formally in the instructions. Note that high similarity values mean "not similar"
        and low values mean "quite similar". (Perhaps we should have called it "dissimilarity"?) "Class"
        means "image category". When it refers to "class A" and "class B", it means for any pair of
        categories, call one A and one B. You do need to process all pairs of categories. Notice that
        the calculation method is similar to the "Option 2" stated above for Part 1. When you are taking the
        difference, it is between the original category A image, and the version of the category A image
        reconstructed using B's principal components.

        A common question was whether the diagonal of the distance matrix should be forced to 0 or not. No,
        just calculate the E(A|A) value according to the formula. Note that E(A|A) ends up just being the
        ordinary reconstruction error for a class, as explained above.

        I was able to drastically speed up this part by being careful about which order to multiply matrices.
        Hint: Don't create a big fat matrix as an intermediate product if you don't have to. Try to group
        multiplications so that you work with longer, thinner matrices. Be sure to read the "Speed and memory"
        part at the top of this post. This is another good take-away skill from the homework. [Optional napkin
        puzzle] To illustrate, what would be the smartest way to multiply these matrices?

        Yes, uj is an eigenvector from B. What this math is doing, is projecting the mean-centered image from
        A onto this axis of B's principal component space.

        The dot product part tests how "aligned" the A image is with B's axis. That gives you a scalar. Then you
        multiply B's axis by this scalar and it gives you a point in B's space along that axis. The process
        forcefully realigns and projects points from A's subspace into B's subspace.
    """
    print("Begin Part 3 Similarity analysis...")
    timeit()
    seed = np.random.RandomState(seed=3)
    keys = list(catData.keys())

    # TODO re-setup Loops to calculate every combination then average the categories

    similarities = dict((key, np.zeros((9, 3072))) for key in keys)
    nA = 0
    nB = 0
    for i in np.arange(0, len(keys)):
        A = keys[i]
        pca_of_A, mean_of_A, A_centered = cache_calcs(A, catData[A])
        # setup up B Loop
        Bkeys = set(keys)
        [Bkeys.discard(key) for key in keys[:i+1]]
        for B in Bkeys:
            sim = similarity(A, B, catData, A_centered, pca_of_A)
            similarities[A][nA] = sim
            similarities[B][nB] = sim
            nA += 1
        nB += 1
        nA = nB

    # calculate the mean of the sample runs
    sim_array = np.zeros((10, 3072))
    i = 0
    for samples in list(similarities.values()):
        sim_array[i] = samples.mean(axis=0)
        i += 1

    D = euclidean_distances(sim_array)
    mds = manifold.MDS(n_components=10, random_state=seed, dissimilarity="precomputed")
    pos = mds.fit(D).embedding_
    pca = PCA(n_components=10).fit(pos)
    pca2D = pca.transform(pos)

    print("First fit processing time: " + timeit())

    if Part3PlotEnabled:
        scatterPlot(pca2D[:, 0], pca2D[:, 1], list(catData.keys()))
        plt.show()

    print("debug")


# Begin Processing
plt.rcdefaults()
Part1PlotEnabled = True
Part2PlotEnabled = True
Part3PlotEnabled = True
labelNames = {}
CIFAR_Data = []

load_data()
CAT_Data = categorize_data(CIFAR_Data, labelNames)

errors, analysis = Part_1(CAT_Data)
Part_2(CAT_Data)

cached_calcs = pd.Series(None)
Part_3(CAT_Data)



print("Debug")
