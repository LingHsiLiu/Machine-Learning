import time

import numpy as np
import matplotlib.pyplot as plt
import pylab
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.vq import kmeans, vq


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


class Problem_1:
    """
    1.  Use an agglomerative clusterer to cluster this data. Produce a dendrogram of this data for each of single link,
        complete link, and group average clustering. You should label the countries on the axis. What structure in the
        data does each method expose? it's fine to look for code, rather than writing your own. Hint: I made plots I
        liked a lot using R's hclust clustering function, and then turning the result into a phylogenetic tree and
        using a fan plot, a trick I found on the web; try plot(as.phylo(hclustresult), type='fan'). You should see
        dendrograms that "make sense" (at least if you remember some European history), and have interesting differences.

    2.  Using k-means, cluster this dataset. What is a good choice of k for this data and why?
    """
    def __init__(self, labels):
        self.seed = np.random.RandomState(seed=3)
        labels = labels.astype('U')
        self.labels = np.char.strip(labels)

    def process(self, data: object):
        data = data.astype(np.float)
        self.data = data
        self.n_samples = len(data)
        self.plot_dendrogram('single')
        self.plot_dendrogram('complete')
        self.plot_dendrogram('average')
        self.plot_kmeans2()

    def plot_kmeans2(self):
        # The best fit appears to be kMeans=3 as it provides 3 separate and distinct groups that
        # are easily segmented and distinct to further explore or analyze difference/similarities

        print('Starting kMeans clustering...')

        idx = np.array([4, 4, 4, 4, 4,
                        4, 4, 4, 4, 4,
                        4, 4, 4, 4, 4,
                        4, 4, 4, 4, 4,
                        4, 4, 4, 4, 4, 4])

        plt.plot(self.data[idx == 0, 0], self.data[idx == 0, 1], 'om',
                 self.data[idx == 1, 0], self.data[idx == 1, 1], 'oc',
                 self.data[idx == 2, 0], self.data[idx == 2, 1], 'oy',
                 self.data[idx == 3, 0], self.data[idx == 3, 1], 'ok',
                 self.data[idx == 4, 0], self.data[idx == 4, 1], 'ob')
        plt.title('Link by - kMeans - Labelled Points')
        for label, x, y in zip(self.labels, self.data[:, 0], self.data[:, 1]):
            plt.annotate(
                label[0],
                xy=(x, y), xytext=((400 + len(label) % 300), (0 + len(label) % 50)),
                textcoords='offset points', ha='right', va='bottom',
                arrowprops=dict(arrowstyle="->", connectionstyle="angle,angleA=0,angleB=90,rad=10"))
        plt.show()

        for i in range(1, 6):
            centroids, _ = kmeans(self.data, i)
            idx, _ = vq(self.data, centroids)

            # plt.plot(data[idx==0,0],data[idx==0,1],'m',
            #   data[idx==1,0],data[idx==1,1],'c')
            plt.plot(self.data[idx == 0, 0], self.data[idx == 0, 1], 'om',
                     self.data[idx == 1, 0], self.data[idx == 1, 1], 'oc',
                     self.data[idx == 2, 0], self.data[idx == 2, 1], 'oy',
                     self.data[idx == 3, 0], self.data[idx == 3, 1], 'ok',
                     self.data[idx == 4, 0], self.data[idx == 4, 1], 'ob')
            plt.plot(centroids[:, 0], centroids[:, 1], 'sg', markersize=8)
            plt.title('Link by - kMeans=' + str(i))

            plt.show();

        print('kMeans clustering complete!')

    def plot_dendrogram(self, clusteringType: str):
        # generate the linkage matrix
        Z = linkage(self.data, clusteringType)
        self.fancy_dendrogram(Z, 0, truncate_mode='lastp',
            leaf_rotation=45., leaf_font_size=8., show_contracted=True,
            annotate_above=10,  # useful in small plots so annotations don't overlap
            title="Hierarchical {0} Clustering Dendrogram".format(clusteringType.capitalize())
        )
        F = pylab.gcf()
        F.set_size_inches((11, 7.5))
        plt.show()

    def fancy_dendrogram(self, *args, **kwargs):
        max_d = kwargs.pop('max_d', None)
        if max_d and 'color_threshold' not in kwargs:
            kwargs['color_threshold'] = max_d
        annotate_above = kwargs.pop('annotate_above', 0)
        titleText = kwargs.pop('title')
        kwargs['labels'] = self.labels

        fig = plt.figure(figsize=(25, 10))
        ddata = dendrogram(*args, **kwargs)
        if not kwargs.get('no_plot', False):
            plt.title(titleText)
            plt.xlabel('Countries')
            plt.ylabel('distance')
            for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
                x = 0.5 * sum(i[1:3])
                y = d[1]
                if y > annotate_above:
                    plt.plot(x, y, 'o', c=c)
                    plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                                 textcoords='offset points',
                                 va='top', ha='center')
            if max_d:
                plt.axhline(y=max_d, c='k')
        return ddata


