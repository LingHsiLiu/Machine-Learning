Problem 1
You can find a dataset dealing with European employment in 1979 at http://lib.stat.cmu.edu/DASL/Stories/EuropeanJobs.html. This dataset gives the percentage of people employed in each of a set of areas in 1979 for each of a set of European countries. Notice this dataset contains only 26 data points. That's fine; it's intended to give you some practice in visualization of clustering.

1. Use an agglomerative clusterer to cluster this data. Produce a dendrogram of this data for each of single link, complete link, and group average clustering. You should label the countries on the axis. What structure in the data does each method expose? it's fine to look for code, rather than writing your own. Hint: I made plots I liked a lot using R's hclust clustering function, and then turning the result into a phylogenetic tree and using a fan plot, a trick I found on the web; try plot(as.phylo(hclustresult), type='fan'). You should see dendrograms that "make sense" (at least if you remember some European history), and have interesting differences.
2. Using k-means, cluster this dataset. What is a good choice of k for this data and why?

Problem 2
Do exercise 6.2 in the Jan 15 version of the course text

Questions about the homework
1. Can we use linear vector quantization functions like lvqinit, lvqtest, lvq1 available in the 'class' library in R for this exercise?
Answer sure; don't know how well they work for this, as haven't used the package. For this one, it may be simpler to build your own than to understand the package

2. How should we handle test/train splits?
Answer You should not test on examples that you used to build the dictionary, but you can train on them. In a perfect world, I would split the volunteers into a dictionary portion (about half), then do a test/train split for the classifier on the remaining half. You can't do that, because for some signals there are very few volunteers. For each category, choose 20% of the signals (or close!) to be test. Then use the others to both build the dictionary and build the classifier.

3. When we carve up the signals into blocks for making the dictionary, what do we do about leftover bits at the end of the signal?
Answer Ignore them; they shouldn't matter (think through the logic of the method again if you're uncertain about this)
