
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def cent_distance(a, b):
    # Use NumPy linear algebra functions to determine the distance
    return abs(np.linalg.norm(a - b))

# This method returns assignments of each point in the dataset to 0 or 1 based on the
# centroid that the point is closer to.
def closestCentroid(x, centroids):
    # assignment array initialized
    assignments = []
    # for each item in input toy dataset iterate
    for i in x:
        # distance array initialized
        distance=[]
        #  for each centroid calculate the distance between each dataset point and centroid and append to distance array
        for j in centroids:
            distance.append(cent_distance(i, j))

        # add the distance array to the assignment array
        assignments.append(np.argmin(distance))

    return np.array(assignments)

# This method determines the new centroid value based on the input dataset and the current cluster assignment
def updateCentroid(x, clusters, K):
    # initialize the new centroid point to blank
    new_centroids = []
    for c in range(K):
        # For each centroid value, update the cluster centroid with the average of all points in this cluster based
        # current assignment.We use the numpy mean to determine the average value
        cluster_mean = np.mean(x[clusters == c], axis=0)
        new_centroids.append(cluster_mean)
    return new_centroids


def kmeans(x, K):
    # initialize the centroids of 2 clusters which could be any x,y point in a range of
    # 0,0 to 11,11
    centroids = 11 * np.random.rand(K,K)

    # Print out the first pair of initialized centroids
    i = 0
    for cent in centroids:
        print('Initialized centroid {}: {}'.format(i, centroids[i]))
        i = i + 1

    # Iterate ten times to refine the centroid using the k-means algorithm
    for i in range(10):
        #  Use method closestCentroid to determine the clusters based on the current centroid
        clusters = closestCentroid(x, centroids)
        # Use method updateCentroid too determine the new improved cluster based on the current cluster alignment
        centroids = updateCentroid(x, clusters, K)

    # Print out the final pair of centroids finalized by k-means algorithm above
    i = 0
    for cent in centroids:
        print('Finalized centroid {}: {}'.format(i, centroids[i]))
        i = i + 1

    # View results using package seaborn and matplotlib
    sns.scatterplot(x=[X[0] for X in x],
                    y=[X[1] for X in x],
                    style=clusters,
                    legend=None
                    )
    plt.plot([x for x, _ in centroids],
             [y for _, y in centroids],
             '+',
             markersize=10,
             )
    plt.show()


# Code Starts here
# Number of clusters as input
K = 2
# Toy dataset of 6 points represented with its x and y co-ordinates
x = np.array([[2,4],
              [1.7,2.8],
              [7,8],
              [8.6,8],
              [3.4,1.5],
              [9,11]])
# Call kmeans method with input as as the dataset and cluster count
kmeans(x,K)
