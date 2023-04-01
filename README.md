This code takes the folllowing input
- Number of clusters to be used. For this implementation we are using 2
- Toy dataset of 6 points represented with its x and y co-ordinates

## **def kmeans(x, K):**

initialize the centroids of 2 clusters which could be any x,y point in a range of 0,0 to 11,11

Print out the first pair of initialized centroids

Iterate ten times to refine the centroid using the k-means algorithm
- Use method closestCentroid to determine the clusters based on the current centroid
- Use method updateCentroid too determine the new improved cluster based on the current cluster alignment centroids = updateCentroid(x, clusters, K)

Print out the final pair of centroids finalized by k-means algorithm above

View results using package seaborn and matplotlib



## **_def cent_distance(a, b):_**
Use NumPy linear algebra functions to determine the distance


## **_def closestCentroid(x, centroids):_**
This method returns assignments of each point in the dataset to 0 or 1 based on the centroid that the point is closer to.

## **_def updateCentroid(x, clusters, K):_**
This method determines the new centroid value based on the input dataset and the current cluster assignment





