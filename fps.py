import fpsample
from scipy.spatial import cKDTree

def k_nearest_neighbors_optimized(points, centroids, k):
    # Build a KD-Tree for fast K-NN search
    kdtree = cKDTree(points)

    # Query KD-Tree to find the k-nearest neighbors for each centroid
    _, knn_indices = kdtree.query(centroids, k=k)
 
    return points[knn_indices], knn_indices
 
def fps(points, num_centroids=2048, h = 9):
    kdline_fps_samples_idx = fpsample.bucket_fps_kdline_sampling(points, num_centroids, h=h)
    return points[kdline_fps_samples_idx], kdline_fps_samples_idx