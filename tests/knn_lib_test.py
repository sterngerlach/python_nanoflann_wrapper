# coding: utf-8
# knn_lib_test.py

import numpy as np

import knn_lib

def main():
    # Set the printing options
    np.set_printoptions(precision=3, suppress=True)

    # Create 3D points
    points = np.array([[1.0, 2.0, 3.0],
                       [4.0, 5.0, 6.0],
                       [7.0, 8.0, 9.0],
                       [1.3, 2.3, 3.3],
                       [4.3, 5.3, 6.3],
                       [7.3, 8.3, 9.3],
                       [1.6, 2.6, 3.6],
                       [4.6, 5.6, 6.6],
                       [7.6, 8.6, 9.6]])

    # Create a K-D tree for a 3D point cloud
    kd_tree = knn_lib.KDTreeNanoFlann(3)

    # Add points to the K-D tree
    kd_tree.add_points(points)

    print(f"Dimensionality: {kd_tree.dims}")
    print(f"Number of points: {kd_tree.num_points}")
    print(f"Point cloud:\n{kd_tree.data_matrix()}")

    # Query point
    query = np.array([4.4, 5.4, 6.4])
    print(f"Query point: {query}")

    # Test the `knn_lib` implementation
    print(f"Testing the `knn_lib` implementation ...")

    # Search the nearest neighbor point
    _, nn_idx, nn_dist_sq, nn_point = kd_tree.query_nn(query)
    print(f"Nearest neighbor index: {nn_idx}, "
          f"squared distance: {nn_dist_sq:.3f}, "
          f"point: {nn_point}")

    # Search the K-nearest neighbor points
    _, knn_indices, knn_dist_sq, knn_points = kd_tree.query_knn(query, 3)
    for idx, dist_sq, point in zip(knn_indices, knn_dist_sq, knn_points):
        print(f"K-nearest neighbor index: {idx}, "
              f"squared distance: {dist_sq:.3f}, "
              f"point: {point}")

    # Perform the radius search
    _, rnn_indices, rnn_dist_sq, rnn_points = \
        kd_tree.query_radius(query, np.sqrt(28.0))
    for idx, dist_sq, point in zip(rnn_indices, rnn_dist_sq, rnn_points):
        print(f"Radius neighbor index: {idx}, "
              f"squared distance: {dist_sq:.3f}, "
              f"point: {point}")

    # Perform the hybrid search
    _, hybrid_indices, hybrid_dist_sq, hybrid_points = \
        kd_tree.query_hybrid(query, 28.0, 4)
    for idx, dist_sq, point in zip(
        hybrid_indices, hybrid_dist_sq, hybrid_points):
        print(f"Hybrid neighbor index: {idx}, "
              f"squared distance: {dist_sq:.3f}, "
              f"point: {point}")

    print()

    # Compare the results with the Naive implementation
    print(f"Testing the naive implementation ...")

    dists = np.sum((points - query) ** 2.0, axis=-1)

    # Search the nearest neighbor point
    nn_idx = np.argmin(dists)
    nn_dist_sq = dists[nn_idx]
    nn_point = points[nn_idx]
    print(f"Nearest neighbor index: {nn_idx}, "
          f"squared distance: {nn_dist_sq:.3f}, "
          f"point: {nn_point}")

    # Search the K-nearest neighbor points
    knn_indices = np.argpartition(dists, 3)[:3]
    knn_dist_sq = dists[knn_indices]
    knn_points = points[knn_indices]
    for idx, dist_sq, point in zip(knn_indices, knn_dist_sq, knn_points):
        print(f"K-nearest neighbor index: {idx}, "
              f"squared distance: {dist_sq:.3f}, "
              f"point: {point}")

    # Perform the radius search
    rnn_indices = np.where(dists <= 28.0)[0]
    rnn_dist_sq = dists[rnn_indices]
    rnn_points = points[rnn_indices]
    rnn_sort = np.argsort(rnn_dist_sq)
    rnn_indices = rnn_indices[rnn_sort]
    rnn_dist_sq = rnn_dist_sq[rnn_sort]
    rnn_points = rnn_points[rnn_sort]
    for idx, dist_sq, point in zip(rnn_indices, rnn_dist_sq, rnn_points):
        print(f"Radius neighbor index: {idx}, "
              f"squared distance: {dist_sq:.3f}, "
              f"point: {point}")

    # Perform the hybrid search
    hybrid_indices = np.where(dists <= 28.0)[0]
    hybrid_dist_sq = dists[hybrid_indices]
    hybrid_points = points[hybrid_indices]
    hybrid_sort = np.argsort(hybrid_dist_sq)[:4]
    hybrid_indices = hybrid_indices[hybrid_sort]
    hybrid_dist_sq = hybrid_dist_sq[hybrid_sort]
    hybrid_points = hybrid_points[hybrid_sort]
    for idx, dist_sq, point in zip(
        hybrid_indices, hybrid_dist_sq, hybrid_points):
        print(f"Hybrid neighbor index: {idx}, "
              f"squared distance: {dist_sq:.3f}, "
              f"point: {point}")

if __name__ == "__main__":
    main()
