# coding: utf-8
# knn_lib_test2.py

import matplotlib.pyplot as plt
import numpy as np

import knn_lib

def create_fig(points: np.ndarray,
               query: np.ndarray,
               knn_indices: list):
    fig = plt.figure(figsize=(5, 3), dpi=300)
    ax = fig.add_subplot(1, 1, 1)

    mask = np.ones(points.shape[0], bool)
    mask[knn_indices] = False

    ax.scatter(points[mask,0], points[mask,1],
               s=5.0, c="gray")
    ax.scatter(points[knn_indices,0], points[knn_indices,1],
               s=10.0, c="blue")
    ax.scatter(query[0], query[1],
               s=10.0, c="red")

    ax.set_aspect("equal")

    return fig

def main():
    # Set the printing options
    np.set_printoptions(precision=3, suppress=True)

    # Create a random set of 2D points
    points = np.random.rand(1024, 2)

    # Create a random query point
    query = np.array([0.5, 0.5])

    # Create a K-D tree for a 2D point cloud
    kd_tree = knn_lib.KDTreeNanoFlann(2)

    # Add points to the K-D tree
    for i in range(1024):
        kd_tree.add_point(points[i])

    # Add points to the K-D tree
    # kd_tree.add_points(points)

    print(f"Dimensionality: {kd_tree.dims}")
    print(f"Number of points: {kd_tree.num_points}")
    print(f"Query point: {query}")

    # Search the K-nearest neighbor points
    _, knn_indices, knn_dist_sq = kd_tree.query_knn(query, 16)
    for idx, dist_sq in zip(knn_indices, knn_dist_sq):
        print(f"K-nearest neighbor index: {idx}, "
              f"squared distance: {dist_sq:.3f}")

    # Visualize the results
    fig = create_fig(points, query, knn_indices)
    fig.savefig(f"result-knn.png", bbox_inches="tight",
                pad_inches=0.05, transparent=False)

    # Perform the radius search
    _, rnn_indices, rnn_dist_sq = kd_tree.query_radius(query, 0.25)
    for idx, dist_sq in zip(rnn_indices, rnn_dist_sq):
        print(f"Radius neighbor index: {idx}, "
              f"squared distance: {dist_sq:.3f}")

    # Visualize the results
    fig = create_fig(points, query, rnn_indices)
    fig.savefig(f"result-radius.png", bbox_inches="tight",
                pad_inches=0.05, transparent=False)

if __name__ == "__main__":
    main()
