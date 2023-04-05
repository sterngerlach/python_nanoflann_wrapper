
// kd_tree_nanoflann.cpp

#include <algorithm>
#include <iostream>
#include <nanoflann.hpp>

#include "kd_tree_nanoflann.hpp"

namespace knn_lib {

// Constructor
KDTreeNanoFlann::KDTreeNanoFlann(const int dimensionality,
                                 const std::size_t leaf_max_size) :
  dims_(dimensionality),
  point_cloud_(),
  kdtree_index_(nullptr)
{
  this->kdtree_index_.reset(new KDTree(dimensionality, this->point_cloud_,
    nanoflann::KDTreeSingleIndexAdaptorParams(leaf_max_size,
      nanoflann::KDTreeSingleIndexAdaptorFlags::None)));
}

// Get the point cloud as a matrix
Eigen::MatrixXd KDTreeNanoFlann::DataMatrix() const
{
  const int num_points = this->NumPoints();
  Eigen::MatrixXd point_cloud_mat { num_points, this->dims_ };

  for (int i = 0; i < num_points; ++i) {
    point_cloud_mat.row(i) = this->point_cloud_.data_[i];
  }

  return point_cloud_mat;
}

// Add a new point and update the K-D tree index
void KDTreeNanoFlann::AddPoint(const Eigen::VectorXd& point)
{
  this->point_cloud_.data_.push_back(point);
  const std::size_t idx = this->point_cloud_.data_.size() - 1;
  this->kdtree_index_->addPoints(idx, idx);
}

// Add a set of new points and update the K-D tree index
void KDTreeNanoFlann::AddPoints(const std::vector<Eigen::VectorXd>& points)
{
  this->point_cloud_.data_.insert(
    this->point_cloud_.data_.end(), points.begin(), points.end());
  const std::size_t idx_start = this->point_cloud_.data_.size() - points.size();
  const std::size_t idx_end = this->point_cloud_.data_.size() - 1;
  this->kdtree_index_->addPoints(idx_start, idx_end);
}

// Remove a point at the specified index
void KDTreeNanoFlann::RemovePoint(const std::size_t idx)
{
  if (idx >= this->point_cloud_.data_.size())
    throw std::runtime_error("`idx` is out of bounds");

  this->kdtree_index_->removePoint(idx);
}

// Query the nearest neighbor point
int KDTreeNanoFlann::QueryNN(const Eigen::VectorXd& query,
                              int& nn_idx,
                              double& nn_dist_sq,
                              const float eps,
                              const bool sorted) const
{
  std::size_t result_idx[1];
  double result_dist_sq[1];

  nanoflann::KNNResultSet<double> nn_result_set { 1 };
  nn_result_set.init(result_idx, result_dist_sq);
  nanoflann::SearchParams search_params { 0, eps, sorted };

  this->kdtree_index_->findNeighbors(
    nn_result_set, query.data(), search_params);
  nn_idx = static_cast<int>(result_idx[0]);
  nn_dist_sq = result_dist_sq[0];

  return static_cast<int>(nn_result_set.size());
}

// Query the K-nearest neighbor points
int KDTreeNanoFlann::QueryKNN(const Eigen::VectorXd& query,
                              const int knn,
                              std::vector<int>& nn_indices,
                              std::vector<double>& nn_dist_sq,
                              const float eps,
                              const bool sorted) const
{
  std::vector<std::size_t> result_indices;
  result_indices.resize(knn);
  nn_dist_sq.resize(knn);

  nanoflann::KNNResultSet<double> nn_result_set { knn };
  nn_result_set.init(result_indices.data(), nn_dist_sq.data());
  nanoflann::SearchParams search_params { 0, eps, sorted };

  this->kdtree_index_->findNeighbors(
    nn_result_set, query.data(), search_params);
  nn_indices.resize(nn_result_set.size());
  nn_dist_sq.resize(nn_result_set.size());
  std::copy_n(result_indices.begin(), nn_result_set.size(),
              nn_indices.begin());

  return static_cast<int>(nn_result_set.size());
}

// Perform the radius search
int KDTreeNanoFlann::QueryRadius(const Eigen::VectorXd& query,
                                 const double radius,
                                 std::vector<int>& nn_indices,
                                 std::vector<double>& nn_dist_sq,
                                 const float eps,
                                 const bool sorted) const
{
  using IndexDistPair = std::pair<std::size_t, double>;
  const double radius_sq = radius * radius;

  std::vector<IndexDistPair> indices_dists;
  nanoflann::RadiusResultSet<double> nn_result_set {
    radius_sq, indices_dists };
  nanoflann::SearchParams search_params { 0, eps, sorted };

  this->kdtree_index_->findNeighbors(nn_result_set, query.data(),
                                     search_params);

  // Sort the results manually
  // Refer to `KDTreeSingleIndexAdaptor::radiusSearch()`
  if (sorted)
    std::sort(indices_dists.begin(), indices_dists.end(),
              nanoflann::IndexDist_Sorter());

  nn_indices.resize(indices_dists.size());
  nn_dist_sq.resize(indices_dists.size());
  std::transform(indices_dists.begin(), indices_dists.end(),
    nn_indices.begin(),
    [](const IndexDistPair& x) { return static_cast<int>(x.first); });
  std::transform(indices_dists.begin(), indices_dists.end(),
    nn_dist_sq.begin(),
    [](const IndexDistPair& x) { return x.second; });

  return static_cast<int>(indices_dists.size());
}

// Perform the hybrid search
int KDTreeNanoFlann::QueryHybrid(const Eigen::VectorXd& query,
                                 const double radius,
                                 const int max_nn,
                                 std::vector<int>& nn_indices,
                                 std::vector<double>& nn_dist_sq,
                                 const float eps,
                                 const bool sorted) const
{
  const double radius_sq = radius * radius;

  std::vector<std::size_t> result_indices;
  result_indices.resize(max_nn);
  nn_dist_sq.resize(max_nn);

  nanoflann::KNNResultSet<double> nn_result_set { max_nn };
  nn_result_set.init(result_indices.data(), nn_dist_sq.data());
  // `sorted` should be set to `true` to use `std::lower_bound()`
  nanoflann::SearchParams search_params { 0, eps, true };

  this->kdtree_index_->findNeighbors(
    nn_result_set, query.data(), search_params);

  // Remove the points with the large distances
  const int actual_nn = std::distance(nn_dist_sq.begin(),
    std::lower_bound(nn_dist_sq.begin(), nn_dist_sq.end(), radius_sq));
  nn_indices.resize(actual_nn);
  nn_dist_sq.resize(actual_nn);
  std::copy_n(result_indices.begin(), actual_nn, nn_indices.begin());

  return actual_nn;
}

} // namespace knn_lib
