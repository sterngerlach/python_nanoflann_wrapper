
// kd_tree_nanoflann.hpp

#pragma once

#include <memory>
#include <vector>

#include <Eigen/Core>
#include <nanoflann.hpp>

namespace knn_lib {

// Point cloud type
template <typename T>
struct PointCloud
{
  // Vector type
  using TVector = Eigen::Matrix<T, -1, 1>;

  // Constructor
  PointCloud() = default;
  // Destructor
  ~PointCloud() = default;

  // Get the number of points
  inline std::size_t kdtree_get_point_count() const
  { return this->data_.size(); }

  // Get the point coordinate
  inline T kdtree_get_pt(const std::size_t idx,
                         const std::size_t dim) const
  { return this->data_[idx][dim]; }

  // Check if the bounding box is already computed
  template <class BBox>
  inline bool kdtree_get_bbox(BBox&) const { return false; }

  // Point cloud data
  std::vector<TVector> data_;
};

// K-D tree based on the NanoFlann library
// Refer to cpp/open3d/geometry/KDTreeFlann.h in Open3D repository
class KDTreeNanoFlann final
{
public:
  using KDTree = nanoflann::KDTreeSingleIndexDynamicAdaptor<
    nanoflann::L2_Simple_Adaptor<double, PointCloud<double>>,
    PointCloud<double>, -1>;

  // Constructor
  KDTreeNanoFlann(const int dimensionality,
                  const std::size_t leaf_max_size = 10);

  // Destructor
  ~KDTreeNanoFlann() = default;

  // Copy constructor (disabled)
  KDTreeNanoFlann(const KDTreeNanoFlann&) = delete;
  // Copy assignment operator (disabled)
  KDTreeNanoFlann& operator=(const KDTreeNanoFlann&) = delete;

  // Get the dimensionality
  inline int Dims() const { return this->dims_; }

  // Get the number of points
  inline int NumPoints() const
  { return static_cast<int>(this->point_cloud_.data_.size()); }

  // Get the reference to the point cloud
  inline const std::vector<Eigen::VectorXd>& Data() const
  { return this->point_cloud_.data_; }

  // Get the point cloud as a matrix
  Eigen::MatrixXd DataMatrix() const;

  // Add a new point and update the K-D tree index
  void AddPoint(const Eigen::VectorXd& point);

  // Add a set of new points and update the K-D tree index
  void AddPoints(const std::vector<Eigen::VectorXd>& points);

  // Remove a point at the specified index
  void RemovePoint(const std::size_t idx);

  // Query the nearest neighbor point
  int QueryNN(const Eigen::VectorXd& query,
              int& nn_idx,
              double& nn_dist_sq,
              const float eps = 0.0f,
              const bool sorted = true) const;

  // Query the K-nearest neighbor points
  int QueryKNN(const Eigen::VectorXd& query,
               const int knn,
               std::vector<int>& nn_indices,
               std::vector<double>& nn_dist_sq,
               const float eps = 0.0f,
               const bool sorted = true) const;

  // Perform the radius search
  int QueryRadius(const Eigen::VectorXd& query,
                  const double radius,
                  std::vector<int>& nn_indices,
                  std::vector<double>& nn_dist_sq,
                  const float eps = 0.0f,
                  const bool sorted = true) const;

  // Perform the hybrid search
  int QueryHybrid(const Eigen::VectorXd& query,
                  const double radius,
                  const int max_nn,
                  std::vector<int>& nn_indices,
                  std::vector<double>& nn_dist_sq,
                  const float eps = 0.0f,
                  const bool sorted = true) const;

private:
  // Dimensionality
  int dims_;
  // Point cloud data
  PointCloud<double> point_cloud_;
  // K-D tree index
  std::unique_ptr<KDTree> kdtree_index_;
};

} // namespace knn_lib
