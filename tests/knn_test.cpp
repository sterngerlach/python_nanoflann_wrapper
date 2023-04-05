
// knn_test.cpp

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <nanoflann.hpp>

template <typename T>
struct Point3d
{
  T x;
  T y;
  T z;
};

template <typename T>
std::ostream& operator<<(std::ostream& os, const Point3d<T>& point)
{
  os << '('
     << point.x << ", "
     << point.y << ", "
     << point.z << ')';
  return os;
}

template <typename T>
struct PointCloud
{
  // Get the number of points
  inline std::size_t kdtree_get_point_count() const
  { return this->points_.size(); }

  // Get the point coordinate
  inline T kdtree_get_pt(const std::size_t idx,
                         const std::size_t dim) const
  {
    if (dim == 0)
      return this->points_[idx].x;
    else if (dim == 1)
      return this->points_[idx].y;
    else
      return this->points_[idx].z;
  }

  // Check if the bounding box is already computed
  template <class BBox>
  inline bool kdtree_get_bbox(BBox&) const { return false; }

  std::vector<Point3d<T>> points_;
};

template <typename T, typename Rand>
void GenerateRandomPointCloud(PointCloud<T>& point_cloud,
                              const std::size_t num_points,
                              const T max_range_x,
                              const T max_range_y,
                              const T max_range_z,
                              Rand rand)
{
  point_cloud.points_.resize(num_points);

  for (std::size_t i = 0; i < num_points; ++i) {
    point_cloud.points_[i].x = max_range_x * rand();
    point_cloud.points_[i].y = max_range_y * rand();
    point_cloud.points_[i].z = max_range_z * rand();
  }
}

template <typename T>
std::string ArrayToString(const T* tensor, const int len,
                          const int precision = 3)
{
  if (len == 0)
    return "()";

  std::ostringstream out;
  out << std::setprecision(precision);
  out << std::fixed;
  out << '(';
  std::copy(tensor, tensor + len - 1, std::ostream_iterator<T>(out, ", "));
  out << tensor[len - 1];
  out << ')';

  return out.str();
}

int main()
{
  constexpr const int N = 4096;
  constexpr const int K = 16;
  constexpr const int Chunk = 100;

  using KDTree = nanoflann::KDTreeSingleIndexDynamicAdaptor<
    nanoflann::L2_Simple_Adaptor<float, PointCloud<float>>,
    PointCloud<float>, 3>;

  std::random_device rand_dev;
  std::default_random_engine rand_engine { rand_dev() };
  std::uniform_real_distribution<float> rand_dist { 0.0, 1.0 };

  // Create an empty point cloud and a K-D tree
  PointCloud<float> point_cloud;
  KDTree kdtree_index { 3, point_cloud, { 15 } };

  // Create a random point cloud
  const auto rand_gen = [&rand_engine, &rand_dist]() {
    return rand_dist(rand_engine); };
  GenerateRandomPointCloud(point_cloud, N, 10.0f, 10.0f, 10.0f, rand_gen);

  // Add points
  for (std::size_t i = 0; i < N; i += Chunk) {
    const std::size_t idx_end = std::min<std::size_t>(i + Chunk, N - 1);
    kdtree_index.addPoints(i, idx_end);
  }

  std::cerr << std::setprecision(3);
  std::cerr << std::fixed;

  // Query a point
  const float query_point[3] = { 0.5f, 0.5f, 0.5f };
  std::cerr << "Query point: "
            << ArrayToString(query_point, 3) << '\n';
  std::cerr << '\n';

  std::vector<std::size_t> result_idx;
  std::vector<float> result_dist_sq;

  // Query the nearest neighbor
  std::cerr << "Querying the nearest neighbor ...\n";
  nanoflann::KNNResultSet<float> nn_result_set { 1 };
  result_idx.resize(1);
  result_dist_sq.resize(1);
  nn_result_set.init(result_idx.data(), result_dist_sq.data());
  nanoflann::SearchParams nn_search_params { 0, 0.0f, true };
  kdtree_index.findNeighbors(nn_result_set, query_point, nn_search_params);

  for (std::size_t i = 0; i < nn_result_set.size(); ++i) {
    std::cerr << i << ": "
              << "index: " << result_idx[i] << ", "
              << "dist: " << result_dist_sq[i] << ", "
              << "point: " << point_cloud.points_[result_idx[i]] << '\n';
  }

  std::cerr << '\n';

  // Query the K-nearest neighbor
  std::cerr << "Querying the K-nearest neighbor ...\n";
  nanoflann::KNNResultSet<float> knn_result_set { K };
  result_idx.resize(K);
  result_dist_sq.resize(K);
  knn_result_set.init(result_idx.data(), result_dist_sq.data());
  nanoflann::SearchParams knn_search_params { 0, 0.0f, true };
  kdtree_index.findNeighbors(knn_result_set, query_point, knn_search_params);

  for (std::size_t i = 0; i < knn_result_set.size(); ++i) {
    std::cerr << i << ": "
              << "index: " << result_idx[i] << ", "
              << "dist: " << result_dist_sq[i] << ", "
              << "point: " << point_cloud.points_[result_idx[i]] << '\n';
  }

  return EXIT_SUCCESS;
}
