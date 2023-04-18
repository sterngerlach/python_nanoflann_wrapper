
// knn_lib.cpp

#include <memory>
#include <tuple>
#include <vector>

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "kd_tree_nanoflann.hpp"

namespace py = pybind11;

#define TO_STRING_CORE(x) #x
#define TO_STRING(x)      TO_STRING_CORE(x)

namespace knn_lib {

void PyBindKDTreeNanoFlann(py::module& m)
{
  using namespace py::literals;

  auto query_nn = [](const KDTreeNanoFlann& tree,
                     const Eigen::VectorXd& query,
                     const float eps,
                     const bool sorted) {
    int nn_idx;
    double nn_dist_sq;
    Eigen::VectorXd nn_point;
    const int k = tree.QueryNN(query, nn_idx, nn_dist_sq, nn_point,
                               eps, sorted);
    return std::make_tuple(k, nn_idx, nn_dist_sq, nn_point);
  };

  auto query_knn = [](const KDTreeNanoFlann& tree,
                      const Eigen::VectorXd& query,
                      const int knn,
                      const float eps,
                      const bool sorted) {
    std::vector<int> nn_indices;
    std::vector<double> nn_dist_sq;
    std::vector<Eigen::VectorXd> nn_points;
    const int k = tree.QueryKNN(query, knn, nn_indices, nn_dist_sq, nn_points,
                                eps, sorted);
    return std::make_tuple(k, nn_indices, nn_dist_sq, nn_points);
  };

  auto query_radius = [](const KDTreeNanoFlann& tree,
                         const Eigen::VectorXd& query,
                         const double radius,
                         const float eps,
                         const bool sorted) {
    std::vector<int> nn_indices;
    std::vector<double> nn_dist_sq;
    std::vector<Eigen::VectorXd> nn_points;
    const int k = tree.QueryRadius(query, radius,
                                   nn_indices, nn_dist_sq, nn_points,
                                   eps, sorted);
    return std::make_tuple(k, nn_indices, nn_dist_sq, nn_points);
  };

  auto query_hybrid = [](const KDTreeNanoFlann& tree,
                         const Eigen::VectorXd& query,
                         const double radius,
                         const int max_nn,
                         const float eps,
                         const bool sorted) {
    std::vector<int> nn_indices;
    std::vector<double> nn_dist_sq;
    std::vector<Eigen::VectorXd> nn_points;
    const int k = tree.QueryHybrid(query, radius, max_nn,
                                   nn_indices, nn_dist_sq, nn_points,
                                   eps, sorted);
    return std::make_tuple(k, nn_indices, nn_dist_sq, nn_points);
  };

  py::class_<KDTreeNanoFlann, std::shared_ptr<KDTreeNanoFlann>>
    kd_tree_nanoflann(m, "KDTreeNanoFlann");
  kd_tree_nanoflann.def(py::init<const int, const std::size_t>(),
    "Constructor", "dimensionality"_a, "leaf_max_size"_a = 10);

  kd_tree_nanoflann.def("point_at", &KDTreeNanoFlann::PointAt);
  kd_tree_nanoflann.def("data_matrix", &KDTreeNanoFlann::DataMatrix);

  kd_tree_nanoflann.def("add_point", &KDTreeNanoFlann::AddPoint,
    "point"_a);
  kd_tree_nanoflann.def("add_points", &KDTreeNanoFlann::AddPoints,
    "points"_a);
  kd_tree_nanoflann.def("remove_point", &KDTreeNanoFlann::RemovePoint,
    "idx"_a);

  kd_tree_nanoflann.def("query_nn", query_nn,
    "query"_a, "eps"_a = 0.0f, "sorted"_a = true);
  kd_tree_nanoflann.def("query_knn", query_knn,
    "query"_a, "knn"_a, "eps"_a = 0.0f, "sorted"_a = true);
  kd_tree_nanoflann.def("query_radius", query_radius,
    "query"_a, "radius"_a, "eps"_a = 0.0f, "sorted"_a = true);
  kd_tree_nanoflann.def("query_hybrid", query_hybrid,
    "query"_a, "radius"_a, "max_nn"_a, "eps"_a = 0.0f, "sorted"_a = true);

  kd_tree_nanoflann.def_property_readonly(
    "dims", &KDTreeNanoFlann::Dims);
  kd_tree_nanoflann.def_property_readonly(
    "num_points", &KDTreeNanoFlann::NumPoints);
  kd_tree_nanoflann.def_property_readonly(
    "data", &KDTreeNanoFlann::Data);
}

} // namespace knn_lib

PYBIND11_MODULE(knn_lib, m)
{
  m.doc() = "K-nearest neighbor library using Nanoflann";

  // Version
#ifdef VERSION_INFO
  m.attr("__version__") = TO_STRING(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif

  // `knn_lib.KDTreeNanoFlann` class
  knn_lib::PyBindKDTreeNanoFlann(m);
}
