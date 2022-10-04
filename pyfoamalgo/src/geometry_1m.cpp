/**
 * Distributed under the terms of the GNU General Public License v3.0.
 *
 * The full license is in the file LICENSE, distributed with this software.
 *
 * Copyright (C) 2020, Jun Zhu. All rights reserved.
 */
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#if defined(FOAMALGO_USE_TBB)
#include "tbb/parallel_for.h"
#include "tbb/blocked_range2d.h"
#endif

#include "foamalgo/geometry_1m.hpp"
#include "pyconfig.hpp"

namespace py = pybind11;


template<typename Geometry>
void declareGeometry1M(py::module &m, std::string&& detector)
{
  using GeometryBase = foam::Detector1MGeometryBase<Geometry>;
  const std::string py_base_class_name = detector + std::string("_Detector1MGeometryBase");

  py::class_<GeometryBase> base(m, py_base_class_name.c_str());

#define FOAM_POSITION_ALL_MODULES(SRC_TYPE, DST_TYPE)                                                               \
  base.def("positionAllModules",                                                                                    \
  (void (GeometryBase::*)(const xt::pytensor<SRC_TYPE, 3>&, xt::pytensor<DST_TYPE, 2>&, bool) const)                \
    &GeometryBase::positionAllModules,                                                                              \
    py::arg("src").noconvert(), py::arg("dst").noconvert(), py::arg("ignore_tile_edge") = false);                   \
  base.def("positionAllModules",                                                                                    \
  (void (GeometryBase::*)(const xt::pytensor<SRC_TYPE, 4>&, xt::pytensor<DST_TYPE, 3>&, bool) const)                \
    &GeometryBase::positionAllModules,                                                                              \
    py::arg("src").noconvert(), py::arg("dst").noconvert(), py::arg("ignore_tile_edge") = false);                   \
  base.def("positionAllModules",                                                                                    \
    (void (GeometryBase::*)(const std::vector<xt::pytensor<SRC_TYPE, 3>>&, xt::pytensor<DST_TYPE, 3>&, bool) const) \
    &GeometryBase::positionAllModules,                                                                              \
    py::arg("src").noconvert(), py::arg("dst").noconvert(), py::arg("ignore_tile_edge") = false);

  FOAM_POSITION_ALL_MODULES(float, float)
  FOAM_POSITION_ALL_MODULES(uint16_t, float)
  FOAM_POSITION_ALL_MODULES(uint16_t, uint16_t)
  FOAM_POSITION_ALL_MODULES(bool, bool)

#define FOAM_DISMANTLE_ALL_MODULES(SRC_TYPE, DST_TYPE)                                                 \
  base.def("dismantleAllModules",                                                                      \
  (void (GeometryBase::*)(const xt::pytensor<SRC_TYPE, 2>&, xt::pytensor<DST_TYPE, 3>&) const)         \
    &GeometryBase::dismantleAllModules,                                                                \
    py::arg("src").noconvert(), py::arg("dst").noconvert());                                           \
  base.def("dismantleAllModules",                                                                      \
  (void (GeometryBase::*)(const xt::pytensor<SRC_TYPE, 3>&, xt::pytensor<DST_TYPE, 4>&) const)         \
    &GeometryBase::dismantleAllModules,                                                                \
    py::arg("src").noconvert(), py::arg("dst").noconvert());

  FOAM_DISMANTLE_ALL_MODULES(float, float)
  FOAM_DISMANTLE_ALL_MODULES(uint16_t, float)
  FOAM_DISMANTLE_ALL_MODULES(uint16_t, uint16_t)
  FOAM_DISMANTLE_ALL_MODULES(bool, bool)

  base.def("assembledShape", &GeometryBase::assembledShape)
    .def_readonly_static("n_quads", &GeometryBase::n_quads)
    .def_readonly_static("n_modules", &GeometryBase::n_modules)
    .def_readonly_static("n_modules_per_quad", &GeometryBase::n_modules_per_quad);

  const std::string py_class_name = detector + std::string("_1MGeometry");

  py::class_<Geometry, GeometryBase> cls(m, py_class_name.c_str());

  cls.def(py::init())
    .def(py::init<const std::array<std::array<std::array<double, 3>, Geometry::n_tiles_per_module>, Geometry::n_modules> &>())
    .def_readonly_static("pixel_size", &Geometry::pixel_size_py)
    .def_readonly_static("module_shape", &Geometry::module_shape)
    .def_readonly_static("tile_shape", &Geometry::tile_shape)
    .def_readonly_static("n_tiles_per_module", &Geometry::n_tiles_per_module)
    .def_readonly_static("quad_orientations", &Geometry::quad_orientations);

}

PYBIND11_MODULE(geometry_1m, m)
{
  xt::import_numpy();

  m.doc() = "1M detector geometry.";

  declareGeometry1M<foam::AGIPD_1MGeometry>(m, "AGIPD");

  declareGeometry1M<foam::LPD_1MGeometry>(m, "LPD");

  declareGeometry1M<foam::DSSC_1MGeometry>(m, "DSSC");
}
