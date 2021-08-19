/**
 * Distributed under the terms of the BSD 3-Clause License.
 *
 * The full license is in the file BSD_LICENSE, distributed with this software.
 *
 * Author: Jun Zhu <jun.zhu@xfel.eu>
 * Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
 * All rights reserved.
 */

#ifndef FOAM_AZIMUTHAL_INTEGRATOR_H
#define FOAM_AZIMUTHAL_INTEGRATOR_H

#include <cmath>

#if defined(FOAM_USE_TBB)
#include <mutex> // There is a bug in "tbb/mutex.h".
#include "tbb/parallel_for.h"
#endif

#include <xtensor/xmath.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xfixed.hpp>

#include "traits.hpp"


namespace foam
{
namespace ai
{

/**
 * Compute the geometry (distance to the center) for azimuthal integration.
 *
 * @param src: Source image. Shape = (y, x)
 * @param poni1: Integration center y, in meter.
 * @param poni2: Integration center x, in meter.
 * @param pixel1: Pixel size along y, in meter.
 * @param pixel2: Pixel size along x, in meter.
 *
 * @return: Array of distance to the PONI, in meter. Shape = (y, x)
 */
template<typename T, typename E>
xt::xtensor<T, 2> computeGeometry(E&& src, T poni1, T poni2, T pixel1, T pixel2)
{
  auto shape = src.shape();
  xt::xtensor<T, 2> geometry = xt::zeros<T>(shape);
  for (size_t i = 0; i < shape[0]; ++i)
  {
    for (size_t j = 0; j < shape[1]; ++j)
    {
      T dx = static_cast<T>(j) * pixel2 - poni2;
      T dy = static_cast<T>(i) * pixel1 - poni1;
      geometry(i, j) = std::sqrt(dx * dx + dy * dy);
    }
  }

  return geometry;
}

/**
 * Compute the geometry (Q-map) for azimuthal integration.
 *
 * @param src: Source image. Shape = (y, x)
 * @param poni1: Integration center y, in meter.
 * @param poni2: Integration center x, in meter.
 * @param pixel1: Pixel size along y, in meter.
 * @param pixel2: Pixel size along x, in meter.
 * @param dist: Sample distance in meter.
 * @param wavelength: Photon wavelength in meter.
 *
 * @return: Array of momentum transfer, in 1/meter. Shape = (y, x).
 */
template<typename T, typename E>
xt::xtensor<T, 2> computeGeometry(E&& src, T poni1, T poni2, T pixel1, T pixel2, T dist, T wavelength)
{
  T four_pi_over_lambda = T(4) * T(M_PI) / wavelength;

  auto shape = src.shape();
  xt::xtensor<T, 2> geometry = xt::zeros<T>(shape);
  for (size_t i = 0; i < shape[0]; ++i)
  {
    for (size_t j = 0; j < shape[1]; ++j)
    {
      T dx = static_cast<T>(j) * pixel2 - poni2;
      T dy = static_cast<T>(i) * pixel1 - poni1;
      // Convert radial distances (in meter) to momentum transfer q (in 1/meter).
      // q = 4 * pi * sin(2 * theta / 2.0) / lambda
      T tmp = std::sqrt(dx * dx + dy * dy);
      geometry(i, j) = four_pi_over_lambda * std::sin(std::atan2(tmp, dist) / T(2));
    }
  }

  return geometry;
}

namespace detail
{

template<typename E1, typename E2, typename E3, typename T>
void histogramAIImp(E1&& src, const E2& geometry, E3& hist, T q_min, T q_max, size_t n_bins, size_t min_count)
{
  using value_type = typename std::decay_t<E3>::value_type;

  double norm = 1. / (static_cast<double>(q_max) - static_cast<double>(q_min));
  xt::xtensor<size_t, 1> counts = xt::zeros<size_t>({ n_bins });

  auto shape = src.shape();
  for (size_t i = 0; i < shape[0]; ++i)
  {
    for (size_t j = 0; j < shape[1]; ++j)
    {
      auto q = static_cast<double>(geometry(i, j));
      auto v = static_cast<value_type>(src(i, j));

      if (std::isnan(v)) continue;

      if (q == q_max)
      {
        hist(n_bins - 1) += v;
        counts(n_bins - 1) += 1;
      } else if ( (q >= q_min) && (q < q_max) )
      {
        auto i_bin = static_cast<size_t>(
          static_cast<double>(n_bins) * (q - static_cast<double>(q_min)) * norm);
        hist(i_bin) += v;
        counts(i_bin) += 1;
      }
    }
  }

  // thresholding
  if (min_count > 1)
  {
    for (size_t i = 0; i < n_bins; ++i)
    {
      if (counts(i) < min_count) hist(i) = 0.;
    }
  }

  // normalizing
  for (size_t i = 0; i < n_bins; ++i)
  {
    if (counts(i) == 0) hist(i) = 0.;
    else
      hist(i) /= static_cast<value_type>(counts(i));
  }
}

} // detail

template<typename E1, typename E2, typename T, EnableIf<std::decay_t<E1>, IsImage> = false>
auto histogramAI(E1&& src, const E2& geometry, T q_min, T q_max, size_t n_bins, size_t min_count=1)
{
  using container_value_type = typename std::decay_t<E1>::value_type;
  using value_type = std::conditional_t<std::is_floating_point<container_value_type>::value,
                                        container_value_type,
                                        T>;
  using vector_type = ReducedVectorType<E1, value_type>;

  vector_type hist = xt::zeros<value_type>({ n_bins });

  detail::histogramAIImp(std::forward<E1>(src), geometry, hist, q_min, q_max, n_bins, min_count);

  vector_type edges = xt::linspace<value_type>(q_min, q_max, n_bins + 1);
  auto&& centers = 0.5 * (xt::view(edges, xt::range(0, -1)) + xt::view(edges, xt::range(1, xt::placeholders::_)));

  return std::make_pair<vector_type, vector_type>(centers, std::move(hist));
}

template<typename E, typename T, EnableIf<std::decay_t<E>, IsImage> = false>
auto histogramAI(E&& src, T poni1, T poni2, T pixel1, T pixel2, size_t npt, size_t min_count=1)
{
  auto geometry = computeGeometry(src, poni1, poni2, pixel1, pixel2);
  std::array<T, 2> bounds = xt::minmax(geometry)();
  return histogramAI(std::forward<E>(src), geometry, bounds[0], bounds[1], npt, min_count);
}

template<typename E1, typename E2, typename T, EnableIf<std::decay_t<E1>, IsImageArray> = false>
auto histogramAI(E1&& src, const E2& geometry, T q_min, T q_max, size_t n_bins, size_t min_count=1)
{
  using container_value_type = typename std::decay_t<E1>::value_type;
  using value_type = std::conditional_t<std::is_floating_point<container_value_type>::value,
                                        container_value_type,
                                        T>;
  using image_type = ReducedImageType<E1, value_type>;
  using vector_type = ReducedVectorType<image_type, value_type>;

  size_t np = src.shape()[0];
  image_type hist = xt::zeros<value_type>({ np, n_bins });

#if defined(FOAM_USE_TBB)
  tbb::parallel_for(tbb::blocked_range<int>(0, np),
    [&src, &geometry, &hist, q_min, q_max, n_bins, min_count]
    (const tbb::blocked_range<int> &block)
    {
      for(int k=block.begin(); k != block.end(); ++k)
      {
#else
      for (size_t k = 0; k < np; ++k)
      {
#endif
        auto hist_view = xt::view(hist, k, xt::all());
        detail::histogramAIImp(xt::view(src, k, xt::all(), xt::all()), geometry, hist_view,
                               q_min, q_max, n_bins, min_count);
      }
#if defined(FOAM_USE_TBB)
    }
  );
#endif

  vector_type edges = xt::linspace<value_type>(q_min, q_max, n_bins + 1);
  auto&& centers = 0.5 * (xt::view(edges, xt::range(0, -1)) + xt::view(edges, xt::range(1, xt::placeholders::_)));

  return std::make_pair<vector_type, image_type>(centers, std::move(hist));
}

} //ai

enum class AzimuthalIntegrationMethod
{
  HISTOGRAM = 0x01,
};


/**
 * @class AzimuthalIntegrator
 * @brief Perform 1D azimuthal integration of image data.
 *
 */
template<typename T = double>
class AzimuthalIntegrator
{
  static_assert(std::is_floating_point<T>::value);

  T dist_; // sample distance, in m
  xt::xtensor_fixed<T, xt::xshape<3>> poni_; // integration center (y, x, z), in meter
  xt::xtensor_fixed<T, xt::xshape<3>> pixel_; // pixel size (y, x, z), in meter
  T wavelength_; // Photon wavelength, in m

  bool initialized_ = false;
  xt::xtensor<T, 2> q_;
  T q_min_;
  T q_max_;

  AzimuthalIntegrationMethod method_;

  /**
   * Initialize Q-map.
   */
  template<typename E>
  void initQ(const E& src);

public:

  AzimuthalIntegrator(T dist, T poni1, T poni2, T pixel1, T pixel2, T wavelength);

  ~AzimuthalIntegrator() = default;

  /**
   * Calculate the 1D azimuthal integration of an image.
   *
   * @param src: source image. Shape = (y, x)
   * @param npt: number of integration points.
   * @param min_count: minimum number of pixels required.
   * @param method: azimuthal integration method.
   *
   * @return (q, s): (momentum transfer, scattering)
   */
  template<typename E, EnableIf<std::decay_t<E>, IsImage> = false>
  auto integrate1d(E&& src, size_t npt, size_t min_count=1,
                   AzimuthalIntegrationMethod method=AzimuthalIntegrationMethod::HISTOGRAM);

  /**
   * Calculate the 1D azimuthal integrations of an array of images.
   *
   * @param src: source image. Shape = (indices, y, x)
   * @param npt: number of integration points.
   * @param min_count: minimum number of pixels required.
   * @param method: azimuthal integration method.
   *
   * @return (q, s): (momentum transfer, scattering)
   */
  template<typename E, EnableIf<std::decay_t<E>, IsImageArray> = false>
  auto integrate1d(E&& src, size_t npt, size_t min_count=1,
                   AzimuthalIntegrationMethod method=AzimuthalIntegrationMethod::HISTOGRAM);
};

template<typename T>
template<typename E>
void AzimuthalIntegrator<T>::initQ(const E& src)
{
  q_ = ai::computeGeometry(src, poni_[0], poni_[1], pixel_[0], pixel_[1], dist_, wavelength_);
  std::array<T, 2> bounds = xt::minmax(q_)();
  q_min_ = bounds[0];
  q_max_ = bounds[1];
}

template<typename T>
AzimuthalIntegrator<T>::AzimuthalIntegrator(T dist, T poni1, T poni2, T pixel1, T pixel2, T wavelength)
  : dist_(dist), poni_({poni1, poni2, 0}), pixel_({pixel1, pixel2, 0}), wavelength_(wavelength)
{
}

template<typename T>
template<typename E, EnableIf<std::decay_t<E>, IsImage>>
auto AzimuthalIntegrator<T>::integrate1d(E&& src,
                                         size_t npt,
                                         size_t min_count,
                                         AzimuthalIntegrationMethod method)
{
  if (npt == 0) npt = 1;

  auto src_shape = src.shape();
  std::array<size_t, 2> q_shape = q_.shape();
  if (!initialized_ || src_shape[0] != q_shape[0] || src_shape[1] != q_shape[1])
  {
    initQ(src);
    initialized_ = true;
  }

  switch(method)
  {
    case AzimuthalIntegrationMethod::HISTOGRAM:
    {
      return ai::histogramAI(std::forward<E>(src), q_, q_min_, q_max_, npt, min_count);
    }
    default:
      throw std::runtime_error("Unknown azimuthal integration method");
  }
}

template<typename T>
template<typename E, EnableIf<std::decay_t<E>, IsImageArray>>
auto AzimuthalIntegrator<T>::integrate1d(E&& src,
                                         size_t npt,
                                         size_t min_count,
                                         AzimuthalIntegrationMethod method)
{
  if (npt == 0) npt = 1;

  auto src_shape = src.shape();
  std::array<size_t, 2> q_shape = q_.shape();
  if (!initialized_ || src_shape[1] != q_shape[0] || src_shape[2] != q_shape[1])
  {
    initQ(xt::view(src, 0, xt::all(), xt::all()));
    initialized_ = true;
  }

  switch(method)
  {
    case AzimuthalIntegrationMethod::HISTOGRAM:
    {
      return ai::histogramAI(std::forward<E>(src), q_, q_min_, q_max_, npt, min_count);
    }
    default:
      throw std::runtime_error("Unknown azimuthal integration method");
  }
}

/**
 * @class ConcentricRingsFinder
 * @brief Detect the center of concentric rings in an image.
 */
template<typename T = double>
class ConcentricRingsFinder
{
  static_assert(std::is_floating_point<T>::value);

  T pixel_x_; // pixel size in x direction
  T pixel_y_; // pixel size in y direction

  template<typename E>
  size_t estimateNPoints(const E& src, T cx, T cy) const;

public:

  ConcentricRingsFinder(T pixel_x, T pixel_y);

  ~ConcentricRingsFinder() = default;

  /**
   * Search for the center of concentric rings in an image.
   *
   * @param src: source image.
   * @param cx0: starting x position, in pixels.
   * @param cy0: starting y position, in pixels.
   * @param min_count: minimum number of pixels required for each grid.
   *
   * @return: the optimized (cx, cy) position in pixels.
   */
  template<typename E, EnableIf<std::decay_t<E>, IsImage> = false>
  std::array<T, 2> search(E&& src, T cx0, T cy0, size_t min_count=1) const;
};

template<typename T>
ConcentricRingsFinder<T>::ConcentricRingsFinder(T pixel_x, T pixel_y)
  : pixel_x_(pixel_x), pixel_y_(pixel_y)
{
}

template<typename T>
template<typename E>
size_t ConcentricRingsFinder<T>::estimateNPoints(const E& src, T cx, T cy) const
{
  auto shape = src.shape();
  auto h = static_cast<T>(shape[0]);
  auto w = static_cast<T>(shape[1]);

  T dx = cx - w;
  T dy = cy - h;
  T max_dist = std::sqrt(cx * cx + cy * cy);
  T dist = std::sqrt(dx * dx + cy * cy);
  if (dist > max_dist) max_dist = dist;
  dist = std::sqrt(cx * cx + dy * dy);
  if (dist > max_dist) max_dist = dist;
  dist = std::sqrt(dx * dx + dy * dy);
  if (dist > max_dist) max_dist = dist;

  return static_cast<size_t>(dist / 2);
}

template<typename T>
template<typename E, EnableIf<std::decay_t<E>, IsImage>>
std::array<T, 2> ConcentricRingsFinder<T>::search(E&& src, T cx0, T cy0, size_t min_count) const
{
  T cx_max = cx0;
  T cy_max = cy0;
  T max_s = -1.f;
  size_t npt = estimateNPoints(src, cx0, cy0);

  int initial_space = 10;
#if defined(FOAM_USE_TBB)
  std::mutex mtx;
  tbb::parallel_for(tbb::blocked_range<int>(-initial_space, initial_space),
    [&src, cx0, cy0, npt, min_count, &cx_max, &cy_max, &max_s, initial_space, &mtx, this]
    (const tbb::blocked_range<int> &block)
    {
      for(int i=block.begin(); i != block.end(); ++i)
      {
#else
      for (int i = -initial_space; i <= initial_space; ++i)
      {
#endif
        for (int j = -initial_space; j <= initial_space; ++j)
        {
          T cx = cx0 + j;
          T cy = cy0 + i;
          T poni1 = cy * pixel_y_;
          T poni2 = cx * pixel_x_;

          auto ret = ai::histogramAI(src, poni1, poni2, pixel_y_, pixel_x_, npt, min_count);

          auto bounds = xt::minmax(ret.second)();
          auto curr_max = static_cast<T>(bounds[1]);

#if defined(FOAM_USE_TBB)
          {
            std::scoped_lock lock(mtx);
#endif
            if (curr_max > max_s)
            {
              max_s = curr_max;
              cx_max = cx;
              cy_max = cy;
            }
#if defined(FOAM_USE_TBB)
          }
#endif
        }
      }
#if defined(FOAM_USE_TBB)
    }
  );
#endif

  return {cx_max, cy_max};
}

} //foam

#endif //FOAM_AZIMUTHAL_INTEGRATOR_H
