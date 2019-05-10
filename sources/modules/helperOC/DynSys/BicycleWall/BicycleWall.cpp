/*
 * Copyright (c) 2017, The Regents of the University of California (Regents).
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *
 *    1. Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *
 *    2. Redistributions in binary form must reproduce the above
 *       copyright notice, this list of conditions and the following
 *       disclaimer in the documentation and/or other materials provided
 *       with the distribution.
 *
 *    3. Neither the name of the copyright holder nor the names of its
 *       contributors may be used to endorse or promote products derived
 *       from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS AS IS
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 * Please contact the author(s) of this library if you have any questions.
 * Authors: Jaime F. Fisac   ( jfisac@eecs.berkeley.edu )
 */

///////////////////////////////////////////////////////////////////////////////
//
// DynSys subclass: relative dynamics between a bicycle model car and a
//                  dynamically extended (+acceleration) simple car
//
///////////////////////////////////////////////////////////////////////////////


#include <helperOC/DynSys/BicycleWall/BicycleWall.hpp>
#include <iostream>
#include <cmath>
#include <limits>
#include <algorithm>
#include <typeinfo>
#include <levelset/Grids/HJI_Grid.hpp>
using namespace helperOC;


BicycleWall::BicycleWall(
    const beacls::FloatVec& x,      // state of tracker (relative to planner)
    const beacls::IntegerVec& dims
) : DynSys(
      5, // states: [y_rel, psi_rel, Ux, Uy, r]
      2, // controls: [delta, Fx]
      1 // disturbances: on r_dot equation [r_d]
    ), dims(dims) {
    std::cout << "Constructed BicycleWall object." << std::endl;
    //!< Process control range
    if (x.size() != DynSys::get_nx()) {
      std::cerr << "Error: " << __func__ << " : Initial state does not have right dimension!" << std::endl;
    }
    //!< Process initial state
    DynSys::set_x(x);
    DynSys::push_back_xhist(x);
}

BicycleWall::BicycleWall(
    beacls::MatFStream* fs,
    beacls::MatVariable* variable_ptr
) :
    DynSys(fs, variable_ptr),
    dims(beacls::IntegerVec()) {
    beacls::IntegerVec dummy;
    load_vector(dims, std::string("dims"), dummy, true, fs, variable_ptr);
}

BicycleWall::~BicycleWall() {}

bool BicycleWall::operator==(const BicycleWall& rhs) const {
  if (this == &rhs) return true;
  else if (!DynSys::operator==(rhs)) return false;
  else if ((dims.size() != rhs.dims.size()) || !std::equal(dims.cbegin(), dims.cend(), rhs.dims.cbegin())) return false;
  return true;
}

bool BicycleWall::operator==(const DynSys& rhs) const {
    if (this == &rhs) return true;
    else if (typeid(*this) != typeid(rhs)) return false;
    else return operator==(dynamic_cast<const BicycleWall&>(rhs));
}

bool BicycleWall::save(
    beacls::MatFStream* fs,
    beacls::MatVariable* variable_ptr) {

  bool result = DynSys::save(fs, variable_ptr);
  if (!dims.empty()) result &= save_vector(dims, std::string("dims"), beacls::IntegerVec(), true, fs, variable_ptr);
  return result;
}

bool BicycleWall::optCtrl(
    std::vector<beacls::FloatVec>& uOpts,
    const FLOAT_TYPE,
    const std::vector<beacls::FloatVec::const_iterator>& x_ites,
    const std::vector<const FLOAT_TYPE*>& deriv_ptrs,
    const beacls::IntegerVec&, // xsizes
    const beacls::IntegerVec& deriv_sizes,
    const helperOC::DynSys_UMode_Type uMode) const {
  const helperOC::DynSys_UMode_Type modified_uMode =
    (uMode == helperOC::DynSys_UMode_Default) ?
    helperOC::DynSys_UMode_Max : uMode;

  // uOpt = delta, Fx
  // Why is this needed? It is found in Plane.cpp
  for (size_t dim = 0; dim < 5; ++dim) {
    if (deriv_sizes[dim] == 0 || deriv_ptrs[dim] == NULL) {
      return false;
    }
  }

  uOpts.resize(get_nu());
  uOpts[0].resize(deriv_sizes[0]);
  uOpts[1].resize(deriv_sizes[0]);

  FLOAT_TYPE Fxf_coeff, Fxr_coeff, Fyf_coeff, Fyr_coeff;            // terms that depend on the control inputs Fx and/or delta.
  FLOAT_TYPE frac_1, frac_2, di, diOpt, Fxi, FxOpt, value, valueOpt, Fxfi, Fxri, afi, ari, Fzfi, Fzri, Fyfi, Fyri;
  FLOAT_TYPE sign = (modified_uMode == helperOC::DynSys_UMode_Max) ? 1 : -1;
  // Yr, psi_r, Ux, Uy,  r
  // 0    1      2   3   4

  // grid for Fx
  size_t N = 40;
  // grid for delta (+/- 18 degrees)
  size_t K = 18;

  for (size_t i = 0; i < deriv_sizes[0]; ++i) {

    valueOpt = (sign > 0) ? std::numeric_limits<FLOAT_TYPE>::lowest() :
                            std::numeric_limits<FLOAT_TYPE>::max();
    FxOpt = 0;
    diOpt = 0;
    for (size_t k = 0; k < K; ++k) {
      frac_1 = ((FLOAT_TYPE)k)/((FLOAT_TYPE)(K-1));
      di = frac_1 * X1::d_max - X1::d_max * (1.0 - frac_1);

      for (size_t n = 0; n < N; ++n) {

        frac_2 = ((FLOAT_TYPE)n)/((FLOAT_TYPE)(N-1));
        Fxi = frac_2 * X1::maxFx + X1::minFx * (1.0 - frac_2);
        Fxfi = getFxf(Fxi);
        Fxri = getFxr(Fxi);
        afi = std::atan2(x_ites[3][i] + X1::a * x_ites[4][i], x_ites[2][i]) - di;
        ari = std::atan2(x_ites[3][i] - X1::b * x_ites[4][i], x_ites[2][i]);
        Fzfi = (X1::m * X1::G * X1::b - X1::h * Fxi) / X1::L;
        Fzri = (X1::m * X1::G * X1::a + X1::h * Fxi) / X1::L;
        Fyfi = fialaTireModel(afi, X1::Caf, X1::mu, Fxfi, Fzfi);
        Fyri = fialaTireModel(ari, X1::Car, X1::mu, Fxri, Fzri);

        Fxf_coeff = deriv_ptrs[2][i] / X1::m * std::cos(di) + deriv_ptrs[3][i] / X1::m * std::sin(di) + deriv_ptrs[4][i] / X1::Izz * X1::a * std::sin(di);
        Fxr_coeff = deriv_ptrs[2][i] / X1::m;
        Fyf_coeff = -deriv_ptrs[2][i] / X1::m * std::sin(di) + deriv_ptrs[3][i] / X1::m * std::cos(di) + deriv_ptrs[4][i] / X1::Izz * X1::a * std::cos(di);
        Fyr_coeff = deriv_ptrs[3][i] / X1::m - deriv_ptrs[4][i] / X1::Izz * X1::b;

        value = Fxf_coeff * Fxfi + Fxr_coeff * Fxri + Fyf_coeff * Fyfi + Fyr_coeff * Fyri;

        if ( ((modified_uMode == helperOC::DynSys_UMode_Max) && (value > valueOpt)) ||
             ((modified_uMode == helperOC::DynSys_UMode_Min) && (value < valueOpt)) ) {
          diOpt = di;
          FxOpt = Fxi;
          valueOpt = value;
        }
      }
    }
    uOpts[0][i] = diOpt;
    uOpts[1][i] = FxOpt;
  }
  return true;
}

bool BicycleWall::optDstb(
    std::vector<beacls::FloatVec>& dOpts,
    const FLOAT_TYPE,
    const std::vector<beacls::FloatVec::const_iterator>& x_ites,
    const std::vector<const FLOAT_TYPE*>& deriv_ptrs,
    const beacls::IntegerVec&,
    const beacls::IntegerVec& deriv_sizes,
    const helperOC::DynSys_DMode_Type dMode) const {

  const helperOC::DynSys_DMode_Type modified_dMode =
    (dMode == helperOC::DynSys_DMode_Default) ?
    helperOC::DynSys_DMode_Min : dMode;

  if ((modified_dMode != helperOC::DynSys_DMode_Max) &&
      (modified_dMode != helperOC::DynSys_DMode_Min)) {
    std::cerr << "Unknown dMode!: " << modified_dMode << std::endl;
    return false;
  }

  // Why is this needed? It is found in Plane.cpp
  for (size_t dim = 0; dim < 5; ++dim) {
    if (deriv_sizes[dim] == 0 || deriv_ptrs[dim] == NULL) {
      return false;
    }
  }

  dOpts.resize(get_nd());
  dOpts[0].resize(deriv_sizes[0]);

  FLOAT_TYPE dVr, r_d_lim;
  FLOAT_TYPE sign = (modified_dMode == helperOC::DynSys_DMode_Max) ? 1 : -1;

  r_d_lim = 0.1;

  for (size_t i = 0; i < deriv_sizes[0]; ++i) {
    dVr = deriv_ptrs[4][i];
    // dOpts[0][i] = (dVr >= 0) ? sign * r_d_lim: -sign * r_d_lim;
    dOpts[0][i] = 0.0;
  }

  return true;
}

FLOAT_TYPE BicycleWall::fialaTireModel(const FLOAT_TYPE a,
                                         const FLOAT_TYPE Ca,
                                         const FLOAT_TYPE mu,
                                         const FLOAT_TYPE Fx,
                                         const FLOAT_TYPE Fz) const {
  FLOAT_TYPE Fmax = mu * Fz;
  if (std::abs(Fx) >= Fmax)
    return 0;
  else {
    FLOAT_TYPE Fymax = std::sqrt(Fmax * Fmax - Fx * Fx);
    FLOAT_TYPE tana = std::tan(a);
    FLOAT_TYPE tana_slide = 3 * Fymax / Ca;
    FLOAT_TYPE ratio = std::abs(tana / tana_slide);
    if (ratio < 1)
      return -Ca * tana * (1 - ratio + ratio * ratio / 3);
    else
      return -std::copysign(Fymax, tana);
  }
}

bool BicycleWall::dynamics_cell_helper(
    std::vector<beacls::FloatVec>& dxs,
    const beacls::FloatVec::const_iterator& y,
    const beacls::FloatVec::const_iterator& psi,
    const beacls::FloatVec::const_iterator& Ux,
    const beacls::FloatVec::const_iterator& Uy,
    const beacls::FloatVec::const_iterator& r,
    const std::vector<beacls::FloatVec >& us,
    const std::vector<beacls::FloatVec >& ds,
    const size_t size_y,
    const size_t size_psi,
    const size_t size_Ux,
    const size_t size_Uy,
    const size_t size_r,
    const size_t dim) const {

  beacls::FloatVec& dx_i = dxs[dim];
  bool result = true;

  switch (dims[dim]) {
    case 0: { // y_dot = Ux * sin(psi) + Uy * cos(psi)
      dx_i.resize(size_y);

      for (size_t i = 0; i < size_y; ++i) {
        dx_i[i] = Ux[i] * std::sin(psi[i]) + Uy[i] * std::cos(psi[i]);
      }
    } break;

    case 1: { // psi_dot = r
      dx_i.resize(size_psi);

      for (size_t i = 0; i < size_psi; ++i) {
        dx_i[i] = r[i];
      }

    } break;

    case 2: {   // Ux_dot = (Fxf * cos(delta) - Fyf * sin(delta) + Fxr + Fx_drag) / m + r * Uy
      dx_i.resize(size_Ux);
      // get delta and Fx control inputs
      const beacls::FloatVec&  d = us[0];
      const beacls::FloatVec& Fx = us[1];

      // variables needed for computation
      FLOAT_TYPE di, Fxi, Fx_dragi, Fxfi, Fxri, afi, Fzfi, Fyfi;

      // looping through
      for (size_t i = 0; i < size_Ux; ++i) {
        // checking if it's a vector or not.
        if (d.size() == size_Ux)
          di = d[i];
        else
          di = d[0];
        if (Fx.size() == size_Ux)
          Fxi = Fx[i];
        else
          Fxi = Fx[0];

        // computing relevant variables
        Fxfi = getFxf(Fxi);
        Fxri = getFxr(Fxi);
        afi = std::atan2(Uy[i] + X1::a * r[i], Ux[i]) - di;
        Fzfi = (X1::m * X1::G * X1::b - X1::h * Fxi) / X1::L;
        Fyfi = fialaTireModel(afi, X1::Caf, X1::mu, Fxfi, Fzfi);
        Fx_dragi = -X1::Cd0 - Ux[i] * (X1::Cd1 + X1::Cd2 * Ux[i]);

        dx_i[i] = (Fxfi * std::cos(di) - Fyfi * std::sin(di) + Fxri + Fx_dragi) / X1::m + r[i] * Uy[i];
      }
    } break;

    case 3: { // Uy_dot = (Fyf * cos(delta) + Fyr + Fxf * sin(delta)) / m - r * Ux
      dx_i.resize(size_Uy);
      const beacls::FloatVec&  d = us[0];
      const beacls::FloatVec& Fx = us[1];
      FLOAT_TYPE di, Fxi, Fxfi, Fxri, afi, ari, Fzfi, Fzri, Fyfi, Fyri;

      for (size_t i = 0; i < size_Uy; ++i) {
        if (d.size() == size_Uy)
          di = d[i];
        else
          di = d[0];
        if (Fx.size() == size_Uy)
          Fxi = Fx[i];
        else
          Fxi = Fx[0];
        Fxfi = getFxf(Fxi);
        Fxri = getFxr(Fxi);
        afi = std::atan2(Uy[i] + X1::a * r[i], Ux[i]) - di;
        ari = std::atan2(Uy[i] - X1::b * r[i], Ux[i]);
        Fzfi = (X1::m * X1::G * X1::b - X1::h * Fxi) / X1::L;
        Fzri = (X1::m * X1::G * X1::a + X1::h * Fxi) / X1::L;
        Fyfi = fialaTireModel(afi, X1::Caf, X1::mu, Fxfi, Fzfi);
        Fyri = fialaTireModel(ari, X1::Car, X1::mu, Fxri, Fzri);
        dx_i[i] = (Fyfi * std::cos(di) + Fyri + Fxfi * std::sin(di)) / X1::m - r[i] * Ux[i];
      }
    } break;

    case 4: { // r_dot = (a * Fyf * cos(delta) + a * Fxf * sin(delta) - b * Fyr) / Izz
      dx_i.resize(size_r);
      const beacls::FloatVec&  d = us[0];
      const beacls::FloatVec& Fx = us[1];
      FLOAT_TYPE di, Fxi, Fxfi, Fxri, afi, ari, Fzfi, Fzri, Fyfi, Fyri;

      for (size_t i = 0; i < size_r; ++i) {
        if (d.size() == size_r)
          di = d[i];
        else
          di = d[0];
        if (Fx.size() == size_r)
          Fxi = Fx[i];
        else
          Fxi = Fx[0];
        Fxfi = getFxf(Fxi);
        Fxri = getFxr(Fxi);
        afi = std::atan2(Uy[i] + X1::a * r[i], Ux[i]) - di;
        ari = std::atan2(Uy[i] - X1::b * r[i], Ux[i]);
        Fzfi = (X1::m * X1::G * X1::b - X1::h * Fxi) / X1::L;
        Fzri = (X1::m * X1::G * X1::a + X1::h * Fxi) / X1::L;
        Fyfi = fialaTireModel(afi, X1::Caf, X1::mu, Fxfi, Fzfi);
        Fyri = fialaTireModel(ari, X1::Car, X1::mu, Fxri, Fzri);
        dx_i[i] = (X1::a * Fyfi * std::cos(di) + X1::a * Fxfi * std::sin(di) - X1::b * Fyri) / X1::Izz;
      }
    } break;

    default:
      std::cerr << "Only dimension 1-5 are defined for dynamics of BicycleWall!" << std::endl;
      result = false;
      break;
  }
  return result;
}

bool BicycleWall::dynamics(
    std::vector<beacls::FloatVec>& dx,
    const FLOAT_TYPE,
    const std::vector<beacls::FloatVec::const_iterator>& x_ites,
    const std::vector<beacls::FloatVec>& us,
    const std::vector<beacls::FloatVec>& ds,
    const beacls::IntegerVec& x_sizes,
    const size_t dst_target_dim) const {

  bool result = true;
  // Compute dynamics for all components.
  if (dst_target_dim == std::numeric_limits<size_t>::max()) {
    for (size_t dim = 0; dim < 5; ++dim) {
      // printf("Dimension %zu \n", dim);
      result &= dynamics_cell_helper(
        dx, x_ites[0], x_ites[1], x_ites[2], x_ites[3], x_ites[4], us, ds,
        x_sizes[0], x_sizes[1], x_sizes[2], x_sizes[3], x_sizes[4], dim);
    }
  }
  // Compute dynamics for a single, specified component.
  else
  {
    if (dst_target_dim < dims.size()) {
      // printf("Target dimension %zu \n", dst_target_dim);
      result &= dynamics_cell_helper(
        dx, x_ites[0], x_ites[1], x_ites[2], x_ites[3], x_ites[4], us, ds,
        x_sizes[0], x_sizes[1], x_sizes[2], x_sizes[3], x_sizes[4], dst_target_dim);
    } else {
      std::cerr << "Invalid target dimension for dynamics: " << dst_target_dim << std::endl;
      result = false;
    }
  }
  return result;
}

#if defined(USER_DEFINED_GPU_DYNSYS_FUNC)
bool BicycleWall::optCtrl_cuda(
  std::vector<beacls::UVec>& u_uvecs,
  const FLOAT_TYPE,
  const std::vector<beacls::UVec>& x_uvecs,
  const std::vector<beacls::UVec>& deriv_uvecs,
  const helperOC::DynSys_UMode_Type uMode
) const {
  const helperOC::DynSys_UMode_Type modified_uMode = (uMode == helperOC::DynSys_UMode_Default) ? helperOC::DynSys_UMode_Max : uMode;
  if (x_uvecs.size() < 5 || x_uvecs[0].empty() || x_uvecs[1].empty() || x_uvecs[2].empty() ||
      x_uvecs[3].empty() || x_uvecs[4].empty() ||
      deriv_uvecs.size() < 5 || deriv_uvecs[0].empty() || deriv_uvecs[1].empty() || deriv_uvecs[2].empty() ||
      deriv_uvecs[3].empty() || deriv_uvecs[4].empty()) return false;
  return BicycleWall_CUDA::optCtrl_execute_cuda(u_uvecs, x_uvecs, deriv_uvecs, modified_uMode);
}
bool BicycleWall::optDstb_cuda(
  std::vector<beacls::UVec>& d_uvecs,
  const FLOAT_TYPE,
  const std::vector<beacls::UVec>& x_uvecs,
  const std::vector<beacls::UVec>& deriv_uvecs,
  const helperOC::DynSys_DMode_Type dMode
) const {
  const helperOC::DynSys_DMode_Type modified_dMode = (dMode == helperOC::DynSys_DMode_Default) ? helperOC::DynSys_DMode_Min : dMode;
  if (x_uvecs.size() < 5 || x_uvecs[0].empty() || x_uvecs[1].empty() || x_uvecs[2].empty() ||
      x_uvecs[3].empty() || x_uvecs[4].empty() ||
      deriv_uvecs.size() < 5 || deriv_uvecs[0].empty() || deriv_uvecs[1].empty() || deriv_uvecs[2].empty() ||
      deriv_uvecs[3].empty() || deriv_uvecs[4].empty()) return false;
  return BicycleWall_CUDA::optDstb_execute_cuda(d_uvecs, x_uvecs, deriv_uvecs, modified_dMode);
}
bool BicycleWall::dynamics_cuda(
  std::vector<beacls::UVec>& dx_uvecs,
  const FLOAT_TYPE,
  const std::vector<beacls::UVec>& x_uvecs,
  const std::vector<beacls::UVec>& u_uvecs,
  const std::vector<beacls::UVec>& d_uvecs,
  const size_t dst_target_dim
) const {
  bool result = true;
  if (dst_target_dim == std::numeric_limits<size_t>::max()) {
    result &= BicycleWall_CUDA::dynamics_cell_helper_execute_cuda_dimAll(dx_uvecs, x_uvecs, u_uvecs, d_uvecs);
  }
  else
  {
    if (dst_target_dim < x_uvecs.size()) {
      return BicycleWall_CUDA::dynamics_cell_helper_execute_cuda(dx_uvecs[dst_target_dim], x_uvecs, u_uvecs, d_uvecs, dst_target_dim);
    } else {
      std::cerr << "Invalid target dimension for dynamics: " << dst_target_dim << std::endl;
      result = false;
    }
  }
  return result;
}
#endif /* defined(USER_DEFINED_GPU_DYNSYS_FUNC) */
