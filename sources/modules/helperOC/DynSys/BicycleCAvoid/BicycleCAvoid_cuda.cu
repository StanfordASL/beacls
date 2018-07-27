// CUDA runtime
#include <cuda_runtime.h>

#include <typedef.hpp>
#include <cuda_macro.hpp>
#include <thrust/device_ptr.h>
#include <thrust/tuple.h>
#include <thrust/for_each.h>
#include <thrust/transform.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/system/cuda/execution_policy.h>
#include "BicycleCAvoid_cuda.hpp"
#include <vector>
#include <algorithm>
#include <helperOC/DynSys/DynSys/DynSys.hpp>
#include <Core/UVec.hpp>
#include <Core/CudaStream.hpp>

#if defined(WITH_GPU)
#if defined(USER_DEFINED_GPU_DYNSYS_FUNC)
namespace BicycleCAvoid_CUDA {

struct Get_optCtrl {
public:
    const FLOAT_TYPE sign;
    Get_optCtrl(const FLOAT_TYPE sign) : sign(sign) {}
    template<typename Tuple>
    __host__ __device__
    void operator()(Tuple v) const
    {
        const FLOAT_TYPE Ux     = thrust::get<2>(v);
        const FLOAT_TYPE Uy     = thrust::get<3>(v);
        const FLOAT_TYPE r      = thrust::get<4>(v);
        const FLOAT_TYPE lam_Ux = thrust::get<5>(v);
        const FLOAT_TYPE lam_Uy = thrust::get<6>(v);
        const FLOAT_TYPE lam_r  = thrust::get<7>(v);

        const FLOAT_TYPE A = lam_Ux / X1::m;                              // Fx coefficient
        const FLOAT_TYPE B = lam_Uy / X1::m + X1::a * lam_r / X1::Izz;    // Fyf coefficient
        const FLOAT_TYPE C = lam_Uy / X1::m - X1::b * lam_r / X1::Izz;    // Fyr coefficient

        const FLOAT_TYPE dOpt = ((B >= 0) - (B < 0))*sign*X1::d_max;

        FLOAT_TYPE valueOpt = sign*1e6;
        FLOAT_TYPE FxOpt = 0;

        for (size_t n = 0; n < 50; ++n) {    // 100% willing to unroll this loop
            FLOAT_TYPE frac = ((FLOAT_TYPE)n)/((FLOAT_TYPE)(50-1));
            FLOAT_TYPE Fx = frac * X1::maxFx + X1::minFx * (1.0 - frac);

            FLOAT_TYPE Fxf = (Fx >= 0)*Fx*X1::fwd_frac + (Fx < 0)*Fx*X1::fwb_frac;    // no branches!
            FLOAT_TYPE Fxr = (Fx >= 0)*Fx*X1::rwd_frac + (Fx < 0)*Fx*X1::rwb_frac;    // no branches!
            FLOAT_TYPE af  = atan2_float_type<FLOAT_TYPE>(Uy + X1::a * r, Ux) - dOpt;
            FLOAT_TYPE ar  = atan2_float_type<FLOAT_TYPE>(Uy - X1::b * r, Ux);
            FLOAT_TYPE Fzf = (X1::m * X1::G * X1::b - X1::h * Fx) / X1::L;
            FLOAT_TYPE Fzr = (X1::m * X1::G * X1::a + X1::h * Fx) / X1::L;
            // Fyf (inlined BicycleCAvoid::fialaTireModel(af, X1::Caf, X1::mu, Fxf, Fzf))
            FLOAT_TYPE a  = af;
            FLOAT_TYPE Ca = X1::Caf;
            FLOAT_TYPE mu = X1::mu;
            FLOAT_TYPE fx = Fxf;    // avoid name conflict with input Fx; damn inlining
            FLOAT_TYPE Fz = Fzf;
            FLOAT_TYPE Fmax = mu * Fz;
            FLOAT_TYPE Fymax = sqrt_float_type<FLOAT_TYPE>((abs_float_type<FLOAT_TYPE>(fx) < Fmax)*(Fmax * Fmax - fx * fx));    // no branches!
            FLOAT_TYPE tana  = tan_float_type<FLOAT_TYPE>(a);
            FLOAT_TYPE ratio = abs_float_type<FLOAT_TYPE>(tana * Ca / (3 * (Fymax > 0) * Fymax + (Fymax <= 0))) + (Fymax <= 0); // ratio >= 1 if Fymax <= 0
            FLOAT_TYPE Fyf = (ratio < 1)*(-Ca * tana * (1 - ratio + ratio * ratio / 3)) + (ratio >= 1)*(-copysign_float_type<FLOAT_TYPE>(Fymax, tana));
            // Fyr (inlined BicycleCAvoid::fialaTireModel(ar, X1::Car, X1::mu, Fxr, Fzr))
            a  = ar;
            Ca = X1::Car;
            mu = X1::mu;
            fx = Fxr;
            Fz = Fzr;
            Fmax = mu*Fz;
            Fymax = sqrt_float_type<FLOAT_TYPE>((abs_float_type<FLOAT_TYPE>(fx) < Fmax)*(Fmax * Fmax - fx * fx));    // no branches!
            tana  = tan_float_type<FLOAT_TYPE>(a);
            ratio = abs_float_type<FLOAT_TYPE>(tana * Ca / (3 * (Fymax > 0) * Fymax + (Fymax <= 0))) + (Fymax <= 0); // ratio >= 1 if Fymax <= 0
            FLOAT_TYPE Fyr = (ratio < 1)*(-Ca * tana * (1 - ratio + ratio * ratio / 3)) + (ratio >= 1)*(-copysign_float_type<FLOAT_TYPE>(Fymax, tana));
            FLOAT_TYPE value = A * Fx + B * Fyf + C * Fyr;
            if ( ((sign > 0) && (value > valueOpt)) || ((sign < 0) && (value < valueOpt)) ) {
                FxOpt = Fx;
                valueOpt = value;
            }
        }
        thrust::get<0>(v) = dOpt;
        thrust::get<1>(v) = FxOpt;
    }
};

bool optCtrl_execute_cuda(
    std::vector<beacls::UVec>& u_uvecs,
    const std::vector<beacls::UVec>& x_uvecs,
    const std::vector<beacls::UVec>& deriv_uvecs,
    const helperOC::DynSys_UMode_Type uMode
) {
    bool result = true;
    beacls::UVec& d_uvec            = u_uvecs[0];
    beacls::UVec& Fx_uvec           = u_uvecs[1];
    const beacls::UVec& Ux_uvec     = x_uvecs[3];        // a bit worried about the size of this uvec; TODO assert?
    const beacls::UVec& Uy_uvec     = x_uvecs[4];        // a bit worried about the size of this uvec; TODO assert?
    const beacls::UVec& r_uvec      = x_uvecs[6];        // a bit worried about the size of this uvec; TODO assert?
    const beacls::UVec& lam_Ux_uvec = deriv_uvecs[3];
    const beacls::UVec& lam_Uy_uvec = deriv_uvecs[4];
    const beacls::UVec& lam_r_uvec  = deriv_uvecs[6];

    beacls::reallocateAsSrc(d_uvec , deriv_uvecs[0]);
    beacls::reallocateAsSrc(Fx_uvec, deriv_uvecs[0]);

    FLOAT_TYPE* d_ptr            = beacls::UVec_<FLOAT_TYPE>(d_uvec).ptr();
    FLOAT_TYPE* Fx_ptr           = beacls::UVec_<FLOAT_TYPE>(Fx_uvec).ptr();
    const FLOAT_TYPE* Ux_ptr     = beacls::UVec_<FLOAT_TYPE>(Ux_uvec).ptr();
    const FLOAT_TYPE* Uy_ptr     = beacls::UVec_<FLOAT_TYPE>(Uy_uvec).ptr();
    const FLOAT_TYPE* r_ptr      = beacls::UVec_<FLOAT_TYPE>(r_uvec).ptr();
    const FLOAT_TYPE* lam_Ux_ptr = beacls::UVec_<FLOAT_TYPE>(lam_Ux_uvec).ptr();
    const FLOAT_TYPE* lam_Uy_ptr = beacls::UVec_<FLOAT_TYPE>(lam_Uy_uvec).ptr();
    const FLOAT_TYPE* lam_r_ptr  = beacls::UVec_<FLOAT_TYPE>(lam_r_uvec).ptr();

    if ((uMode == helperOC::DynSys_UMode_Max) || (uMode == helperOC::DynSys_UMode_Min)) {
        const FLOAT_TYPE sign = (uMode == helperOC::DynSys_UMode_Max) ? 1 : -1;
        if (beacls::is_cuda(lam_Ux_uvec) && beacls::is_cuda(lam_Uy_uvec) && beacls::is_cuda(lam_r_uvec)) {
            thrust::device_ptr<FLOAT_TYPE> d_dev_ptr            = thrust::device_pointer_cast(d_ptr);
            thrust::device_ptr<FLOAT_TYPE> Fx_dev_ptr           = thrust::device_pointer_cast(Fx_ptr);
            thrust::device_ptr<const FLOAT_TYPE> Ux_dev_ptr     = thrust::device_pointer_cast(Ux_ptr);
            thrust::device_ptr<const FLOAT_TYPE> Uy_dev_ptr     = thrust::device_pointer_cast(Uy_ptr);
            thrust::device_ptr<const FLOAT_TYPE> r_dev_ptr      = thrust::device_pointer_cast(r_ptr);
            thrust::device_ptr<const FLOAT_TYPE> lam_Ux_dev_ptr = thrust::device_pointer_cast(lam_Ux_ptr);
            thrust::device_ptr<const FLOAT_TYPE> lam_Uy_dev_ptr = thrust::device_pointer_cast(lam_Uy_ptr);
            thrust::device_ptr<const FLOAT_TYPE> lam_r_dev_ptr  = thrust::device_pointer_cast(lam_r_ptr);
            cudaStream_t d_stream = beacls::get_stream(d_uvec);
            Fx_uvec.set_cudaStream(d_uvec.get_cudaStream());
            auto Tuple = thrust::make_tuple(d_dev_ptr,
                                            Fx_dev_ptr,
                                            Ux_dev_ptr,
                                            Uy_dev_ptr,
                                            r_dev_ptr,
                                            lam_Ux_dev_ptr,
                                            lam_Uy_dev_ptr,
                                            lam_r_dev_ptr);
            auto Iterator = thrust::make_zip_iterator(Tuple);
            thrust::for_each(thrust::cuda::par.on(d_stream), Iterator, Iterator + d_uvec.size(), Get_optCtrl(sign));
        } else {
            const FLOAT_TYPE Ux     = Ux_ptr[0];
            const FLOAT_TYPE Uy     = Uy_ptr[0];
            const FLOAT_TYPE r      = r_ptr[0];
            const FLOAT_TYPE lam_Ux = lam_Ux_ptr[0];
            const FLOAT_TYPE lam_Uy = lam_Uy_ptr[0];
            const FLOAT_TYPE lam_r  = lam_r_ptr[0];

            const FLOAT_TYPE A = lam_Ux / X1::m;                              // Fx coefficient
            const FLOAT_TYPE B = lam_Uy / X1::m + X1::a * lam_r / X1::Izz;    // Fyf coefficient
            const FLOAT_TYPE C = lam_Uy / X1::m - X1::b * lam_r / X1::Izz;    // Fyr coefficient

            const FLOAT_TYPE dOpt = ((B >= 0) - (B < 0))*sign*X1::d_max;

            FLOAT_TYPE valueOpt = sign*1e6;
            FLOAT_TYPE FxOpt = 0;

            for (size_t n = 0; n < 50; ++n) {    // 100% willing to unroll this loop
                FLOAT_TYPE frac = ((FLOAT_TYPE)n)/((FLOAT_TYPE)(50-1));
                FLOAT_TYPE Fx = frac * X1::maxFx + X1::minFx * (1.0 - frac);

                FLOAT_TYPE Fxf = (Fx >= 0)*Fx*X1::fwd_frac + (Fx < 0)*Fx*X1::fwb_frac;    // no branches!
                FLOAT_TYPE Fxr = (Fx >= 0)*Fx*X1::rwd_frac + (Fx < 0)*Fx*X1::rwb_frac;    // no branches!
                FLOAT_TYPE af  = atan2_float_type<FLOAT_TYPE>(Uy + X1::a * r, Ux) - dOpt;
                FLOAT_TYPE ar  = atan2_float_type<FLOAT_TYPE>(Uy - X1::b * r, Ux);
                FLOAT_TYPE Fzf = (X1::m * X1::G * X1::b - X1::h * Fx) / X1::L;
                FLOAT_TYPE Fzr = (X1::m * X1::G * X1::a + X1::h * Fx) / X1::L;
                // Fyf (inlined BicycleCAvoid::fialaTireModel(af, X1::Caf, X1::mu, Fxf, Fzf))
                FLOAT_TYPE a  = af;
                FLOAT_TYPE Ca = X1::Caf;
                FLOAT_TYPE mu = X1::mu;
                FLOAT_TYPE fx = Fxf;    // avoid name conflict with input Fx; damn inlining
                FLOAT_TYPE Fz = Fzf;
                FLOAT_TYPE Fmax = mu * Fz;
                FLOAT_TYPE Fymax = sqrt_float_type<FLOAT_TYPE>((abs_float_type<FLOAT_TYPE>(fx) < Fmax)*(Fmax * Fmax - fx * fx));    // no branches!
                FLOAT_TYPE tana  = tan_float_type<FLOAT_TYPE>(a);
                FLOAT_TYPE ratio = abs_float_type<FLOAT_TYPE>(tana * Ca / (3 * (Fymax > 0) * Fymax + (Fymax <= 0))) + (Fymax <= 0); // ratio >= 1 if Fymax <= 0
                FLOAT_TYPE Fyf = (ratio < 1)*(-Ca * tana * (1 - ratio + ratio * ratio / 3)) + (ratio >= 1)*(-copysign_float_type<FLOAT_TYPE>(Fymax, tana));
                // Fyr (inlined BicycleCAvoid::fialaTireModel(ar, X1::Car, X1::mu, Fxr, Fzr))
                a  = ar;
                Ca = X1::Car;
                mu = X1::mu;
                fx = Fxr;
                Fz = Fzr;
                Fmax = mu*Fz;
                Fymax = sqrt_float_type<FLOAT_TYPE>((abs_float_type<FLOAT_TYPE>(fx) < Fmax)*(Fmax * Fmax - fx * fx));    // no branches!
                tana  = tan_float_type<FLOAT_TYPE>(a);
                ratio = abs_float_type<FLOAT_TYPE>(tana * Ca / (3 * (Fymax > 0) * Fymax + (Fymax <= 0))) + (Fymax <= 0); // ratio >= 1 if Fymax <= 0
                FLOAT_TYPE Fyr = (ratio < 1)*(-Ca * tana * (1 - ratio + ratio * ratio / 3)) + (ratio >= 1)*(-copysign_float_type<FLOAT_TYPE>(Fymax, tana));
                FLOAT_TYPE value = A * Fx + B * Fyf + C * Fyr;
                if ( ((sign > 0) && (value > valueOpt)) || ((sign < 0) && (value < valueOpt)) ) {
                    FxOpt = Fx;
                    valueOpt = value;
                }
            }
            d_ptr[0]  = dOpt;
            Fx_ptr[0] = FxOpt;
        }
    } else {
        std::cerr << "Unknown uMode!: " << uMode << std::endl;
        result = false;
    }
    return result;
}

struct Get_optDstb {
public:
    const FLOAT_TYPE sign;
    Get_optDstb(const FLOAT_TYPE sign) : sign(sign) {}
    template<typename Tuple>
    __host__ __device__
    void operator()(Tuple v) const
    {
        const FLOAT_TYPE V      = thrust::get<2>(v);
        const FLOAT_TYPE lam_w  = thrust::get<3>(v);
        const FLOAT_TYPE lam_Ax = thrust::get<4>(v);
        FLOAT_TYPE lam_Ay = lam_w / V;
        FLOAT_TYPE lam_norm = hypot_float_type<FLOAT_TYPE>(lam_Ax, lam_Ay);
        FLOAT_TYPE maxA = X1::maxA_approx;
        FLOAT_TYPE X1maxAx = X1::maxAx;    // not sure why I have to do this??
        FLOAT_TYPE X1maxP2mx = X1::maxP2mx;    // not sure why I have to do this??
        if (lam_norm < 0.001) {    // fuk tuples, i.e., I ain't debranching this
            thrust::get<0>(v) = 0;
            thrust::get<1>(v) = 0;
        } else {
            FLOAT_TYPE desAx = sign * lam_Ax * maxA / lam_norm;
            FLOAT_TYPE desAy = sign * lam_Ay * maxA / lam_norm;
            FLOAT_TYPE maxAx = min_float_type<FLOAT_TYPE>(X1maxAx, X1maxP2mx / V);  // max longitudinal acceleration
            FLOAT_TYPE maxAy = X1::w_per_v_max_lowspeed * V * V;                        // also bounded by maxA
            if (desAx > maxAx) {
                if (abs_float_type<FLOAT_TYPE>(desAy) < maxAy) {
                    maxAy = min_float_type<FLOAT_TYPE>(sqrt_float_type<FLOAT_TYPE>(maxA * maxA - maxAx * maxAx), maxAy);
                }
                thrust::get<0>(v) = copysign_float_type<FLOAT_TYPE>(maxAy, desAy) / V;
                thrust::get<1>(v) = maxAx;
            } else {
                if (abs_float_type<FLOAT_TYPE>(desAy) > maxAy) {
                    if (desAx > 0) {
                        maxAx = min_float_type<FLOAT_TYPE>(sqrt_float_type<FLOAT_TYPE>(maxA * maxA - maxAy * maxAy), maxAx);
                        thrust::get<0>(v) = copysign_float_type<FLOAT_TYPE>(maxAy, desAy) / V;
                        thrust::get<1>(v) = maxAx;
                    } else {
                        thrust::get<0>(v) = copysign_float_type<FLOAT_TYPE>(maxAy, desAy) / V;
                        thrust::get<1>(v) = -sqrt_float_type<FLOAT_TYPE>(maxA * maxA - maxAy * maxAy);
                    }
                } else {
                    thrust::get<0>(v) = desAy / V;
                    thrust::get<1>(v) = maxAx;
                }
            }
        }
    }
};

bool optDstb_execute_cuda(
    std::vector<beacls::UVec>& d_uvecs,
    const std::vector<beacls::UVec>& x_uvecs,
    const std::vector<beacls::UVec>& deriv_uvecs,
    const helperOC::DynSys_DMode_Type dMode
) {
    bool result = true;
    beacls::UVec& w_uvec = d_uvecs[0];
    beacls::UVec& a_uvec = d_uvecs[1];
    const beacls::UVec& V_uvec      = x_uvecs[5];        // a bit worried about the size of this uvec; TODO assert?
    const beacls::UVec& lam_w_uvec  = deriv_uvecs[2];
    const beacls::UVec& lam_Ax_uvec = deriv_uvecs[5];

    beacls::reallocateAsSrc(w_uvec, deriv_uvecs[0]);
    beacls::reallocateAsSrc(a_uvec, deriv_uvecs[0]);

    FLOAT_TYPE* w_ptr            = beacls::UVec_<FLOAT_TYPE>(w_uvec).ptr();
    FLOAT_TYPE* a_ptr            = beacls::UVec_<FLOAT_TYPE>(a_uvec).ptr();
    const FLOAT_TYPE* V_ptr      = beacls::UVec_<FLOAT_TYPE>(V_uvec).ptr();
    const FLOAT_TYPE* lam_w_ptr  = beacls::UVec_<FLOAT_TYPE>(lam_w_uvec).ptr();
    const FLOAT_TYPE* lam_Ax_ptr = beacls::UVec_<FLOAT_TYPE>(lam_Ax_uvec).ptr();

    if ((dMode == helperOC::DynSys_DMode_Max) || (dMode == helperOC::DynSys_DMode_Min)) {
        const FLOAT_TYPE sign = (dMode == helperOC::DynSys_DMode_Max) ? 1 : -1;
        if (beacls::is_cuda(lam_w_uvec) && beacls::is_cuda(lam_Ax_uvec)) {
            thrust::device_ptr<FLOAT_TYPE> w_dev_ptr            = thrust::device_pointer_cast(w_ptr);
            thrust::device_ptr<FLOAT_TYPE> a_dev_ptr            = thrust::device_pointer_cast(a_ptr);
            thrust::device_ptr<const FLOAT_TYPE> V_dev_ptr      = thrust::device_pointer_cast(V_ptr);
            thrust::device_ptr<const FLOAT_TYPE> lam_w_dev_ptr  = thrust::device_pointer_cast(lam_w_ptr);
            thrust::device_ptr<const FLOAT_TYPE> lam_Ax_dev_ptr = thrust::device_pointer_cast(lam_Ax_ptr);
            cudaStream_t d_stream = beacls::get_stream(w_uvec);
            a_uvec.set_cudaStream(w_uvec.get_cudaStream());
            auto Tuple = thrust::make_tuple(w_dev_ptr,
                                            a_dev_ptr,
                                            V_dev_ptr,
                                            lam_w_dev_ptr,
                                            lam_Ax_dev_ptr);
            auto Iterator = thrust::make_zip_iterator(Tuple);
            thrust::for_each(thrust::cuda::par.on(d_stream), Iterator, Iterator + w_uvec.size(), Get_optDstb(sign));
        } else {
            const FLOAT_TYPE V      = V_ptr[0];
            const FLOAT_TYPE lam_w  = lam_w_ptr[0];
            const FLOAT_TYPE lam_Ax = lam_Ax_ptr[0];
            FLOAT_TYPE lam_Ay = lam_w / V;
            FLOAT_TYPE lam_norm = hypot_float_type<FLOAT_TYPE>(lam_Ax, lam_Ay);
            FLOAT_TYPE maxA = X1::maxA_approx;
            if (lam_norm < 0.001) {    // fuk tuples, i.e., I ain't debranching this
                w_ptr[0] = 0;
                a_ptr[0] = 0;
            } else {
                FLOAT_TYPE desAx = sign * lam_Ax * maxA / lam_norm;
                FLOAT_TYPE desAy = sign * lam_Ay * maxA / lam_norm;
                FLOAT_TYPE maxAx = min_float_type<FLOAT_TYPE>(X1::maxAx, X1::maxP2mx / V);  // max longitudinal acceleration
                FLOAT_TYPE maxAy = X1::w_per_v_max_lowspeed * V * V;                        // also bounded by maxA
                if (desAx > maxAx) {
                    if (abs_float_type<FLOAT_TYPE>(desAy) < maxAy) {
                        maxAy = min_float_type<FLOAT_TYPE>(sqrt_float_type<FLOAT_TYPE>(maxA * maxA - maxAx * maxAx), maxAy);
                    }
                    w_ptr[0] = copysign_float_type<FLOAT_TYPE>(maxAy, desAy) / V;
                    a_ptr[0] = maxAx;
                } else {
                    if (abs_float_type<FLOAT_TYPE>(desAy) > maxAy) {
                        if (desAx > 0) {
                            maxAx = min_float_type<FLOAT_TYPE>(sqrt_float_type<FLOAT_TYPE>(maxA * maxA - maxAy * maxAy), maxAx);
                            w_ptr[0] = copysign_float_type<FLOAT_TYPE>(maxAy, desAy) / V;
                            a_ptr[0] = maxAx;
                        } else {
                            w_ptr[0] = copysign_float_type<FLOAT_TYPE>(maxAy, desAy) / V;
                            a_ptr[0] = -sqrt_float_type<FLOAT_TYPE>(maxA * maxA - maxAy * maxAy);
                        }
                    } else {
                        w_ptr[0] = desAy / V;
                        a_ptr[0] = maxAx;
                    }
                }
            }
        }
    } else {
        std::cerr << "Unknown dMode!: " << dMode << std::endl;
        result = false;
    }
    return result;
}

struct Get_dynamics_x_rel_y_rel
{
    Get_dynamics_x_rel_y_rel(){}
    template<typename Tuple>
    __host__ __device__
    void operator()(Tuple v) const
    {
        const FLOAT_TYPE psi_rel = thrust::get<2>(v);
        const FLOAT_TYPE x_rel   = thrust::get<3>(v);
        const FLOAT_TYPE y_rel   = thrust::get<4>(v);
        const FLOAT_TYPE Ux      = thrust::get<5>(v);
        const FLOAT_TYPE Uy      = thrust::get<6>(v);
        const FLOAT_TYPE V       = thrust::get<7>(v);
        const FLOAT_TYPE r       = thrust::get<8>(v);
        FLOAT_TYPE sin_psi_rel;
        FLOAT_TYPE cos_psi_rel;
        sincos_float_type<FLOAT_TYPE>(psi_rel, sin_psi_rel, cos_psi_rel);
        thrust::get<0>(v) = V * cos_psi_rel - Ux + y_rel * r;    // dx_rel
        thrust::get<1>(v) = V * sin_psi_rel - Uy - x_rel * r;    // dy_rel
    }
};

struct Get_dynamics_psi_rel_V__W_A
{
    Get_dynamics_psi_rel_V__W_A(){}
    template<typename Tuple>
    __host__ __device__
    void operator()(Tuple v) const
    {
        const FLOAT_TYPE r  = thrust::get<2>(v);
        const FLOAT_TYPE w  = thrust::get<3>(v);
        const FLOAT_TYPE a  = thrust::get<4>(v);
        thrust::get<0>(v) = w - r;    // dpsi_rel
        thrust::get<1>(v) = a;        // dV
    }
};

struct Get_dynamics_psi_rel_V__w_a
{
    const FLOAT_TYPE w;
    const FLOAT_TYPE a;
    Get_dynamics_psi_rel_V__w_a(const FLOAT_TYPE w, const FLOAT_TYPE a) : w(w), a(a) {}
    template<typename Tuple>
    __host__ __device__
    void operator()(Tuple v) const
    {
        const FLOAT_TYPE r  = thrust::get<2>(v);
        thrust::get<0>(v) = w - r;    // dpsi_rel
        thrust::get<1>(v) = a;        // dV
    }
};

struct Get_dynamics_Ux_Uy_r__D_FX
{
    Get_dynamics_Ux_Uy_r__D_FX(){}
    template<typename Tuple>
    __host__ __device__
    void operator()(Tuple v) const
    {
        const FLOAT_TYPE Ux = thrust::get<3>(v);
        const FLOAT_TYPE Uy = thrust::get<4>(v);
        const FLOAT_TYPE r  = thrust::get<5>(v);
        const FLOAT_TYPE d  = thrust::get<6>(v);
        const FLOAT_TYPE Fx = thrust::get<7>(v);
        FLOAT_TYPE Fxf = (Fx >= 0)*Fx*X1::fwd_frac + (Fx < 0)*Fx*X1::fwb_frac;    // no branches! (... apparently this is unnecessary)
        FLOAT_TYPE Fxr = (Fx >= 0)*Fx*X1::rwd_frac + (Fx < 0)*Fx*X1::rwb_frac;    // no branches!
        FLOAT_TYPE Fx_drag = -X1::Cd0 - Ux * (X1::Cd1 + X1::Cd2 * Ux);
        FLOAT_TYPE af  = atan2_float_type<FLOAT_TYPE>(Uy + X1::a * r, Ux) - d;
        FLOAT_TYPE ar = atan2_float_type<FLOAT_TYPE>(Uy - X1::b * r, Ux);
        FLOAT_TYPE Fzf = (X1::m * X1::G * X1::b - X1::h * Fx) / X1::L;
        FLOAT_TYPE Fzr = (X1::m * X1::G * X1::a + X1::h * Fx) / X1::L;
        // Fyf (inlined BicycleCAvoid::fialaTireModel(af, X1::Caf, X1::mu, Fxf, Fzf))
        FLOAT_TYPE a  = af;
        FLOAT_TYPE Ca = X1::Caf;
        FLOAT_TYPE mu = X1::mu;
        FLOAT_TYPE fx = Fxf;    // avoid name conflict with input Fx; damn inlining
        FLOAT_TYPE Fz = Fzf;
        FLOAT_TYPE Fmax = mu * Fz;
        FLOAT_TYPE Fymax = sqrt_float_type<FLOAT_TYPE>((abs_float_type<FLOAT_TYPE>(fx) < Fmax)*(Fmax * Fmax - fx * fx));    // no branches!
        FLOAT_TYPE tana  = tan_float_type<FLOAT_TYPE>(a);
        FLOAT_TYPE ratio = abs_float_type<FLOAT_TYPE>(tana * Ca / (3 * (Fymax > 0) * Fymax + (Fymax <= 0))) + (Fymax <= 0); // ratio >= 1 if Fymax <= 0
        FLOAT_TYPE Fyf = (ratio < 1)*(-Ca * tana * (1 - ratio + ratio * ratio / 3)) + (ratio >= 1)*(-copysign_float_type<FLOAT_TYPE>(Fymax, tana));
        // Fyr (inlined BicycleCAvoid::fialaTireModel(ar, X1::Car, X1::mu, Fxr, Fzr))
        a  = ar;
        Ca = X1::Car;
        mu = X1::mu;
        fx = Fxr;
        Fz = Fzr;
        Fmax = mu*Fz;
        Fymax = sqrt_float_type<FLOAT_TYPE>((abs_float_type<FLOAT_TYPE>(fx) < Fmax)*(Fmax * Fmax - fx * fx));    // no branches!
        tana  = tan_float_type<FLOAT_TYPE>(a);
        ratio = abs_float_type<FLOAT_TYPE>(tana * Ca / (3 * (Fymax > 0) * Fymax + (Fymax <= 0))) + (Fymax <= 0); // ratio >= 1 if Fymax <= 0
        FLOAT_TYPE Fyr = (ratio < 1)*(-Ca * tana * (1 - ratio + ratio * ratio / 3)) + (ratio >= 1)*(-copysign_float_type<FLOAT_TYPE>(Fymax, tana));
        thrust::get<0>(v) = (Fx + Fx_drag) / X1::m + r * Uy;          // dUx
        thrust::get<1>(v) = (Fyf + Fyr) / X1::m - r * Ux;             // dUy
        thrust::get<2>(v) = (X1::a * Fyf - X1::b * Fyr) / X1::Izz;    // dr
    }
};

struct Get_dynamics_x_rel
{
    Get_dynamics_x_rel(){}
    template<typename Tuple>
    __host__ __device__
    void operator()(Tuple v) const
    {
        const FLOAT_TYPE psi_rel = thrust::get<1>(v);
        const FLOAT_TYPE y_rel   = thrust::get<2>(v);
        const FLOAT_TYPE Ux      = thrust::get<3>(v);
        const FLOAT_TYPE V       = thrust::get<4>(v);
        const FLOAT_TYPE r       = thrust::get<5>(v);
        thrust::get<0>(v) = V * cos_float_type<FLOAT_TYPE>(psi_rel) - Ux + y_rel * r;    // dx_rel
    }
};

struct Get_dynamics_y_rel
{
    Get_dynamics_y_rel(){}
    template<typename Tuple>
    __host__ __device__
    void operator()(Tuple v) const
    {
        const FLOAT_TYPE psi_rel = thrust::get<1>(v);
        const FLOAT_TYPE x_rel   = thrust::get<2>(v);
        const FLOAT_TYPE Uy      = thrust::get<3>(v);
        const FLOAT_TYPE V       = thrust::get<4>(v);
        const FLOAT_TYPE r       = thrust::get<5>(v);
        thrust::get<0>(v) = V * sin_float_type<FLOAT_TYPE>(psi_rel) - Uy - x_rel * r;    // dy_rel
    }
};

struct Get_dynamics_psi_rel__W
{
    Get_dynamics_psi_rel__W(){}
    template<typename Tuple>
    __host__ __device__
    void operator()(Tuple v) const
    {
        const FLOAT_TYPE r  = thrust::get<1>(v);
        const FLOAT_TYPE w  = thrust::get<2>(v);
        thrust::get<0>(v) = w - r;    // dpsi_rel
    }
};

struct Get_dynamics_psi_rel__w
{
    const FLOAT_TYPE w;
    Get_dynamics_psi_rel__w(const FLOAT_TYPE w) : w(w) {}
    template<typename Tuple>
    __host__ __device__
    void operator()(Tuple v) const
    {
        const FLOAT_TYPE r  = thrust::get<1>(v);
        thrust::get<0>(v) = w - r;    // dpsi_rel
    }
};

struct Get_dynamics_V__A
{
    Get_dynamics_V__A(){}
    template<typename Tuple>
    __host__ __device__
    void operator()(Tuple v) const
    {
        const FLOAT_TYPE a  = thrust::get<1>(v);
        thrust::get<0>(v) = a;    // dV
    }
};

struct Get_dynamics_V__a
{
    const FLOAT_TYPE a;
    Get_dynamics_V__a(const FLOAT_TYPE a) : a(a) {}
    template<typename Tuple>
    __host__ __device__
    void operator()(Tuple v) const
    {
        thrust::get<0>(v) = a;    // dV
    }
};

struct Get_dynamics_Ux__FX
{
    Get_dynamics_Ux__FX(){}
    template<typename Tuple>
    __host__ __device__
    void operator()(Tuple v) const
    {
        const FLOAT_TYPE Ux = thrust::get<1>(v);
        const FLOAT_TYPE Uy = thrust::get<2>(v);
        const FLOAT_TYPE r  = thrust::get<3>(v);
        const FLOAT_TYPE Fx = thrust::get<4>(v);
        FLOAT_TYPE Fx_drag = -X1::Cd0 - Ux * (X1::Cd1 + X1::Cd2 * Ux);
        thrust::get<0>(v) = (Fx + Fx_drag) / X1::m + r * Uy;    // dUx
    }
};

struct Get_dynamics_Uy__D_FX
{
    Get_dynamics_Uy__D_FX(){}
    template<typename Tuple>
    __host__ __device__
    void operator()(Tuple v) const
    {
        const FLOAT_TYPE Ux = thrust::get<1>(v);
        const FLOAT_TYPE Uy = thrust::get<2>(v);
        const FLOAT_TYPE r  = thrust::get<3>(v);
        const FLOAT_TYPE d  = thrust::get<4>(v);
        const FLOAT_TYPE Fx = thrust::get<5>(v);
        FLOAT_TYPE Fxf = (Fx >= 0)*Fx*X1::fwd_frac + (Fx < 0)*Fx*X1::fwb_frac;    // no branches!
        FLOAT_TYPE Fxr = (Fx >= 0)*Fx*X1::rwd_frac + (Fx < 0)*Fx*X1::rwb_frac;    // no branches!
        FLOAT_TYPE af  = atan2_float_type<FLOAT_TYPE>(Uy + X1::a * r, Ux) - d;
        FLOAT_TYPE ar  = atan2_float_type<FLOAT_TYPE>(Uy - X1::b * r, Ux);
        FLOAT_TYPE Fzf = (X1::m * X1::G * X1::b - X1::h * Fx) / X1::L;
        FLOAT_TYPE Fzr = (X1::m * X1::G * X1::a + X1::h * Fx) / X1::L;
        // Fyf (inlined BicycleCAvoid::fialaTireModel(af, X1::Caf, X1::mu, Fxf, Fzf))
        FLOAT_TYPE a  = af;
        FLOAT_TYPE Ca = X1::Caf;
        FLOAT_TYPE mu = X1::mu;
        FLOAT_TYPE fx = Fxf;    // avoid name conflict with input Fx; damn inlining
        FLOAT_TYPE Fz = Fzf;
        FLOAT_TYPE Fmax = mu * Fz;
        FLOAT_TYPE Fymax = sqrt_float_type<FLOAT_TYPE>((abs_float_type<FLOAT_TYPE>(fx) < Fmax)*(Fmax * Fmax - fx * fx));    // no branches!
        FLOAT_TYPE tana  = tan_float_type<FLOAT_TYPE>(a);
        FLOAT_TYPE ratio = abs_float_type<FLOAT_TYPE>(tana * Ca / (3 * (Fymax > 0) * Fymax + (Fymax <= 0))) + (Fymax <= 0); // ratio >= 1 if Fymax <= 0
        FLOAT_TYPE Fyf = (ratio < 1)*(-Ca * tana * (1 - ratio + ratio * ratio / 3)) + (ratio >= 1)*(-copysign_float_type<FLOAT_TYPE>(Fymax, tana));
        // Fyr (inlined BicycleCAvoid::fialaTireModel(ar, X1::Car, X1::mu, Fxr, Fzr))
        a  = ar;
        Ca = X1::Car;
        mu = X1::mu;
        fx = Fxr;
        Fz = Fzr;
        Fmax = mu*Fz;
        Fymax = sqrt_float_type<FLOAT_TYPE>((abs_float_type<FLOAT_TYPE>(fx) < Fmax)*(Fmax * Fmax - fx * fx));    // no branches!
        tana  = tan_float_type<FLOAT_TYPE>(a);
        ratio = abs_float_type<FLOAT_TYPE>(tana * Ca / (3 * (Fymax > 0) * Fymax + (Fymax <= 0))) + (Fymax <= 0); // ratio >= 1 if Fymax <= 0
        FLOAT_TYPE Fyr = (ratio < 1)*(-Ca * tana * (1 - ratio + ratio * ratio / 3)) + (ratio >= 1)*(-copysign_float_type<FLOAT_TYPE>(Fymax, tana));
        thrust::get<0>(v) = (Fyf + Fyr) / X1::m - r * Ux;    // dUy
    }
};

struct Get_dynamics_r__D_FX
{
    Get_dynamics_r__D_FX(){}
    template<typename Tuple>
    __host__ __device__
    void operator()(Tuple v) const
    {
        const FLOAT_TYPE Ux = thrust::get<1>(v);
        const FLOAT_TYPE Uy = thrust::get<2>(v);
        const FLOAT_TYPE r  = thrust::get<3>(v);
        const FLOAT_TYPE d  = thrust::get<4>(v);
        const FLOAT_TYPE Fx = thrust::get<5>(v);
        FLOAT_TYPE Fxf = (Fx >= 0)*Fx*X1::fwd_frac + (Fx < 0)*Fx*X1::fwb_frac;    // no branches!
        FLOAT_TYPE Fxr = (Fx >= 0)*Fx*X1::rwd_frac + (Fx < 0)*Fx*X1::rwb_frac;    // no branches!
        FLOAT_TYPE af  = atan2_float_type<FLOAT_TYPE>(Uy + X1::a * r, Ux) - d;
        FLOAT_TYPE ar  = atan2_float_type<FLOAT_TYPE>(Uy - X1::b * r, Ux);
        FLOAT_TYPE Fzf = (X1::m * X1::G * X1::b - X1::h * Fx) / X1::L;
        FLOAT_TYPE Fzr = (X1::m * X1::G * X1::a + X1::h * Fx) / X1::L;
        // Fyf (inlined BicycleCAvoid::fialaTireModel(af, X1::Caf, X1::mu, Fxf, Fzf))
        FLOAT_TYPE a  = af;
        FLOAT_TYPE Ca = X1::Caf;
        FLOAT_TYPE mu = X1::mu;
        FLOAT_TYPE fx = Fxf;    // avoid name conflict with input Fx; damn inlining
        FLOAT_TYPE Fz = Fzf;
        FLOAT_TYPE Fmax = mu * Fz;
        FLOAT_TYPE Fymax = sqrt_float_type<FLOAT_TYPE>((abs_float_type<FLOAT_TYPE>(fx) < Fmax)*(Fmax * Fmax - fx * fx));    // no branches!
        FLOAT_TYPE tana  = tan_float_type<FLOAT_TYPE>(a);
        FLOAT_TYPE ratio = abs_float_type<FLOAT_TYPE>(tana * Ca / (3 * (Fymax > 0) * Fymax + (Fymax <= 0))) + (Fymax <= 0); // ratio >= 1 if Fymax <= 0
        FLOAT_TYPE Fyf = (ratio < 1)*(-Ca * tana * (1 - ratio + ratio * ratio / 3)) + (ratio >= 1)*(-copysign_float_type<FLOAT_TYPE>(Fymax, tana));
        // Fyr (inlined BicycleCAvoid::fialaTireModel(ar, X1::Car, X1::mu, Fxr, Fzr))
        a  = ar;
        Ca = X1::Car;
        mu = X1::mu;
        fx = Fxr;
        Fz = Fzr;
        Fmax = mu*Fz;
        Fymax = sqrt_float_type<FLOAT_TYPE>((abs_float_type<FLOAT_TYPE>(fx) < Fmax)*(Fmax * Fmax - fx * fx));    // no branches!
        tana  = tan_float_type<FLOAT_TYPE>(a);
        ratio = abs_float_type<FLOAT_TYPE>(tana * Ca / (3 * (Fymax > 0) * Fymax + (Fymax <= 0))) + (Fymax <= 0); // ratio >= 1 if Fymax <= 0
        FLOAT_TYPE Fyr = (ratio < 1)*(-Ca * tana * (1 - ratio + ratio * ratio / 3)) + (ratio >= 1)*(-copysign_float_type<FLOAT_TYPE>(Fymax, tana));
        thrust::get<0>(v) = (X1::a * Fyf - X1::b * Fyr) / X1::Izz;    // dr
    }
};

bool dynamics_cell_helper_execute_cuda_dimAll(
    std::vector<beacls::UVec>& dx_uvecs,
    const std::vector<beacls::UVec>& x_uvecs,
    const std::vector<beacls::UVec>& u_uvecs,
    const std::vector<beacls::UVec>& d_uvecs
) {
    bool result = true;
    beacls::UVec& dx_rel_uvec        = dx_uvecs[0];
    beacls::UVec& dy_rel_uvec        = dx_uvecs[1];
    beacls::UVec& dpsi_rel_uvec      = dx_uvecs[2];
    beacls::UVec& dUx_uvec           = dx_uvecs[3];
    beacls::UVec& dUy_uvec           = dx_uvecs[4];
    beacls::UVec& dV_uvec            = dx_uvecs[5];
    beacls::UVec& dr_uvec            = dx_uvecs[6];
    const beacls::UVec& x_rel_uvec   = x_uvecs[0];
    const beacls::UVec& y_rel_uvec   = x_uvecs[1];
    const beacls::UVec& psi_rel_uvec = x_uvecs[2];
    const beacls::UVec& Ux_uvec      = x_uvecs[3];
    const beacls::UVec& Uy_uvec      = x_uvecs[4];
    const beacls::UVec& V_uvec       = x_uvecs[5];
    const beacls::UVec& r_uvec       = x_uvecs[6];
    const beacls::UVec& d_uvec       = u_uvecs[0];
    const beacls::UVec& Fx_uvec      = u_uvecs[1];
    const beacls::UVec& w_uvec       = d_uvecs[0];
    const beacls::UVec& a_uvec       = d_uvecs[1];

    beacls::reallocateAsSrc(dx_rel_uvec  , x_rel_uvec);    // sizing approach taken in P5D_Dubins.cpp, as opposed to below
    beacls::reallocateAsSrc(dy_rel_uvec  , y_rel_uvec);
    beacls::reallocateAsSrc(dpsi_rel_uvec, psi_rel_uvec);
    beacls::reallocateAsSrc(dUx_uvec     , Ux_uvec);
    beacls::reallocateAsSrc(dUy_uvec     , Uy_uvec);
    beacls::reallocateAsSrc(dV_uvec      , V_uvec);
    beacls::reallocateAsSrc(dr_uvec      , r_uvec);

    // beacls::reallocateAsSrc(dx_rel_uvec  , r_uvec);    // r appears in the dynamics of every state except dV/dt
    // beacls::reallocateAsSrc(dy_rel_uvec  , r_uvec);
    // beacls::reallocateAsSrc(dpsi_rel_uvec, r_uvec);
    // beacls::reallocateAsSrc(dUx_uvec     , r_uvec);
    // beacls::reallocateAsSrc(dUy_uvec     , r_uvec);
    // beacls::reallocateAsSrc(dV_uvec      , a_uvec);    // might end up being a scalar?
    // beacls::reallocateAsSrc(dr_uvec      , r_uvec);

    FLOAT_TYPE* dx_rel_ptr        = beacls::UVec_<FLOAT_TYPE>(dx_rel_uvec).ptr();
    FLOAT_TYPE* dy_rel_ptr        = beacls::UVec_<FLOAT_TYPE>(dy_rel_uvec).ptr();
    FLOAT_TYPE* dpsi_rel_ptr      = beacls::UVec_<FLOAT_TYPE>(dpsi_rel_uvec).ptr();
    FLOAT_TYPE* dUx_ptr           = beacls::UVec_<FLOAT_TYPE>(dUx_uvec).ptr();
    FLOAT_TYPE* dUy_ptr           = beacls::UVec_<FLOAT_TYPE>(dUy_uvec).ptr();
    FLOAT_TYPE* dV_ptr            = beacls::UVec_<FLOAT_TYPE>(dV_uvec).ptr();
    FLOAT_TYPE* dr_ptr            = beacls::UVec_<FLOAT_TYPE>(dr_uvec).ptr();
    const FLOAT_TYPE* x_rel_ptr   = beacls::UVec_<FLOAT_TYPE>(x_rel_uvec).ptr();
    const FLOAT_TYPE* y_rel_ptr   = beacls::UVec_<FLOAT_TYPE>(y_rel_uvec).ptr();
    const FLOAT_TYPE* psi_rel_ptr = beacls::UVec_<FLOAT_TYPE>(psi_rel_uvec).ptr();
    const FLOAT_TYPE* Ux_ptr      = beacls::UVec_<FLOAT_TYPE>(Ux_uvec).ptr();
    const FLOAT_TYPE* Uy_ptr      = beacls::UVec_<FLOAT_TYPE>(Uy_uvec).ptr();
    const FLOAT_TYPE* V_ptr       = beacls::UVec_<FLOAT_TYPE>(V_uvec).ptr();
    const FLOAT_TYPE* r_ptr       = beacls::UVec_<FLOAT_TYPE>(r_uvec).ptr();
    const FLOAT_TYPE* d_ptr       = beacls::UVec_<FLOAT_TYPE>(d_uvec).ptr();
    const FLOAT_TYPE* Fx_ptr      = beacls::UVec_<FLOAT_TYPE>(Fx_uvec).ptr();
    const FLOAT_TYPE* w_ptr       = beacls::UVec_<FLOAT_TYPE>(w_uvec).ptr();
    const FLOAT_TYPE* a_ptr       = beacls::UVec_<FLOAT_TYPE>(a_uvec).ptr();

    thrust::device_ptr<FLOAT_TYPE> dx_rel_dev_ptr   = thrust::device_pointer_cast(dx_rel_ptr);
    thrust::device_ptr<FLOAT_TYPE> dy_rel_dev_ptr   = thrust::device_pointer_cast(dy_rel_ptr);
    thrust::device_ptr<FLOAT_TYPE> dpsi_rel_dev_ptr = thrust::device_pointer_cast(dpsi_rel_ptr);
    thrust::device_ptr<FLOAT_TYPE> dUx_dev_ptr      = thrust::device_pointer_cast(dUx_ptr);
    thrust::device_ptr<FLOAT_TYPE> dUy_dev_ptr      = thrust::device_pointer_cast(dUy_ptr);
    thrust::device_ptr<FLOAT_TYPE> dV_dev_ptr       = thrust::device_pointer_cast(dV_ptr);
    thrust::device_ptr<FLOAT_TYPE> dr_dev_ptr       = thrust::device_pointer_cast(dr_ptr);

    cudaStream_t dx_stream = beacls::get_stream(dx_rel_uvec);
    dx_rel_uvec  .set_cudaStream(dx_rel_uvec.get_cudaStream());
    dy_rel_uvec  .set_cudaStream(dx_rel_uvec.get_cudaStream());
    dpsi_rel_uvec.set_cudaStream(dx_rel_uvec.get_cudaStream());
    dUx_uvec     .set_cudaStream(dx_rel_uvec.get_cudaStream());
    dUy_uvec     .set_cudaStream(dx_rel_uvec.get_cudaStream());
    dV_uvec      .set_cudaStream(dx_rel_uvec.get_cudaStream());
    dr_uvec      .set_cudaStream(dx_rel_uvec.get_cudaStream());

    thrust::device_ptr<const FLOAT_TYPE> x_rel_dev_ptr   = thrust::device_pointer_cast(x_rel_ptr);
    thrust::device_ptr<const FLOAT_TYPE> y_rel_dev_ptr   = thrust::device_pointer_cast(y_rel_ptr);
    thrust::device_ptr<const FLOAT_TYPE> psi_rel_dev_ptr = thrust::device_pointer_cast(psi_rel_ptr);
    thrust::device_ptr<const FLOAT_TYPE> Ux_dev_ptr      = thrust::device_pointer_cast(Ux_ptr);
    thrust::device_ptr<const FLOAT_TYPE> Uy_dev_ptr      = thrust::device_pointer_cast(Uy_ptr);
    thrust::device_ptr<const FLOAT_TYPE> V_dev_ptr       = thrust::device_pointer_cast(V_ptr);
    thrust::device_ptr<const FLOAT_TYPE> r_dev_ptr       = thrust::device_pointer_cast(r_ptr);

    //!< limit of template variables of thrust::tuple is 10, therefore divide the thrust calls; also divided by control/disturbance inputs
    auto dst_src_Tuple_0 = thrust::make_tuple(dx_rel_dev_ptr,
                                              dy_rel_dev_ptr,
                                              psi_rel_dev_ptr,
                                              x_rel_dev_ptr,
                                              y_rel_dev_ptr,
                                              Ux_dev_ptr,
                                              Uy_dev_ptr,
                                              V_dev_ptr,
                                              r_dev_ptr);
    auto dst_src_Iterator_0 = thrust::make_zip_iterator(dst_src_Tuple_0);
    thrust::for_each(thrust::cuda::par.on(dx_stream), dst_src_Iterator_0, dst_src_Iterator_0 + x_rel_uvec.size(), Get_dynamics_x_rel_y_rel());

    // controls/disturbances that appear in more than one dynamics component are guaranteed to be cuda `UVec`s?
    thrust::device_ptr<const FLOAT_TYPE> d_dev_ptr  = thrust::device_pointer_cast(d_ptr);
    thrust::device_ptr<const FLOAT_TYPE> Fx_dev_ptr = thrust::device_pointer_cast(Fx_ptr);
    beacls::synchronizeUVec(d_uvec);
    beacls::synchronizeUVec(Fx_uvec);

    if (beacls::is_cuda(d_uvec) && beacls::is_cuda(Fx_uvec)) {
        auto dst_src_Tuple_1 = thrust::make_tuple(dUx_dev_ptr,
                                                  dUy_dev_ptr,
                                                  dr_dev_ptr,
                                                  Ux_dev_ptr,
                                                  Uy_dev_ptr,
                                                  r_dev_ptr,
                                                  d_dev_ptr,
                                                  Fx_dev_ptr);
        auto dst_src_Iterator_1 = thrust::make_zip_iterator(dst_src_Tuple_1);
        thrust::for_each(thrust::cuda::par.on(dx_stream), dst_src_Iterator_1, dst_src_Iterator_1 + dUx_uvec.size(), Get_dynamics_Ux_Uy_r__D_FX());
    } else {
        std::cerr << __FILE__ << ":" << __LINE__ << ":" << __FUNCTION__ << " Invalid data size" << std::endl;
        return false;
    }

    if (beacls::is_cuda(w_uvec) && beacls::is_cuda(a_uvec)) {
        thrust::device_ptr<const FLOAT_TYPE> w_dev_ptr = thrust::device_pointer_cast(w_ptr);
        thrust::device_ptr<const FLOAT_TYPE> a_dev_ptr = thrust::device_pointer_cast(a_ptr);
        beacls::synchronizeUVec(w_uvec);    // not in PlaneCAvoid_cuda.cu
        beacls::synchronizeUVec(a_uvec);    // not in PlaneCAvoid_cuda.cu
        auto dst_src_Tuple_2 = thrust::make_tuple(dpsi_rel_dev_ptr,
                                                  dV_dev_ptr,
                                                  r_dev_ptr,
                                                  w_dev_ptr,
                                                  a_dev_ptr);
        auto dst_src_Iterator_2 = thrust::make_zip_iterator(dst_src_Tuple_2);
        thrust::for_each(thrust::cuda::par.on(dx_stream), dst_src_Iterator_2, dst_src_Iterator_2 + psi_rel_uvec.size(), Get_dynamics_psi_rel_V__W_A());
    } else {
        const FLOAT_TYPE w = w_ptr[0];
        const FLOAT_TYPE a = a_ptr[0];
        auto dst_src_Tuple_2 = thrust::make_tuple(dpsi_rel_dev_ptr,
                                                  dV_dev_ptr,
                                                  r_dev_ptr);
        auto dst_src_Iterator_2 = thrust::make_zip_iterator(dst_src_Tuple_2);
        thrust::for_each(thrust::cuda::par.on(dx_stream), dst_src_Iterator_2, dst_src_Iterator_2 + psi_rel_uvec.size(), Get_dynamics_psi_rel_V__w_a(w, a));
    }
    return result;
}

bool dynamics_cell_helper_execute_cuda(
    beacls::UVec& dx_uvec,
    const std::vector<beacls::UVec>& x_uvecs,
    const std::vector<beacls::UVec>& u_uvecs,
    const std::vector<beacls::UVec>& d_uvecs,
    const size_t dim
) {
    bool result = true;

    beacls::reallocateAsSrc(dx_uvec, x_uvecs[dim]);
    thrust::device_ptr<FLOAT_TYPE> dx_dim_dev_ptr = thrust::device_pointer_cast(beacls::UVec_<FLOAT_TYPE>(dx_uvec).ptr());
    cudaStream_t dx_stream = beacls::get_stream(dx_uvec);
    // not sure if doing all casts up front like this is over-transferring memory
    thrust::device_ptr<const FLOAT_TYPE> x_rel_dev_ptr   = thrust::device_pointer_cast(beacls::UVec_<FLOAT_TYPE>(x_uvecs[0]).ptr());
    thrust::device_ptr<const FLOAT_TYPE> y_rel_dev_ptr   = thrust::device_pointer_cast(beacls::UVec_<FLOAT_TYPE>(x_uvecs[1]).ptr());
    thrust::device_ptr<const FLOAT_TYPE> psi_rel_dev_ptr = thrust::device_pointer_cast(beacls::UVec_<FLOAT_TYPE>(x_uvecs[2]).ptr());
    thrust::device_ptr<const FLOAT_TYPE> Ux_dev_ptr      = thrust::device_pointer_cast(beacls::UVec_<FLOAT_TYPE>(x_uvecs[3]).ptr());
    thrust::device_ptr<const FLOAT_TYPE> Uy_dev_ptr      = thrust::device_pointer_cast(beacls::UVec_<FLOAT_TYPE>(x_uvecs[4]).ptr());
    thrust::device_ptr<const FLOAT_TYPE> V_dev_ptr       = thrust::device_pointer_cast(beacls::UVec_<FLOAT_TYPE>(x_uvecs[5]).ptr());
    thrust::device_ptr<const FLOAT_TYPE> r_dev_ptr       = thrust::device_pointer_cast(beacls::UVec_<FLOAT_TYPE>(x_uvecs[6]).ptr());
    const beacls::UVec& d_uvec  = u_uvecs[0];
    const beacls::UVec& Fx_uvec = u_uvecs[1];
    const beacls::UVec& w_uvec  = d_uvecs[0];
    const beacls::UVec& a_uvec  = d_uvecs[1];
    switch (dim) {
    case 0:    // dx_rel
        {
            auto Tuple0 = thrust::make_tuple(dx_dim_dev_ptr,
                                            psi_rel_dev_ptr,
                                            y_rel_dev_ptr,
                                            Ux_dev_ptr,
                                            V_dev_ptr,
                                            r_dev_ptr);
            auto Iterator0 = thrust::make_zip_iterator(Tuple0);
            thrust::for_each(thrust::cuda::par.on(dx_stream), Iterator0, Iterator0 + dx_uvec.size(), Get_dynamics_x_rel());
        }
        break;
    case 1:    // dy_rel
        {
            auto Tuple1 = thrust::make_tuple(dx_dim_dev_ptr,
                                            psi_rel_dev_ptr,
                                            x_rel_dev_ptr,
                                            Uy_dev_ptr,
                                            V_dev_ptr,
                                            r_dev_ptr);
            auto Iterator1 = thrust::make_zip_iterator(Tuple1);
            thrust::for_each(thrust::cuda::par.on(dx_stream), Iterator1, Iterator1 + dx_uvec.size(), Get_dynamics_y_rel());
        }
        break;
    case 2:    // dpsi_rel
        if (beacls::is_cuda(w_uvec)) {
            thrust::device_ptr<const FLOAT_TYPE> w_dev_ptr = thrust::device_pointer_cast(beacls::UVec_<FLOAT_TYPE>(w_uvec).ptr());
            beacls::synchronizeUVec(w_uvec);
            auto Tuple2 = thrust::make_tuple(dx_dim_dev_ptr,
                                            r_dev_ptr,
                                            w_dev_ptr);
            auto Iterator2 = thrust::make_zip_iterator(Tuple2);
            thrust::for_each(thrust::cuda::par.on(dx_stream), Iterator2, Iterator2 + dx_uvec.size(), Get_dynamics_psi_rel__W());
        } else {
            const FLOAT_TYPE w = beacls::UVec_<FLOAT_TYPE>(w_uvec).ptr()[0];
            auto Tuple2 = thrust::make_tuple(dx_dim_dev_ptr,
                                            r_dev_ptr);
            auto Iterator2 = thrust::make_zip_iterator(Tuple2);
            thrust::for_each(thrust::cuda::par.on(dx_stream), Iterator2, Iterator2 + dx_uvec.size(), Get_dynamics_psi_rel__w(w));
        }
        break;
    case 3:    // dUx
        if (beacls::is_cuda(Fx_uvec)) {
            thrust::device_ptr<const FLOAT_TYPE> Fx_dev_ptr = thrust::device_pointer_cast(beacls::UVec_<FLOAT_TYPE>(Fx_uvec).ptr());
            beacls::synchronizeUVec(Fx_uvec);
            auto Tuple3 = thrust::make_tuple(dx_dim_dev_ptr,
                                            Ux_dev_ptr,
                                            Uy_dev_ptr,
                                            r_dev_ptr,
                                            Fx_dev_ptr);
            auto Iterator3 = thrust::make_zip_iterator(Tuple3);
            thrust::for_each(thrust::cuda::par.on(dx_stream), Iterator3, Iterator3 + dx_uvec.size(), Get_dynamics_Ux__FX());
        } else {
            std::cerr << __FILE__ << ":" << __LINE__ << ":" << __FUNCTION__ << " Invalid data size" << std::endl;
            return false;
        }
        break;
    case 4:    // dUy
        if (beacls::is_cuda(Fx_uvec) && beacls::is_cuda(d_uvec)) {
            thrust::device_ptr<const FLOAT_TYPE> Fx_dev_ptr = thrust::device_pointer_cast(beacls::UVec_<FLOAT_TYPE>(Fx_uvec).ptr());
            thrust::device_ptr<const FLOAT_TYPE> d_dev_ptr  = thrust::device_pointer_cast(beacls::UVec_<FLOAT_TYPE>(d_uvec).ptr());
            beacls::synchronizeUVec(Fx_uvec);
            beacls::synchronizeUVec(d_uvec);
            auto Tuple4 = thrust::make_tuple(dx_dim_dev_ptr,
                                            Ux_dev_ptr,
                                            Uy_dev_ptr,
                                            r_dev_ptr,
                                            d_dev_ptr,
                                            Fx_dev_ptr);
            auto Iterator4 = thrust::make_zip_iterator(Tuple4);
            thrust::for_each(thrust::cuda::par.on(dx_stream), Iterator4, Iterator4 + dx_uvec.size(), Get_dynamics_Uy__D_FX());
        } else {
            std::cerr << __FILE__ << ":" << __LINE__ << ":" << __FUNCTION__ << " Invalid data size" << std::endl;
            return false;
        }
        break;
    case 5:    // dV
        if (beacls::is_cuda(a_uvec)) {
            thrust::device_ptr<const FLOAT_TYPE> a_dev_ptr = thrust::device_pointer_cast(beacls::UVec_<FLOAT_TYPE>(a_uvec).ptr());
            beacls::synchronizeUVec(a_uvec);
            auto Tuple5 = thrust::make_tuple(dx_dim_dev_ptr,
                                            a_dev_ptr);
            auto Iterator5 = thrust::make_zip_iterator(Tuple5);
            thrust::for_each(thrust::cuda::par.on(dx_stream), Iterator5, Iterator5 + dx_uvec.size(), Get_dynamics_V__A());
        } else {
            const FLOAT_TYPE a = beacls::UVec_<FLOAT_TYPE>(a_uvec).ptr()[0];
            auto Tuple5 = thrust::make_tuple(dx_dim_dev_ptr);
            auto Iterator5 = thrust::make_zip_iterator(Tuple5);
            thrust::for_each(thrust::cuda::par.on(dx_stream), Iterator5, Iterator5 + dx_uvec.size(), Get_dynamics_V__a(a));
        }
        break;
    case 6:    // dr
        if (beacls::is_cuda(Fx_uvec) && beacls::is_cuda(d_uvec)) {
            thrust::device_ptr<const FLOAT_TYPE> Fx_dev_ptr = thrust::device_pointer_cast(beacls::UVec_<FLOAT_TYPE>(Fx_uvec).ptr());
            thrust::device_ptr<const FLOAT_TYPE> d_dev_ptr  = thrust::device_pointer_cast(beacls::UVec_<FLOAT_TYPE>(d_uvec).ptr());
            beacls::synchronizeUVec(Fx_uvec);
            beacls::synchronizeUVec(d_uvec);
            auto Tuple6 = thrust::make_tuple(dx_dim_dev_ptr,
                                            Ux_dev_ptr,
                                            Uy_dev_ptr,
                                            r_dev_ptr,
                                            d_dev_ptr,
                                            Fx_dev_ptr);
            auto Iterator6 = thrust::make_zip_iterator(Tuple6);
            thrust::for_each(thrust::cuda::par.on(dx_stream), Iterator6, Iterator6 + dx_uvec.size(), Get_dynamics_r__D_FX());
        } else {
            std::cerr << __FILE__ << ":" << __LINE__ << ":" << __FUNCTION__ << " Invalid data size" << std::endl;
            return false;
        }
        break;
    default:
        std::cerr << "Only dimension 1-7 are defined for dynamics of BicycleCAvoid!" << std::endl;
        result = false;
        break;
    }
    return result;
}

};     /* namespace BicycleCAvoid_CUDA */
#endif /* defined(USER_DEFINED_GPU_DYNSYS_FUNC) */
#endif /* defined(WITH_GPU) */
