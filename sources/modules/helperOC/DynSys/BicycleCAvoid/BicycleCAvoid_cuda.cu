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

struct Get_optCtrl_dim0dim1_D {
public:
    const FLOAT_TYPE vrangeA0;
    const FLOAT_TYPE vrangeA1;
    const FLOAT_TYPE wMax;
    Get_optCtrl_dim0dim1_D(
        const FLOAT_TYPE vrangeA0,
        const FLOAT_TYPE vrangeA1,
        const FLOAT_TYPE wMax) :
        vrangeA0(vrangeA0),
        vrangeA1(vrangeA1),
        wMax(wMax) {}
    template<typename Tuple>
    __host__ __device__
    void operator()(Tuple v) const
    {
        const FLOAT_TYPE y0 = thrust::get<2>(v);
        const FLOAT_TYPE y1 = thrust::get<3>(v);
        const FLOAT_TYPE deriv0 = thrust::get<4>(v);
        const FLOAT_TYPE deriv1 = thrust::get<5>(v);
        const FLOAT_TYPE deriv2 = thrust::get<6>(v);
        const FLOAT_TYPE det0 = -deriv0;
        const FLOAT_TYPE det1 = deriv0 * y1 - deriv1 * y0 - deriv2;
        thrust::get<0>(v) = (det0 >= 0) ? vrangeA1 : vrangeA0;
        thrust::get<1>(v) = (det1 >= 0) ? wMax : -wMax;
    }
};

struct Get_optCtrl_dim1_d {
public:
    const FLOAT_TYPE wMax;
    const FLOAT_TYPE d0;
    const FLOAT_TYPE d1;
    const FLOAT_TYPE d2;
    Get_optCtrl_dim1_d(const FLOAT_TYPE wMax, const FLOAT_TYPE d0, const FLOAT_TYPE d1, const FLOAT_TYPE d2) :
        wMax(wMax), d0(d0), d1(d1), d2(d2) {}
    __host__ __device__
    FLOAT_TYPE operator()(const FLOAT_TYPE y0, const FLOAT_TYPE y1) const
    {
        const FLOAT_TYPE det1 = d0 * y1 - d1 * y0 - d2;
        return (det1 >= 0) ? wMax : -wMax;
    }
};

bool optCtrl_execute_cuda(
    std::vector<beacls::UVec>& u_uvecs,
    const std::vector<beacls::UVec>& x_uvecs,
    const std::vector<beacls::UVec>& deriv_uvecs,
    const FLOAT_TYPE wMaxA,
    const std::vector<FLOAT_TYPE>& vRangeA,
    const helperOC::DynSys_UMode_Type uMode
)
{
    bool result = true;
    beacls::reallocateAsSrc(u_uvecs[0], deriv_uvecs[0]);
    beacls::reallocateAsSrc(u_uvecs[1], x_uvecs[0]);
    FLOAT_TYPE* uOpt0_ptr = beacls::UVec_<FLOAT_TYPE>(u_uvecs[0]).ptr();
    FLOAT_TYPE* uOpt1_ptr = beacls::UVec_<FLOAT_TYPE>(u_uvecs[1]).ptr();
    const FLOAT_TYPE* y0_ptr = beacls::UVec_<FLOAT_TYPE>(x_uvecs[0]).ptr();
    const FLOAT_TYPE* y1_ptr = beacls::UVec_<FLOAT_TYPE>(x_uvecs[1]).ptr();
    const FLOAT_TYPE* deriv0_ptr = beacls::UVec_<FLOAT_TYPE>(deriv_uvecs[0]).ptr();
    const FLOAT_TYPE* deriv1_ptr = beacls::UVec_<FLOAT_TYPE>(deriv_uvecs[1]).ptr();
    const FLOAT_TYPE* deriv2_ptr = beacls::UVec_<FLOAT_TYPE>(deriv_uvecs[2]).ptr();
    thrust::device_ptr<const FLOAT_TYPE> y0_dev_ptr = thrust::device_pointer_cast(y0_ptr);
    thrust::device_ptr<const FLOAT_TYPE> y1_dev_ptr = thrust::device_pointer_cast(y1_ptr);
    const FLOAT_TYPE vRangeA0 = vRangeA[0];
    const FLOAT_TYPE vRangeA1 = vRangeA[1];
    if ((uMode == helperOC::DynSys_UMode_Max) || (uMode == helperOC::DynSys_UMode_Min)) {
        const FLOAT_TYPE moded_wMaxA = (uMode == helperOC::DynSys_UMode_Max) ? wMaxA : -wMaxA;
        const FLOAT_TYPE moded_vRangeA0 = (uMode == helperOC::DynSys_UMode_Max) ? vRangeA0 : vRangeA1;
        const FLOAT_TYPE moded_vRangeA1 = (uMode == helperOC::DynSys_UMode_Max) ? vRangeA1 : vRangeA0;
        cudaStream_t u_stream = beacls::get_stream(u_uvecs[1]);
        thrust::device_ptr<FLOAT_TYPE> uOpt1_dev_ptr = thrust::device_pointer_cast(uOpt1_ptr);
        if(is_cuda(deriv_uvecs[0]) && is_cuda(deriv_uvecs[1]) && is_cuda(deriv_uvecs[2])){
            thrust::device_ptr<FLOAT_TYPE> uOpt0_dev_ptr = thrust::device_pointer_cast(uOpt0_ptr);
            u_uvecs[0].set_cudaStream(u_uvecs[1].get_cudaStream());
            thrust::device_ptr<const FLOAT_TYPE> deriv0_dev_ptr = thrust::device_pointer_cast(deriv0_ptr);
            thrust::device_ptr<const FLOAT_TYPE> deriv1_dev_ptr = thrust::device_pointer_cast(deriv1_ptr);
            thrust::device_ptr<const FLOAT_TYPE> deriv2_dev_ptr = thrust::device_pointer_cast(deriv2_ptr);
            auto src_dst_Tuple = thrust::make_tuple(uOpt0_dev_ptr, uOpt1_dev_ptr,
                y0_dev_ptr, y1_dev_ptr, deriv0_dev_ptr, deriv1_dev_ptr, deriv2_dev_ptr);
            auto src_dst_Iterator = thrust::make_zip_iterator(src_dst_Tuple);
            thrust::for_each(thrust::cuda::par.on(u_stream),
                src_dst_Iterator, src_dst_Iterator + x_uvecs[0].size(),
                Get_optCtrl_dim0dim1_D(moded_vRangeA0, moded_vRangeA1, moded_wMaxA));
        }
        else {
            const FLOAT_TYPE d0 = deriv0_ptr[0];
            const FLOAT_TYPE d1 = deriv1_ptr[0];
            const FLOAT_TYPE d2 = deriv2_ptr[0];
            const FLOAT_TYPE det0 = -d0;
            uOpt0_ptr[0] = (det0 >= 0) ? moded_vRangeA1 : moded_vRangeA0;
            thrust::transform(thrust::cuda::par.on(u_stream),
                y0_dev_ptr, y0_dev_ptr + x_uvecs[0].size(), y1_dev_ptr, uOpt1_dev_ptr,
                Get_optCtrl_dim1_d(moded_wMaxA, d0, d1, d2));
        }
    }
    else {
        std::cerr << "Unknown uMode!: " << uMode << std::endl;
        result = false;
    }
    return result;
}

struct Get_optDstb_dim0_d {
public:
    const FLOAT_TYPE vrangeB0;
    const FLOAT_TYPE vrangeB1;
    const FLOAT_TYPE d0;
    const FLOAT_TYPE d1;
    Get_optDstb_dim0_d(const FLOAT_TYPE vrangeB0, const FLOAT_TYPE vrangeB1, const FLOAT_TYPE d0, const FLOAT_TYPE d1) :
        vrangeB0(vrangeB0), vrangeB1(vrangeB1), d0(d0), d1(d1) {}
    __host__ __device__
    FLOAT_TYPE operator()(const FLOAT_TYPE y2) const
    {
        FLOAT_TYPE sin_y2;
        FLOAT_TYPE cos_y2;
        sincos_float_type<FLOAT_TYPE>(y2, sin_y2, cos_y2);

        const FLOAT_TYPE det0 = d0 * cos_y2 + d1 * sin_y2;
        return (det0 >= 0) ? vrangeB1 : vrangeB0;
    }
};

struct Get_optDstb_dim0dim1dim2dim3dim4_D {
public:
    const FLOAT_TYPE vrangeB0;
    const FLOAT_TYPE vrangeB1;
    const FLOAT_TYPE wMaxB;
    const FLOAT_TYPE dMaxA_0_dMaxB_0;
    const FLOAT_TYPE dMaxA_1_dMaxB_1;
    Get_optDstb_dim0dim1dim2dim3dim4_D(
        const FLOAT_TYPE vrangeB0,
        const FLOAT_TYPE vrangeB1,
        const FLOAT_TYPE wMaxB,
        const FLOAT_TYPE dMaxA_0_dMaxB_0,
        const FLOAT_TYPE dMaxA_1_dMaxB_1) :
        vrangeB0(vrangeB0),
        vrangeB1(vrangeB1),
        wMaxB(wMaxB),
        dMaxA_0_dMaxB_0(dMaxA_0_dMaxB_0),
        dMaxA_1_dMaxB_1(dMaxA_1_dMaxB_1) {}
    template<typename Tuple>
    __host__ __device__
    void operator()(Tuple v) const
    {
        const FLOAT_TYPE y2 = thrust::get<5>(v);
        const FLOAT_TYPE deriv0 = thrust::get<6>(v);
        const FLOAT_TYPE deriv1 = thrust::get<7>(v);
        const FLOAT_TYPE deriv2 = thrust::get<8>(v);
        FLOAT_TYPE sin_y2;
        FLOAT_TYPE cos_y2;
        sincos_float_type<FLOAT_TYPE>(y2, sin_y2, cos_y2);
        const FLOAT_TYPE normDeriv = sqrt_float_type<FLOAT_TYPE>(deriv0 * deriv0 + deriv1 * deriv1);
        const FLOAT_TYPE det0 = deriv0 * cos_y2 + deriv1 * sin_y2;
        thrust::get<0>(v) = (det0 >= 0) ? vrangeB1 : vrangeB0;
        if (normDeriv == 0) {
            thrust::get<2>(v) = 0;
            thrust::get<3>(v) = 0;
        } else {
            thrust::get<2>(v) = dMaxA_0_dMaxB_0 * deriv0 / normDeriv;
            thrust::get<3>(v) = dMaxA_0_dMaxB_0 * deriv1 / normDeriv;
        }
        if (deriv2 >= 0) {
            thrust::get<1>(v) = wMaxB;
            thrust::get<4>(v) = dMaxA_1_dMaxB_1;
        } else {
            thrust::get<1>(v) = -wMaxB;
            thrust::get<4>(v) = -dMaxA_1_dMaxB_1;
        }
    }
};
bool optDstb_execute_cuda(
    std::vector<beacls::UVec>& d_uvecs,
    const std::vector<beacls::UVec>& x_uvecs,
    const std::vector<beacls::UVec>& deriv_uvecs,
    const std::vector<FLOAT_TYPE>& dMaxA,
    const std::vector<FLOAT_TYPE>& dMaxB,
    const std::vector<FLOAT_TYPE>& vRangeB,
    const FLOAT_TYPE wMaxB,
    const helperOC::DynSys_DMode_Type dMode
)
{
    bool result = true;
    beacls::reallocateAsSrc(d_uvecs[0], x_uvecs[2]);
    beacls::reallocateAsSrc(d_uvecs[1], deriv_uvecs[0]);
    beacls::reallocateAsSrc(d_uvecs[2], deriv_uvecs[0]);
    beacls::reallocateAsSrc(d_uvecs[3], deriv_uvecs[0]);
    beacls::reallocateAsSrc(d_uvecs[4], deriv_uvecs[0]);
    FLOAT_TYPE* dOpt0_ptr = beacls::UVec_<FLOAT_TYPE>(d_uvecs[0]).ptr();
    FLOAT_TYPE* dOpt1_ptr = beacls::UVec_<FLOAT_TYPE>(d_uvecs[1]).ptr();
    FLOAT_TYPE* dOpt2_ptr = beacls::UVec_<FLOAT_TYPE>(d_uvecs[2]).ptr();
    FLOAT_TYPE* dOpt3_ptr = beacls::UVec_<FLOAT_TYPE>(d_uvecs[3]).ptr();
    FLOAT_TYPE* dOpt4_ptr = beacls::UVec_<FLOAT_TYPE>(d_uvecs[4]).ptr();
    const FLOAT_TYPE* y2_ptr = beacls::UVec_<FLOAT_TYPE>(x_uvecs[2]).ptr();
    const FLOAT_TYPE* deriv0_ptr = beacls::UVec_<FLOAT_TYPE>(deriv_uvecs[0]).ptr();
    const FLOAT_TYPE* deriv1_ptr = beacls::UVec_<FLOAT_TYPE>(deriv_uvecs[1]).ptr();
    const FLOAT_TYPE* deriv2_ptr = beacls::UVec_<FLOAT_TYPE>(deriv_uvecs[2]).ptr();
    const FLOAT_TYPE dMaxA_0 = dMaxA[0];
    const FLOAT_TYPE dMaxA_1 = dMaxA[1];
    const FLOAT_TYPE dMaxB_0 = dMaxB[0];
    const FLOAT_TYPE dMaxB_1 = dMaxB[1];
    const FLOAT_TYPE vRangeB0 = vRangeB[0];
    const FLOAT_TYPE vRangeB1 = vRangeB[1];
    const FLOAT_TYPE dMaxA_0_dMaxB_0 = dMaxA_0 + dMaxB_0;
    const FLOAT_TYPE dMaxA_1_dMaxB_1 = dMaxA_1 + dMaxB_1;
    if ((dMode == helperOC::DynSys_DMode_Max) || (dMode == helperOC::DynSys_DMode_Min)) {
        const FLOAT_TYPE moded_wMaxB = (dMode == helperOC::DynSys_DMode_Max) ? wMaxB : -wMaxB;
        const FLOAT_TYPE moded_vRangeB0 = (dMode == helperOC::DynSys_DMode_Max) ? vRangeB0 : vRangeB1;
        const FLOAT_TYPE moded_vRangeB1 = (dMode == helperOC::DynSys_DMode_Max) ? vRangeB1 : vRangeB0;
        const FLOAT_TYPE moded_dMaxA_0_dMaxB_0 = (dMode == helperOC::DynSys_DMode_Max) ? dMaxA_0_dMaxB_0 : -dMaxA_0_dMaxB_0;
        const FLOAT_TYPE moded_dMaxA_1_dMaxB_1 = (dMode == helperOC::DynSys_DMode_Max) ? dMaxA_1_dMaxB_1 : -dMaxA_1_dMaxB_1;
        thrust::device_ptr<FLOAT_TYPE> dOpt0_dev_ptr = thrust::device_pointer_cast(dOpt0_ptr);
        cudaStream_t d_stream = beacls::get_stream(d_uvecs[0]);
        thrust::device_ptr<const FLOAT_TYPE> y2_dev_ptr = thrust::device_pointer_cast(y2_ptr);
        if (beacls::is_cuda(deriv_uvecs[0]) && beacls::is_cuda(deriv_uvecs[1]) && beacls::is_cuda(deriv_uvecs[2])){
            thrust::device_ptr<FLOAT_TYPE> dOpt1_dev_ptr = thrust::device_pointer_cast(dOpt1_ptr);
            thrust::device_ptr<FLOAT_TYPE> dOpt2_dev_ptr = thrust::device_pointer_cast(dOpt2_ptr);
            thrust::device_ptr<FLOAT_TYPE> dOpt3_dev_ptr = thrust::device_pointer_cast(dOpt3_ptr);
            thrust::device_ptr<FLOAT_TYPE> dOpt4_dev_ptr = thrust::device_pointer_cast(dOpt4_ptr);
            thrust::device_ptr<const FLOAT_TYPE> deriv0_dev_ptr = thrust::device_pointer_cast(deriv0_ptr);
            thrust::device_ptr<const FLOAT_TYPE> deriv1_dev_ptr = thrust::device_pointer_cast(deriv1_ptr);
            thrust::device_ptr<const FLOAT_TYPE> deriv2_dev_ptr = thrust::device_pointer_cast(deriv2_ptr);
            d_uvecs[1].set_cudaStream(d_uvecs[0].get_cudaStream());
            d_uvecs[2].set_cudaStream(d_uvecs[0].get_cudaStream());
            d_uvecs[3].set_cudaStream(d_uvecs[0].get_cudaStream());
            d_uvecs[4].set_cudaStream(d_uvecs[0].get_cudaStream());
            auto dst_src_Tuple = thrust::make_tuple(
                dOpt0_dev_ptr, dOpt1_dev_ptr, dOpt2_dev_ptr, dOpt3_dev_ptr, dOpt4_dev_ptr,
                y2_dev_ptr, deriv0_dev_ptr, deriv1_dev_ptr, deriv2_dev_ptr);
            auto dst_src_Iterator = thrust::make_zip_iterator(dst_src_Tuple);
            thrust::for_each(thrust::cuda::par.on(d_stream),
                dst_src_Iterator, dst_src_Iterator + deriv_uvecs[0].size(),
                Get_optDstb_dim0dim1dim2dim3dim4_D(moded_vRangeB0, moded_vRangeB1,
                    moded_wMaxB, moded_dMaxA_0_dMaxB_0, moded_dMaxA_1_dMaxB_1));
        }
        else {
            const FLOAT_TYPE d0 = deriv0_ptr[0];
            const FLOAT_TYPE d1 = deriv1_ptr[0];
            const FLOAT_TYPE d2 = deriv2_ptr[0];
            thrust::transform(thrust::cuda::par.on(d_stream),
                y2_dev_ptr, y2_dev_ptr + x_uvecs[2].size(), dOpt0_dev_ptr,
                Get_optDstb_dim0_d(moded_vRangeB0, moded_vRangeB1, d0, d1));
            const FLOAT_TYPE det1 = d2;
            const FLOAT_TYPE det4 = d2;
            dOpt1_ptr[0] = (det1 >= 0) ? moded_wMaxB : -moded_wMaxB;
            dOpt4_ptr[0] = (det4 >= 0) ? moded_dMaxA_1_dMaxB_1 : -moded_dMaxA_1_dMaxB_1;
            const FLOAT_TYPE denom = sqrt_float_type<FLOAT_TYPE>(d0 * d0 + d1 * d1);
            dOpt2_ptr[0] = (denom == 0) ? 0 : moded_dMaxA_0_dMaxB_0 * d0 / denom;
            dOpt3_ptr[0] = (denom == 0) ? 0 : moded_dMaxA_0_dMaxB_0 * d1 / denom;
        }
    }
    else {
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
        FLOAT_TYPE Fxf     = (Fx >= 0)*Fx*X1::fwd_frac + (Fx < 0)*Fx*X1::fwb_frac;    // no branches!
        FLOAT_TYPE Fxr     = (Fx >= 0)*Fx*X1::rwd_frac + (Fx < 0)*Fx*X1::rwb_frac;    // no branches!
        FLOAT_TYPE Fx_drag = -X1::Cd0 - Ux * (X1::Cd1 + X1::Cd2 * Ux);
        FLOAT_TYPE af = atan2_float_type<FLOAT_TYPE>(Uy + X1::a * r, Ux) - d;
        FLOAT_TYPE ar = atan2_float_type<FLOAT_TYPE>(Uy - X1::b * r, Ux);
        FLOAT_TYPE Fzf = (X1::m * X1::G * X1::b - X1::h * Fx) / X1::L;
        FLOAT_TYPE Fzr = (X1::m * X1::G * X1::a + X1::h * Fx) / X1::L;
        // Fyf (inlined BicycleCAvoid::fialaTireModel(af, X1::Caf, X1::mu, Fxf, Fzf))
        FLOAT_TYPE a  = af;
        FLOAT_TYPE Ca = X1::Caf;
        FLOAT_TYPE mu = X1::mu;
        FLOAT_TYPE Fx = Fxf;
        FLOAT_TYPE Fz = Fzf;
        FLOAT_TYPE Fymax = sqrt_float_type<FLOAT_TYPE>((abs_float_type<FLOAT_TYPE>(Fx) < Fmax)*(Fmax * Fmax - Fx * Fx));    // no branches!
        FLOAT_TYPE tana  = tan_float_type<FLOAT_TYPE>(a);
        FLOAT_TYPE ratio = abs_float_type<FLOAT_TYPE>(tana * Ca / (3 * (Fymax > 0) * Fymax + (Fymax <= 0))) + (Fymax <= 0); // ratio >= 1 if Fymax <= 0
        FLOAT_TYPE Fyf = (ratio < 1)*(-Ca * tana * (1 - ratio + ratio * ratio / 3)) + (ratio >= 1)*(-copysign_float_type<FLOAT_TYPE>(Fymax, tana));
        // Fyr (inlined BicycleCAvoid::fialaTireModel(ar, X1::Car, X1::mu, Fxr, Fzr))
        a  = ar;
        Ca = X1::Car;
        mu = X1::mu;
        Fx = Fxr;
        Fz = Fzr;
        Fymax = sqrt_float_type<FLOAT_TYPE>((abs_float_type<FLOAT_TYPE>(Fx) < Fmax)*(Fmax * Fmax - Fx * Fx));    // no branches!
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
        FLOAT_TYPE Fxf     = (Fx >= 0)*Fx*X1::fwd_frac + (Fx < 0)*Fx*X1::fwb_frac;    // no branches!
        FLOAT_TYPE Fxr     = (Fx >= 0)*Fx*X1::rwd_frac + (Fx < 0)*Fx*X1::rwb_frac;    // no branches!
        FLOAT_TYPE af = atan2_float_type<FLOAT_TYPE>(Uy + X1::a * r, Ux) - d;
        FLOAT_TYPE ar = atan2_float_type<FLOAT_TYPE>(Uy - X1::b * r, Ux);
        FLOAT_TYPE Fzf = (X1::m * X1::G * X1::b - X1::h * Fx) / X1::L;
        FLOAT_TYPE Fzr = (X1::m * X1::G * X1::a + X1::h * Fx) / X1::L;
        // Fyf (inlined BicycleCAvoid::fialaTireModel(af, X1::Caf, X1::mu, Fxf, Fzf))
        FLOAT_TYPE a  = af;
        FLOAT_TYPE Ca = X1::Caf;
        FLOAT_TYPE mu = X1::mu;
        FLOAT_TYPE Fx = Fxf;
        FLOAT_TYPE Fz = Fzf;
        FLOAT_TYPE Fymax = sqrt_float_type<FLOAT_TYPE>((abs_float_type<FLOAT_TYPE>(Fx) < Fmax)*(Fmax * Fmax - Fx * Fx));    // no branches!
        FLOAT_TYPE tana  = tan_float_type<FLOAT_TYPE>(a);
        FLOAT_TYPE ratio = abs_float_type<FLOAT_TYPE>(tana * Ca / (3 * (Fymax > 0) * Fymax + (Fymax <= 0))) + (Fymax <= 0); // ratio >= 1 if Fymax <= 0
        FLOAT_TYPE Fyf = (ratio < 1)*(-Ca * tana * (1 - ratio + ratio * ratio / 3)) + (ratio >= 1)*(-copysign_float_type<FLOAT_TYPE>(Fymax, tana));
        // Fyr (inlined BicycleCAvoid::fialaTireModel(ar, X1::Car, X1::mu, Fxr, Fzr))
        a  = ar;
        Ca = X1::Car;
        mu = X1::mu;
        Fx = Fxr;
        Fz = Fzr;
        Fymax = sqrt_float_type<FLOAT_TYPE>((abs_float_type<FLOAT_TYPE>(Fx) < Fmax)*(Fmax * Fmax - Fx * Fx));    // no branches!
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
        FLOAT_TYPE Fxf     = (Fx >= 0)*Fx*X1::fwd_frac + (Fx < 0)*Fx*X1::fwb_frac;    // no branches!
        FLOAT_TYPE Fxr     = (Fx >= 0)*Fx*X1::rwd_frac + (Fx < 0)*Fx*X1::rwb_frac;    // no branches!
        FLOAT_TYPE af = atan2_float_type<FLOAT_TYPE>(Uy + X1::a * r, Ux) - d;
        FLOAT_TYPE ar = atan2_float_type<FLOAT_TYPE>(Uy - X1::b * r, Ux);
        FLOAT_TYPE Fzf = (X1::m * X1::G * X1::b - X1::h * Fx) / X1::L;
        FLOAT_TYPE Fzr = (X1::m * X1::G * X1::a + X1::h * Fx) / X1::L;
        // Fyf (inlined BicycleCAvoid::fialaTireModel(af, X1::Caf, X1::mu, Fxf, Fzf))
        FLOAT_TYPE a  = af;
        FLOAT_TYPE Ca = X1::Caf;
        FLOAT_TYPE mu = X1::mu;
        FLOAT_TYPE Fx = Fxf;
        FLOAT_TYPE Fz = Fzf;
        FLOAT_TYPE Fymax = sqrt_float_type<FLOAT_TYPE>((abs_float_type<FLOAT_TYPE>(Fx) < Fmax)*(Fmax * Fmax - Fx * Fx));    // no branches!
        FLOAT_TYPE tana  = tan_float_type<FLOAT_TYPE>(a);
        FLOAT_TYPE ratio = abs_float_type<FLOAT_TYPE>(tana * Ca / (3 * (Fymax > 0) * Fymax + (Fymax <= 0))) + (Fymax <= 0); // ratio >= 1 if Fymax <= 0
        FLOAT_TYPE Fyf = (ratio < 1)*(-Ca * tana * (1 - ratio + ratio * ratio / 3)) + (ratio >= 1)*(-copysign_float_type<FLOAT_TYPE>(Fymax, tana));
        // Fyr (inlined BicycleCAvoid::fialaTireModel(ar, X1::Car, X1::mu, Fxr, Fzr))
        a  = ar;
        Ca = X1::Car;
        mu = X1::mu;
        Fx = Fxr;
        Fz = Fzr;
        Fymax = sqrt_float_type<FLOAT_TYPE>((abs_float_type<FLOAT_TYPE>(Fx) < Fmax)*(Fmax * Fmax - Fx * Fx));    // no branches!
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

    thrust::device_ptr<FLOAT_TYPE> x_rel_dev_ptr   = thrust::device_pointer_cast(x_rel_ptr);
    thrust::device_ptr<FLOAT_TYPE> y_rel_dev_ptr   = thrust::device_pointer_cast(y_rel_ptr);
    thrust::device_ptr<FLOAT_TYPE> psi_rel_dev_ptr = thrust::device_pointer_cast(psi_rel_ptr);
    thrust::device_ptr<FLOAT_TYPE> Ux_dev_ptr      = thrust::device_pointer_cast(Ux_ptr);
    thrust::device_ptr<FLOAT_TYPE> Uy_dev_ptr      = thrust::device_pointer_cast(Uy_ptr);
    thrust::device_ptr<FLOAT_TYPE> V_dev_ptr       = thrust::device_pointer_cast(V_ptr);
    thrust::device_ptr<FLOAT_TYPE> r_dev_ptr       = thrust::device_pointer_cast(r_ptr);

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
        auto Tuple = thrust::make_tuple(dx_dim_dev_ptr,
                                        psi_rel_dev_ptr,
                                        y_rel_dev_ptr,
                                        Ux_dev_ptr,
                                        V_dev_ptr,
                                        r_dev_ptr);
        auto Iterator = thrust::make_zip_iterator(Tuple);
        thrust::for_each(thrust::cuda::par.on(dx_stream), Iterator, Iterator + dx_uvec.size(), Get_dynamics_x_rel());
        break;
    case 1:    // dy_rel
        auto Tuple = thrust::make_tuple(dx_dim_dev_ptr,
                                        psi_rel_dev_ptr,
                                        x_rel_dev_ptr,
                                        Uy_dev_ptr,
                                        V_dev_ptr,
                                        r_dev_ptr);
        auto Iterator = thrust::make_zip_iterator(Tuple);
        thrust::for_each(thrust::cuda::par.on(dx_stream), Iterator, Iterator + dx_uvec.size(), Get_dynamics_y_rel());
        break;
    case 2:    // dpsi_rel
        if (beacls::is_cuda(w_uvec)) {
            thrust::device_ptr<const FLOAT_TYPE> w_dev_ptr = thrust::device_pointer_cast(beacls::UVec_<FLOAT_TYPE>(w_uvec).ptr());
            beacls::synchronizeUVec(w_uvec);
            auto Tuple = thrust::make_tuple(dx_dim_dev_ptr,
                                            r_dev_ptr,
                                            w_dev_ptr);
            auto Iterator = thrust::make_zip_iterator(Tuple);
            thrust::for_each(thrust::cuda::par.on(dx_stream), Iterator, Iterator + dx_uvec.size(), Get_dynamics_psi_rel__W());
        } else {
            const FLOAT_TYPE w = beacls::UVec_<FLOAT_TYPE>(w_uvec).ptr()[0]
            auto Tuple = thrust::make_tuple(dx_dim_dev_ptr,
                                            r_dev_ptr);
            auto Iterator = thrust::make_zip_iterator(Tuple);
            thrust::for_each(thrust::cuda::par.on(dx_stream), Iterator, Iterator + dx_uvec.size(), Get_dynamics_psi_rel__w(w));
        }
        break;
    case 3:    // dUx
        if (beacls::is_cuda(Fx_uvec)) {
            thrust::device_ptr<const FLOAT_TYPE> Fx_dev_ptr = thrust::device_pointer_cast(beacls::UVec_<FLOAT_TYPE>(Fx_uvec).ptr());
            beacls::synchronizeUVec(Fx_uvec);
            auto Tuple = thrust::make_tuple(dx_dim_dev_ptr,
                                            Ux_dev_ptr,
                                            Uy_dev_ptr,
                                            r_dev_ptr,
                                            Fx_dev_ptr);
            auto Iterator = thrust::make_zip_iterator(Tuple);
            thrust::for_each(thrust::cuda::par.on(dx_stream), Iterator, Iterator + dx_uvec.size(), Get_dynamics_Ux__FX());
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
            auto Tuple = thrust::make_tuple(dx_dim_dev_ptr,
                                            Ux_dev_ptr,
                                            Uy_dev_ptr,
                                            r_dev_ptr,
                                            d_dev_ptr,
                                            Fx_dev_ptr);
            auto Iterator = thrust::make_zip_iterator(Tuple);
            thrust::for_each(thrust::cuda::par.on(dx_stream), Iterator, Iterator + dx_uvec.size(), Get_dynamics_Uy__D_FX());
        } else {
            std::cerr << __FILE__ << ":" << __LINE__ << ":" << __FUNCTION__ << " Invalid data size" << std::endl;
            return false;
        }
        break;
    case 5:    // dV
        if (beacls::is_cuda(a_uvec)) {
            thrust::device_ptr<const FLOAT_TYPE> a_dev_ptr = thrust::device_pointer_cast(beacls::UVec_<FLOAT_TYPE>(a_uvec).ptr());
            beacls::synchronizeUVec(a_uvec);
            auto Tuple = thrust::make_tuple(dx_dim_dev_ptr,
                                            a_dev_ptr);
            auto Iterator = thrust::make_zip_iterator(Tuple);
            thrust::for_each(thrust::cuda::par.on(dx_stream), Iterator, Iterator + dx_uvec.size(), Get_dynamics_V__A());
        } else {
            const FLOAT_TYPE a = beacls::UVec_<FLOAT_TYPE>(a_uvec).ptr()[0]
            auto Tuple = thrust::make_tuple(dx_dim_dev_ptr);
            auto Iterator = thrust::make_zip_iterator(Tuple);
            thrust::for_each(thrust::cuda::par.on(dx_stream), Iterator, Iterator + dx_uvec.size(), Get_dynamics_V__a(a));
        }
        break;
    case 6:    // dr
        if (beacls::is_cuda(Fx_uvec) && beacls::is_cuda(d_uvec)) {
            thrust::device_ptr<const FLOAT_TYPE> Fx_dev_ptr = thrust::device_pointer_cast(beacls::UVec_<FLOAT_TYPE>(Fx_uvec).ptr());
            thrust::device_ptr<const FLOAT_TYPE> d_dev_ptr  = thrust::device_pointer_cast(beacls::UVec_<FLOAT_TYPE>(d_uvec).ptr());
            beacls::synchronizeUVec(Fx_uvec);
            beacls::synchronizeUVec(d_uvec);
            auto Tuple = thrust::make_tuple(dx_dim_dev_ptr,
                                            Ux_dev_ptr,
                                            Uy_dev_ptr,
                                            r_dev_ptr,
                                            d_dev_ptr,
                                            Fx_dev_ptr);
            auto Iterator = thrust::make_zip_iterator(Tuple);
            thrust::for_each(thrust::cuda::par.on(dx_stream), Iterator, Iterator + dx_uvec.size(), Get_dynamics_r__D_FX());
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
