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
 * Authors: Stanford ASL
 */

///////////////////////////////////////////////////////////////////////////////
//
// DynSys subclass: relative dynamics between a bicycle model car and a
//                  dynamically extended (+acceleration) simple car
//
///////////////////////////////////////////////////////////////////////////////

#ifndef __BicycleCAvoid_hpp__
#define __BicycleCAvoid_hpp__

//! Prefix to generate Visual C++ DLL
#ifdef _MAKE_VC_DLL
#define PREFIX_VC_DLL __declspec(dllexport)

//! Don't add prefix, except dll generating
#else
#define PREFIX_VC_DLL
#endif

#include <helperOC/DynSys/BicycleCAvoid/X1Params.hpp>
#include <helperOC/DynSys/DynSys/DynSys.hpp>
#include <typedef.hpp>
#include <cstddef>
#include <vector>
#include <iostream>
#include <cstring>
#include <utility>
using namespace std::rel_ops;
namespace helperOC {
    class BicycleCAvoid : public DynSys {
    public:
    protected:
        beacls::IntegerVec dims;    //!< Dimensions that are active
    public:
        /* TODO: change
        @brief Constructor. Creates a Dubins Car object with a unique ID,
            state x, and reachable set information reachInfo
        Dynamics:
            \dot{x}_1 = v * cos(x_3) + d1
            \dot{x}_2 = v * sin(x_3) + d2
            \dot{x}_3 = u            + d3
                v \in [vrange(1), vrange(2)]
                u \in [-wMax, wMax]

        @param  [in]        x   state [xpos; ypos; theta]
        @param  [in]        uMax    maximum turn rate
        @param  [in]        dMax    maximum turn rate
        @param  [in]        va  Vehicle A Speeds
        @param  [in]        vd  Vehicle B Speeds
        @return a Dubins Car object
        */
        PREFIX_VC_DLL
            BicycleCAvoid(
                const beacls::FloatVec& x,
                const beacls::IntegerVec& dims = beacls::IntegerVec{ 0,1,2,3,4,5,6 }
            );
        PREFIX_VC_DLL
            BicycleCAvoid(
                beacls::MatFStream* fs,
                beacls::MatVariable* variable_ptr = NULL
            );
        PREFIX_VC_DLL
            virtual ~BicycleCAvoid();
        PREFIX_VC_DLL
            virtual bool operator==(const BicycleCAvoid& rhs) const;
        PREFIX_VC_DLL
            virtual bool operator==(const DynSys& rhs) const;
        PREFIX_VC_DLL
            virtual bool save(
                beacls::MatFStream* fs,
                beacls::MatVariable* variable_ptr = NULL
            );
        virtual BicycleCAvoid* clone() const {
            return new BicycleCAvoid(*this);
        }
        /*
        @brief Optimal control function
        */
        PREFIX_VC_DLL
            bool optCtrl(
                std::vector<beacls::FloatVec >& uOpts,
                const FLOAT_TYPE t,
                const std::vector<beacls::FloatVec::const_iterator >& x_ites,
                const std::vector<const FLOAT_TYPE*>& deriv_ptrs,
                const beacls::IntegerVec& x_sizes,
                const beacls::IntegerVec& deriv_sizes,
                const helperOC::DynSys_UMode_Type uMode
            ) const;
        /*
        @brief Optimal disturbance function
        */
        PREFIX_VC_DLL
            bool optDstb(
                std::vector<beacls::FloatVec >& dOpts,
                const FLOAT_TYPE t,
                const std::vector<beacls::FloatVec::const_iterator >& x_ites,
                const std::vector<const FLOAT_TYPE*>& deriv_ptrs,
                const beacls::IntegerVec& x_sizes,
                const beacls::IntegerVec& deriv_sizes,
                const helperOC::DynSys_DMode_Type dMode
            ) const;

        // /*
        // @brief Helper function for computing control constraints
        // */
        // bool BicycleCAvoid::ctrlConstraint(std::vector<FLOAT_TYPE>& b,
        //     std::vector<beacls::FloatVec>& m,
        //     std::vector<beacls::FloatVec>& us,
        //     const std::vector<beacls::FloatVec::const_iterator>& x_ites, 
        //     const std::vector<const FLOAT_TYPE*>& deriv_ptrs,  
        //     const beacls::IntegerVec& x_sizes, // xsizes
        //     const beacls::IntegerVec& deriv_sizes,
        //     const helperOC::DynSys_DMode_Type dMode
        //     ) const;

        /*
        @brief Helper function for dynamics
        */
        bool dynamics_cell_helper(
            std::vector<beacls::FloatVec>& dxs,
            const beacls::FloatVec::const_iterator& x_rel,
            const beacls::FloatVec::const_iterator& y_rel,
            const beacls::FloatVec::const_iterator& psi_rel,
            const beacls::FloatVec::const_iterator& Ux,
            const beacls::FloatVec::const_iterator& Uy,
            const beacls::FloatVec::const_iterator& V,
            const beacls::FloatVec::const_iterator& r,
            const std::vector<beacls::FloatVec >& us,
            const std::vector<beacls::FloatVec >& ds,
            const size_t size_x_rel,
            const size_t size_y_rel,
            const size_t size_psi_rel,
            const size_t size_Ux,
            const size_t size_Uy,
            const size_t size_V,
            const size_t size_r,
            const size_t dim
        ) const;
        /*
        @brief Dynamics of the BicycleCAvoid system
        */
        PREFIX_VC_DLL
            bool dynamics(
                std::vector<beacls::FloatVec >& dx,
                const FLOAT_TYPE t,
                const std::vector<beacls::FloatVec::const_iterator >& x_ites,
                const std::vector<beacls::FloatVec >& us,
                const std::vector<beacls::FloatVec >& ds,
                const beacls::IntegerVec& x_sizes,
                const size_t dst_target_dim
            ) const;


///////////////////////////////////////////////////////////////////////////////
//
//         Cuda functions below.  ---//---  Not implemented for now.
//
///////////////////////////////////////////////////////////////////////////////


// #if defined(USER_DEFINED_GPU_DYNSYS_FUNC) && 0
//         PREFIX_VC_DLL
//             bool optCtrl_cuda(
//                 std::vector<beacls::UVec>& u_uvecs,
//                 const FLOAT_TYPE t,
//                 const std::vector<beacls::UVec>& x_uvecs,
//                 const std::vector<beacls::UVec>& deriv_uvecs,
//                 const helperOC::DynSys_UMode_Type uMode
//             ) const;
//         /*
//         @brief Optimal disturbance function
//         */
//         PREFIX_VC_DLL
//             bool optDstb_cuda(
//                 std::vector<beacls::UVec>& d_uvecs,
//                 const FLOAT_TYPE t,
//                 const std::vector<beacls::UVec>& x_uvecs,
//                 const std::vector<beacls::UVec>& deriv_uvecs,
//                 const helperOC::DynSys_DMode_Type dMode
//             ) const;
//         /*
//         @brief Dynamics of the BicycleCAvoid system
//         */
//         PREFIX_VC_DLL
//             bool dynamics_cuda(
//                 std::vector<beacls::UVec>& dx_uvecs,
//                 const FLOAT_TYPE t,
//                 const std::vector<beacls::UVec>& x_uvecs,
//                 const std::vector<beacls::UVec>& u_uvecs,
//                 const std::vector<beacls::UVec>& d_uvecs,
//                 const size_t dst_target_dim
//             ) const;
// #endif /* defined(USER_DEFINED_GPU_DYNSYS_FUNC) */

    private:
        /** @overload
        Disable operator=
        */
        BicycleCAvoid& operator=(const BicycleCAvoid& rhs);
        /** @overload
        Disable copy constructor
        */
        BicycleCAvoid(const BicycleCAvoid& rhs) :
            DynSys(rhs),
            dims(rhs.dims)  //!< Dimensions that are active
        {}

        FLOAT_TYPE fialaTireModel(const FLOAT_TYPE a,
                                  const FLOAT_TYPE Ca,
                                  const FLOAT_TYPE mu,
                                  const FLOAT_TYPE Fx,
                                  const FLOAT_TYPE Fz);

        bool gradientFialaTireModel(FLOAT_TYPE& da,
                                  FLOAT_TYPE& dfx,
                                  FLOAT_TYPE& dfz
                                  const FLOAT_TYPE a,
                                  const FLOAT_TYPE Ca,
                                  const FLOAT_TYPE mu,
                                  const FLOAT_TYPE Fx,
                                  const FLOAT_TYPE Fz);

        FLOAT_TYPE getFxf(const FLOAT_TYPE Fx) {
            return (Fx > 0) ? Fx*X1::fwd_frac : Fx*X1::fwb_frac;
        }
        FLOAT_TYPE getFxr(const FLOAT_TYPE Fx) {
            return (Fx > 0) ? Fx*X1::rwd_frac : Fx*X1::rwb_frac;
        }
    };
};
#endif  /* __BicycleCAvoid_hpp__ */
