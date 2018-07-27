#ifndef __X1Params_hpp__
#define __X1Params_hpp__

#include <cmath>
#include <typedef.hpp>

namespace X1 {

static const FLOAT_TYPE G = 9.80665;                                                   // gravity
static const FLOAT_TYPE mfl = 484;                                                     // weight at front left wheel (kg)
static const FLOAT_TYPE mfr = 455;                                                     // weight at front right wheel (kg)
static const FLOAT_TYPE mrl = 521;                                                     // weight at rear left wheel (kg)
static const FLOAT_TYPE mrr = 504;                                                     // weight at rear right wheel (kg)
static const FLOAT_TYPE m = mfl + mfr + mrl + mrr;                                     // total vehicle mass (kg)
static const FLOAT_TYPE Ixx = 175;                                                     // pitch moment of inertia (kg*m^2)
static const FLOAT_TYPE Iyy = 1000;                                                    // roll moment of inertia (kg*m^2)
static const FLOAT_TYPE Izz = 2900;                                                    // yaw moment of inertia (kg*m^2)

// Dimensions
static const FLOAT_TYPE L = 2.87;                                                      // wheelbase (m)
static const FLOAT_TYPE d  = 1.63;                                                     // track width (m)
static const FLOAT_TYPE a = (mrl + mrr)/m*L;                                           // distance from CG to front axle (m)
static const FLOAT_TYPE b = (mfl + mfr)/m*L;                                           // distance from CG to rear axle (m)
static const FLOAT_TYPE hf = 0.1;                                                      // front roll center height (m)
static const FLOAT_TYPE hr = 0.1;                                                      // rear roll center height (m)
static const FLOAT_TYPE h1 = 0.37;                                                     // CG height w.r.t. line connecting front and rear roll centers (m)
static const FLOAT_TYPE h = hf*b/L + hr*a/L + h1;                                      // CG height (m)
static const FLOAT_TYPE ab = a + 0.4953;                                               // distance from CG to front bumper (m)
static const FLOAT_TYPE bb = b + 0.5715;                                               // distance from CG to rear bumper (m)
static const FLOAT_TYPE w = 1.87;                                                      // physical width (m)

// Tire Model Parameters
static const FLOAT_TYPE mu = 0.92;                                                     // coefficient of friction
static const FLOAT_TYPE Caf = 140e3;                                                   // front tire (pair) cornering stiffness (N/rad)
static const FLOAT_TYPE Car = 190e3;                                                   // rear tire (pair) cornering stiffness (N/rad)

// Longitudinal Actuation Parameters
static const FLOAT_TYPE maxFx = 5600;                                                  // max positive longitudinal force (N)
static const FLOAT_TYPE maxPower = 75e3;                                               // max motor power output (W)

// Longitudinal Drag Force Parameters (FxDrag = Cd0 + Cd1*Ux + Cd2*Ux^2)
static const FLOAT_TYPE Cd0 = 241.0;                                                   // rolling resistance (N)
static const FLOAT_TYPE Cd1 = 25.1;                                                    // linear drag coefficint (N/(m/s))
static const FLOAT_TYPE Cd2 = 0.0;                                                     // quadratic "aero" drag coefficint (N/(m/s)^2)

// Drive and Brake Distribution
static const FLOAT_TYPE fwd_frac = 0.0;                                                // front wheel drive fraction for implementing desired Fx
static const FLOAT_TYPE rwd_frac = 1 - fwd_frac;                                       // rear wheel drive fraction for implementing desired Fx
static const FLOAT_TYPE fwb_frac = 0.6;                                                // front wheel brake fraction for implementing desired Fx
static const FLOAT_TYPE rwb_frac = 1 - fwb_frac;                                       // rear wheel brake fraction for implementing desired Fx

// Maximum Brake Force
// static const FLOAT_TYPE minFx = std::max(-m*G*a*mu/(L*rwb_frac + mu*h),
//                                          -m*G*b*mu/(L*fwb_frac - mu*h));
static const FLOAT_TYPE minFx = -16793.7;    // CUDA can't deal with the above for some reason
static const FLOAT_TYPE maxAx = maxFx/m;
static const FLOAT_TYPE minAx = minFx/m;
static const FLOAT_TYPE maxP2mx = maxPower/m;

// Maximum Steering
static const FLOAT_TYPE d_max = 18*3.1415926/180;
// static const FLOAT_TYPE w_per_v_max_lowspeed = std::tan(d_max)/L;
static const FLOAT_TYPE w_per_v_max_lowspeed = 0.113212;    // CUDA can't deal with the above for some reason

static const FLOAT_TYPE maxA_approx = 0.9*mu*G; // TODO: check, used to be w_v_max_highspeed

};

#endif  /* __X1Params_hpp__ */