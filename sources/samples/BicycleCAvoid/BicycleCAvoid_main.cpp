#define _USE_MATH_DEFINES
#include <levelset/levelset.hpp>
#include <helperOC/helperOC.hpp>
#include <helperOC/DynSys/DynSys/DynSysSchemeData.hpp>
#include <helperOC/DynSys/BicycleCAvoid/BicycleCAvoid.hpp>
#include <helperOC/ComputeGradients.hpp>
#include <cmath>
#include <numeric>
#include <functional>
#include <cfloat>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <cstring>

/**
    @brief Tests the Bicycle class by computing a reachable set and then computing the optimal trajectory from the reachable set.
    */
int main(int argc, char *argv[])
{
    // 0 (name); 1 is the grid size (comma separated, with last number repeated across all remaining dimensions)
    beacls::IntegerVec gN;
    gN.resize(7, 11);
    if (argc > 1) {
        std::stringstream ss(std::string(argv[1]));
        int d, i = 0;
        while (ss >> d) {
            gN[i] = d;
            i++;
            if (ss.peek() == ',')
                ss.ignore();
        }
        while (i < 7) {
            gN[i] = gN[i-1];
            i++;
        }
    }

    /////////////////////////////////////////////////////
    // Ask Mo about these settings
    // Not sure what all the settings are...?
    // dump_file, line_length_of_chunk, model_size,
    bool keepLast = true;
    bool calculateTTRduringSolving = false;
    levelset::DelayedDerivMinMax_Type delayedDerivMinMax = levelset::DelayedDerivMinMax_Disable;

    bool useCuda = false;
    if (argc > 2) {
        useCuda = (atoi(argv[2]) == 0) ? false : true;
    }
    int num_of_threads = 0;
    if (argc > 3) {
        num_of_threads = atoi(argv[3]);
    }
    int num_of_gpus = 0;
    if (argc > 4) {
        num_of_gpus = atoi(argv[4]);
    }
    size_t line_length_of_chunk = 1;
    if (argc > 5) {
        line_length_of_chunk = atoi(argv[5]);
    }
    bool enable_user_defined_dynamics_on_gpu = true;
    if (argc > 6) {
        enable_user_defined_dynamics_on_gpu = (atoi(argv[6]) == 0) ? false : true;
    }

    //!< Grid
    //!< Choose this to be just big enough to cover the reachable set

//  ..|'''.|  '||''|.   '||' '||''|.       .|'''.|  '||' |'''''||  '||''''|  
// .|'     '   ||   ||   ||   ||   ||      ||..  '   ||      .|'    ||  .    
// ||    ....  ||''|'    ||   ||    ||      ''|||.   ||     ||      ||''|    
// '|.    ||   ||   |.   ||   ||    ||    .     '||  ||   .|'       ||       
//  ''|...'|  .||.  '|' .||. .||...|'     |'....|'  .||. ||......| .||.....| 
                                                                          
                                                                          

    // const beacls::FloatVec gMin = beacls::FloatVec{ (FLOAT_TYPE)-10, (FLOAT_TYPE)-15, (FLOAT_TYPE)0 };
    // const beacls::FloatVec gMax = beacls::FloatVec{ (FLOAT_TYPE)25, (FLOAT_TYPE)15, (FLOAT_TYPE)(2*M_PI) };
    // x_rel, y_rel, psi_rel, Ux, Uy, v, r
    const beacls::FloatVec gMin = beacls::FloatVec{ (FLOAT_TYPE)(-15),
                                                    (FLOAT_TYPE)(-5),
                                                    (FLOAT_TYPE)(-M_PI/2),
                                                    (FLOAT_TYPE)3,
                                                    (FLOAT_TYPE)(-5),
                                                    (FLOAT_TYPE)3,
                                                    (FLOAT_TYPE)(-1) };
    const beacls::FloatVec gMax = beacls::FloatVec{ (FLOAT_TYPE)15,
                                                    (FLOAT_TYPE)5,
                                                    (FLOAT_TYPE)M_PI/2,
                                                    (FLOAT_TYPE)12,
                                                    (FLOAT_TYPE)5,
                                                    (FLOAT_TYPE)12,
                                                    (FLOAT_TYPE)1 };
    std::cout << "Running with grid sizes/bounds:" << std::endl;
    std::cout << "  x_rel: [" << gMin[0] << ", " << gMax[0] << "] (" << gN[0] << ")" << std::endl;
    std::cout << "  y_rel: [" << gMin[1] << ", " << gMax[1] << "] (" << gN[1] << ")" << std::endl;
    std::cout << "psi_rel: [" << gMin[2] << ", " << gMax[2] << "] (" << gN[2] << ")" << std::endl;
    std::cout << "     Ux: [" << gMin[3] << ", " << gMax[3] << "] (" << gN[3] << ")" << std::endl;
    std::cout << "     Uy: [" << gMin[4] << ", " << gMax[4] << "] (" << gN[4] << ")" << std::endl;
    std::cout << "      v: [" << gMin[5] << ", " << gMax[5] << "] (" << gN[5] << ")" << std::endl;
    std::cout << "      r: [" << gMin[6] << ", " << gMax[6] << "] (" << gN[6] << ")" << std::endl;
    levelset::HJI_Grid* g = helperOC::createGrid(gMin, gMax, gN); // , beacls::IntegerVec{2});

    //!< Time
    //!< Choose tMax to be large enough for the set to converge
    // Ask Mo what are reasonable values
    const FLOAT_TYPE tMax = 3;
    const FLOAT_TYPE dt = (FLOAT_TYPE)0.1;
    const beacls::FloatVec tau = generateArithmeticSequence<FLOAT_TYPE>(0., dt, tMax);

                                                                                                                                               

//     |     '||'  '|'  ..|''||   '||' '||''|.       .|'''.|  '||''''|  |''||''| 
//    |||     '|.  .'  .|'    ||   ||   ||   ||      ||..  '   ||  .       ||    
//   |  ||     ||  |   ||      ||  ||   ||    ||      ''|||.   ||''|       ||    
//  .''''|.     |||    '|.     ||  ||   ||    ||    .     '||  ||          ||    
// .|.  .||.     |      ''|...|'  .||. .||...|'     |'....|'  .||.....|   .||.   
                                                                              
    // Ask Mo about this definition.
    const FLOAT_TYPE targetR = 3; //!< collision radius
    beacls::FloatVec data0;
    // ignore all dimensions except x_rel,y_rel 
    levelset::ShapeCylinder(beacls::IntegerVec{2, 3, 4, 5, 6}, beacls::FloatVec{ (FLOAT_TYPE)0, (FLOAT_TYPE)0}, targetR).execute(g, data0);

    //!< Additional solver parameters
    helperOC::DynSysSchemeData* sD = new helperOC::DynSysSchemeData;
    sD->set_grid(g);
    // Note: Initial conditions are needed if you want a trajectory.
    sD->dynSys = new helperOC::BicycleCAvoid(beacls::FloatVec{(FLOAT_TYPE)0.,
                                                              (FLOAT_TYPE)0.,
                                                              (FLOAT_TYPE)0.,
                                                              (FLOAT_TYPE)0.,
                                                              (FLOAT_TYPE)0.,
                                                              (FLOAT_TYPE)0.,
                                                              (FLOAT_TYPE)0.},
                                                              beacls::IntegerVec{ 0,1,2,3,4,5,6 });
    sD->uMode = helperOC::DynSys_UMode_Max;
    sD->dMode = helperOC::DynSys_DMode_Min;

    // Target set and visualization
    helperOC::HJIPDE_extraArgs extraArgs;
    helperOC::HJIPDE_extraOuts extraOuts;
    extraArgs.visualize = false;
    extraArgs.deleteLastPlot = true;
    extraArgs.keepLast = true;

    extraArgs.execParameters.line_length_of_chunk = line_length_of_chunk;
    extraArgs.execParameters.calcTTR = calculateTTRduringSolving;
    extraArgs.keepLast = keepLast;
    extraArgs.execParameters.useCuda = useCuda;
    extraArgs.execParameters.num_of_gpus = num_of_gpus;
    extraArgs.execParameters.num_of_threads = num_of_threads;
    extraArgs.execParameters.delayedDerivMinMax = delayedDerivMinMax;
    extraArgs.execParameters.enable_user_defined_dynamics_on_gpu = enable_user_defined_dynamics_on_gpu;

    helperOC::HJIPDE* hjipde = new helperOC::HJIPDE();

    //!< Call solver and save

    beacls::FloatVec tau2;
    std::vector<beacls::FloatVec > datas;
    hjipde->solve(datas, tau2, extraOuts, data0, tau, sD, helperOC::HJIPDE::MinWithType_Zero, extraArgs);
    if (hjipde) delete hjipde;

    beacls::UVecType execType = useCuda ? beacls::UVecType_Cuda : beacls::UVecType_Vector;
    std::vector<beacls::FloatVec> derivC;
    std::vector<beacls::FloatVec> derivL;
    std::vector<beacls::FloatVec> derivR;
    beacls::FloatVec data;
    data.reserve(datas[0].size() * datas.size());
    std::for_each(datas.cbegin(), datas.cend(), [&data](const auto& rhs) { 
        data.insert(data.end(), rhs.cbegin(), rhs.cend());
    });
    helperOC::ComputeGradients(g, helperOC::ApproximationAccuracy_veryHigh, execType)(
        derivC, derivL, derivR, g, data, data.size(), false, extraArgs.execParameters);

// '||''''| '||' '||'      '||''''|         .|'''.|      |     '||'  '|' '||' '|.   '|'  ..|'''.|  
//  ||  .    ||   ||        ||  .           ||..  '     |||     '|.  .'   ||   |'|   |  .|'     '  
//  ||''|    ||   ||        ||''|            ''|||.    |  ||     ||  |    ||   | '|. |  ||    .... 
//  ||       ||   ||        ||             .     '||  .''''|.     |||     ||   |   |||  '|.    ||  
// .||.     .||. .||.....| .||.....|       |'....|'  .|.  .||.     |     .||. .|.   '|   ''|...'|  
                                              
                                                                                                
    std::string BicycleCAvoid_filename("BicycleCAvoid.mat");

    beacls::MatFStream* fs = beacls::openMatFStream(BicycleCAvoid_filename, beacls::MatOpenMode_Write);
    beacls::MatVariable* struct_var = beacls::createMatStruct("avoid_set");

    beacls::IntegerVec Ns = g->get_Ns();
    g->save_grid(std::string("g"), fs, struct_var);
    save_vector_of_vectors(derivC, std::string("deriv"), Ns, false, fs, struct_var);
    save_vector_of_vectors(datas, std::string("data"), Ns, false, fs, struct_var);
    beacls::writeMatVariable(fs, struct_var);
    beacls::closeMatVariable(struct_var);
    beacls::closeMatFStream(fs);

    if (sD) delete sD;
    if (g) delete g;
    return 0;
}

