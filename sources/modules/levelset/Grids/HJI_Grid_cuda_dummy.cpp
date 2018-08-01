#include <typedef.hpp>
#include <cuda_macro.hpp>
#include <algorithm>
#include "HJI_Grid_cuda.hpp"

#if !defined(WITH_GPU)
void HJI_Grid_repeat_cuda
(
	beacls::UVec& dst,
	const beacls::UVec& src,
	const size_t dimension,
	const size_t start_index,
	const size_t loop_length,
	const size_t stride,
	const size_t dim_length
) {
	beacls::synchronizeUVec(src);
}

#endif /* !defined(WITH_GPU)  */
