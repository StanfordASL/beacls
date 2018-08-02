#include <typedef.hpp>
#include <cuda_macro.hpp>
#include <algorithm>
#include "HJI_Grid_cuda.hpp"

#if !defined(WITH_GPU)
void kernel_HJI_Grid_repeat_cuda(
	FLOAT_TYPE  *dst_ptr,
	const FLOAT_TYPE *src_ptr,
	const size_t start,
	const size_t loop_length,
	const size_t stride,
	const size_t dim_length,
	const size_t blockDim_x,
	const size_t blockIdx_x,
	const size_t gridDim_x,
	const size_t threadIdx_x
	) {
	const size_t tid = threadIdx_x;
	size_t i = (blockIdx_x * blockDim_x + tid);
	const size_t gridSize = blockDim_x * gridDim_x;
	while (i < loop_length) {
		const size_t global_index = i + start;
		const size_t src_idx = (global_index / stride) % dim_length;
		dst_ptr[i] = src_ptr[src_idx];
		i += gridSize;
	}
}
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
	FLOAT_TYPE* dst_ptr = beacls::UVec_<FLOAT_TYPE>(dst).ptr();
	const FLOAT_TYPE* src_ptr = beacls::UVec_<const FLOAT_TYPE>(src).ptr();
	size_t num_of_threads_x;
	size_t num_of_blocks_x;
	get_cuda_thread_size_1d<size_t>(
		num_of_threads_x,
		num_of_blocks_x,
		loop_length,
		512
		);
	for (size_t blockIdx_x = 0; blockIdx_x < num_of_blocks_x; ++blockIdx_x) {
		for (size_t threadIdx_x = 0; threadIdx_x < num_of_threads_x; ++threadIdx_x) {
			kernel_HJI_Grid_repeat_cuda(
				dst_ptr, src_ptr,
				start_index, loop_length, stride, dim_length,
				num_of_blocks_x,
				blockIdx_x,
				num_of_threads_x,
				threadIdx_x);
		}
	}
}

#endif /* !defined(WITH_GPU)  */
