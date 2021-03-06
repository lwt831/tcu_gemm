#ifndef GEMM_KERNEL_H
#define GEMM_KERNEL_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cooperative_groups.h>
#include <mma.h>
#include <cublas.h>
#include <cublas_v2.h>
#include "checkerror.h"
#include "helper.cuh"

#define WARP_SIZE (32)
#define lane_copy_num (8)

using namespace nvcuda;


//general input kernel(M, N < 16)
template<int BLOCKS_PER_GRID, int WARPS_PER_BLOCK, int WMMA_M, int WMMA_N, int WMMA_K>
__global__ void compute_hgemm_fold_general16(half *A, half *B, float *output, int M, int N, int K){
	const unsigned int blockId = blockIdx.x;
	const unsigned int warpId = threadIdx.x / WARP_SIZE;
	const unsigned int laneId = threadIdx.x % WARP_SIZE;

	__shared__ half a_buff[WARPS_PER_BLOCK*16*16];
	__shared__ half b_buff[WARPS_PER_BLOCK*16*16];
	__shared__ float c_buff[WARPS_PER_BLOCK*16*16];

	wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> A_frag;
	wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> B_frag;
	wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> C_frag;
	wmma::fill_fragment(C_frag, 0.0f);

	typedef half copy_t;
	int seg_num;
	int seg_size;
	if(M > N){
		seg_num = WMMA_M / M;
		seg_size = M * 16;
	}else{
		seg_num = WMMA_M / N;
		seg_size = N * 16;
	}
	//loop over K
	unsigned int kk = (WARPS_PER_BLOCK * blockId + warpId) * WMMA_K * seg_num;
	const unsigned int kk_per_it = BLOCKS_PER_GRID * WARPS_PER_BLOCK * WMMA_K * seg_num;
	//start point at shared mem of each warp
	const half *warp_a_buff = a_buff + (warpId * 256);
	const half *warp_b_buff = b_buff + (warpId * 256);
	//shared mem ptr of each lane
	copy_t *lane_shared_ptr_a = (copy_t *)(warp_a_buff + (laneId/16)*seg_size + laneId%16);
	copy_t *lane_shared_ptr_b = (copy_t *)(warp_b_buff + (laneId/16)*seg_size + laneId%16);

	copy_t *lane_ptr_a = A + kk + laneId;
	copy_t *lane_ptr_b = B + (kk + laneId) * N;

	unsigned int even_seg = seg_num / 2;
	bool odd_seg = seg_num > 2 * even_seg;//is ita odd number of segments in shared mem?
	while(kk < K){
		for(int seg_i = 0; seg_i < even_seg; seg_i++){
			for(int m = 0; m < M; m++){
				*(lane_shared_ptr_a + seg_i * 2 * seg_size + m * 16) = *(lane_ptr_a + seg_i * 32 + m * K);
			}
			for(int n = 0; n < N; n++){
				*(lane_shared_ptr_b + seg_i * 2 * seg_size + n * 16) = *(lane_ptr_b + seg_i * 32 * N + n);
			}
		}
		if(odd_seg){
			if(laneId < 16){
				for(int m = 0; m < M; m++){
					*(lane_shared_ptr_a + even_seg * 2 * seg_size + m * 16) = *(lane_ptr_a + even_seg * 32 + m * K);
				}
				for(int n = 0; n < N; n++){
					*(lane_shared_ptr_b + even_seg * 2 * seg_size + n * 16) = *(lane_ptr_b + even_seg * 32 * N + n);
				}
			}
		}
		wmma::load_matrix_sync(A_frag, warp_a_buff, 16);
		wmma::load_matrix_sync(B_frag, warp_b_buff, 16);
		wmma::mma_sync(C_frag, A_frag, B_frag, C_frag);

		kk += kk_per_it;
		lane_ptr_a += kk_per_it;
		lane_ptr_b += kk_per_it * N;
	}
	wmma::store_matrix_sync(c_buff + warpId * 256, C_frag, 16, wmma::mem_row_major);
	__syncthreads();//必要

	//naive block level reduction
	if(warpId == 0){
		wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> fetch_c_frag;
		int collect_iter = 1;
#pragma unroll
		while(collect_iter < WARPS_PER_BLOCK){
			wmma::load_matrix_sync(fetch_c_frag, c_buff + collect_iter*256, 16, wmma::mem_row_major);
#pragma unroll
			for(int i=0;i<C_frag.num_elements;i++)
				C_frag.x[i] = C_frag.x[i] + fetch_c_frag.x[i];
			collect_iter++;			
		}
		wmma::store_matrix_sync(c_buff, C_frag, 16, wmma::mem_row_major);//store block level res
	}

	__syncthreads();
	//print_matrix_device(a_buff + 1 * 256, 16, 16);////
	if(M > N){
		if(threadIdx.x < seg_num * M * N){
			int lane_cbuff_index = (threadIdx.x%(M*N)/N)*16 + threadIdx.x%N + (threadIdx.x/(M*N))*(M*16 + M);
			atomicAdd(output + threadIdx.x%(M*N), c_buff[lane_cbuff_index]);
		}
	}
	else{
		if(threadIdx.x < seg_num * M * N){
			int lane_cbuff_index = (threadIdx.x%(M*N)/N)*16 + threadIdx.x%N + (threadIdx.x/(M*N))*(N*16 + N);
			atomicAdd(output + threadIdx.x%(M*N), c_buff[lane_cbuff_index]);
		}
	}
}

template<int WARPS_PER_BLOCK, int WMMA_M, int WMMA_N, int WMMA_K>
__global__ void compute_hgemm_skinny_wide(half *A, half *B, float *output, int M, int N, int K){//suppose M <= 16 and N is multiple of 16
	if(blockIdx.y >= N / 16){//多余的block什么都不做
		return;
	}

	const unsigned int warpId = threadIdx.x / WARP_SIZE;
	const unsigned int laneId = threadIdx.x % WARP_SIZE;

	__shared__ half a_buff[WARPS_PER_BLOCK*16*16];
	__shared__ half b_buff[WARPS_PER_BLOCK*16*16];
	__shared__ float c_buff[WARPS_PER_BLOCK*16*16];

	wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> A_frag;
	wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> B_frag;
	wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> C_frag;
	wmma::fill_fragment(C_frag, 0.0f);

	typedef half copy_t;
	//loop over K
	unsigned int kk = (WARPS_PER_BLOCK * blockIdx.x + warpId) * WMMA_K;
	const unsigned int kk_per_it = gridDim.x * WARPS_PER_BLOCK * WMMA_K;
	//start point at shared mem of each warp
	const half *warp_a_buff = a_buff + (warpId * 256);
	const half *warp_b_buff = b_buff + (warpId * 256);
	//shared mem ptr of each lane
	copy_t *lane_shared_ptr_a = (copy_t *)(warp_a_buff + laneId);
	copy_t *lane_shared_ptr_b = (copy_t *)(warp_b_buff + laneId);

	copy_t *lane_ptr_a = A + kk + laneId;
	copy_t *lane_ptr_b = B + (kk + laneId) * N + blockIdx.y * 16;

	while(kk < K){
		if(laneId < 16){
			for(int i = 0; i < M; i++){
				*(lane_shared_ptr_a + i * WMMA_N) = *(lane_ptr_a + i * K);
			}
			for(int i = 0; i < 16; i++){
				*(lane_shared_ptr_b + i * WMMA_N) = *(lane_ptr_b + i);
			}
		}
		wmma::load_matrix_sync(A_frag, warp_a_buff, 16);
		wmma::load_matrix_sync(B_frag, warp_b_buff, 16);
		wmma::mma_sync(C_frag, A_frag, B_frag, C_frag);

		kk += kk_per_it;
		lane_ptr_a += kk_per_it;
		lane_ptr_b += kk_per_it * N;
	}
	wmma::store_matrix_sync(c_buff + warpId * 256, C_frag, 16, wmma::mem_row_major);
	__syncthreads();//必要

	//naive block level reduction
	if(warpId == 0){
		wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> fetch_c_frag;
		wmma::fill_fragment(A_frag, 0.0f);
		wmma::fill_fragment(B_frag, 0.0f);

		int collect_iter = 1;
#pragma unroll
		while(collect_iter < WARPS_PER_BLOCK){
			wmma::load_matrix_sync(fetch_c_frag, c_buff + collect_iter * 256, 16, wmma::mem_row_major);//load c of each warp from shared memory
#pragma unroll
			for(int i = 0;i < C_frag.num_elements;i++)
				C_frag.x[i] = C_frag.x[i] + fetch_c_frag.x[i];
			collect_iter++;			
		}
		wmma::store_matrix_sync(c_buff, C_frag, 16, wmma::mem_row_major);//store block level res
	}
	//TODO blcok level reduction with atomic operation
	for(int i = 0; i < M; i++){

	}
	__syncthreads();		
	if(threadIdx.x < M * 16){
		unsigned int lane_cbuff_index = threadIdx.x;
		float* lane_output_addr = output + (threadIdx.x / 16) * N + blockIdx.y * 16 + threadIdx.x % 16;
		atomicAdd(lane_output_addr, c_buff[lane_cbuff_index]);
	}
}



template<int BLOCKS_PER_GRID, int WARPS_PER_BLOCK>
__global__ void dense_mv(half *A, half *X, float *Y, int M, int K){
	
	const unsigned int warpId = threadIdx.x / WARP_SIZE;
	const unsigned int laneId = threadIdx.x % WARP_SIZE;

	__shared__ half x_buff[WARPS_PER_BLOCK * 256];
	__shared__ half A_buff[WARPS_PER_BLOCK * 256];
	__shared__ float y_buff[WARPS_PER_BLOCK * 256];
	unsigned int global_warp_index = blockIdx.x * WARPS_PER_BLOCK + warpId;
	//unsigned int warp_row_index = 16 * (((blockIdx.x * WARPS_PER_BLOCK + warpId) * 16) / M);
	//unsigned int warp_col_index = 16 * (((blockIdx.x * WARPS_PER_BLOCK + warpId) * 16) % M);
	unsigned int global_step = BLOCKS_PER_GRID * WARPS_PER_BLOCK;

	half *lane_shmem_xptr = x_buff + warpId * 256 + laneId;

	wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> A_frag;
	wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> B_frag;
	wmma::fragment<wmma::accumulator, 16, 16, 16, float> C_frag;
	wmma::fill_fragment(C_frag, 0.0f);

	unsigned it = 0;
	while(global_warp_index + it < (M / 16)){
		unsigned int warp_row_index = 16 * (((global_warp_index + it) * 16) / M);
		unsigned int warp_col_index = 16 * (((global_warp_index + it) * 16) % M);
		half *lane_global_Aptr = A + warp_row_index * M + warp_col_index * 16 + (laneId / 16) * M + laneId % 16;
		half *lane_shmem_Aptr = A_buff + warpId * 256 + laneId;
		for(int i = 0;i < 8; i++){
			*lane_shmem_Aptr = *lane_shmem_Aptr;
			lane_shmem_Aptr += 32;
			lane_global_Aptr += 2*M;
		}
		wmma::load_matrix_sync(A_frag, A_buff + warpId * 256, 16);

		half *lane_global_xptr = X + ((global_warp_index * 16) % M) * 16 + laneId;
		if(laneId < 16)
			*lane_shmem_xptr = *lane_global_xptr;
		wmma::load_matrix_sync(B_frag, x_buff + warpId * 256, 16);
		wmma::mma_sync(C_frag, A_frag, B_frag, C_frag);

		it += global_step;
	}
	wmma::store_matrix_sync(y_buff, C_frag, 16, wmma::mem_row_major);

	float *lane_global_yptr = Y + ((global_warp_index * 16) % M) * 16 + laneId;
	if(laneId < 16)
		atomicAdd(lane_global_yptr, y_buff[warpId * 256 + laneId]);
	
}




template<int BLOCKS_PER_GRID, int WARPS_PER_BLOCK, int WMMA_M, int WMMA_N, int WMMA_K>
__global__ void compute_hgemm_kernel_pad(half *A, half *B, float *output, int M, int N, int K){

	const unsigned int blockId = blockIdx.x;
	const unsigned int warpId = threadIdx.x / WARP_SIZE;
	const unsigned int laneId = threadIdx.x % WARP_SIZE;

	typedef half copy_t;

	__shared__ half a_buff[WARPS_PER_BLOCK*16*16];
	__shared__ half b_buff[WARPS_PER_BLOCK*16*16];
	__shared__ float c_buff[WARPS_PER_BLOCK*16*16];
	__shared__ half tail_buffer[2*256];

	wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> A_frag;
	wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> B_frag;
	wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> C_frag;
	wmma::fill_fragment(C_frag, 0.0f);

	//loop over K 
	unsigned int kk = (WARPS_PER_BLOCK * blockId + warpId) * WMMA_K;
	const unsigned int kk_per_it = BLOCKS_PER_GRID * WARPS_PER_BLOCK * WMMA_K;
	//start point at shared mem of each warp
	const half *warp_a_buff = a_buff + (warpId * 256);
	const half *warp_b_buff = b_buff + (warpId * 256);
	//shared mem ptr of each lane
	copy_t *lane_shared_ptr_a = (copy_t *)(warp_a_buff + laneId);
	copy_t *lane_shared_ptr_b = (copy_t *)(warp_b_buff + laneId);
	//global ptr of each lane
	copy_t *lane_ptr_a = A + kk + (laneId / 16) * K + (laneId % 16);
	copy_t *lane_ptr_b = B +  (laneId / 16) + (kk + (laneId % 16)) * N;

	
	while(kk < K){
		if(laneId < 16){
			for(int i = 0; i < M; i++){
				*(lane_shared_ptr_a + i * WMMA_N) = *(lane_ptr_a + i * K);
			}
			for(int i = 0; i < N; i++){
				*(lane_shared_ptr_b + i * WMMA_N) = *(lane_ptr_b + i);
			}
		}
		wmma::load_matrix_sync(A_frag, warp_a_buff, 16);
		wmma::load_matrix_sync(B_frag, warp_b_buff, 16);
		wmma::mma_sync(C_frag, A_frag, B_frag, C_frag);


		kk += kk_per_it;
		lane_ptr_a += kk_per_it;
		lane_ptr_b += kk_per_it * N;
	}
	wmma::store_matrix_sync(c_buff + warpId * 256, C_frag, 16, wmma::mem_row_major);
	__syncthreads();//必要

	//naive block level reduction
	if(warpId == 0){
		wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> fetch_c_frag;
		wmma::fill_fragment(A_frag, 0.0f);
		wmma::fill_fragment(B_frag, 0.0f);

		int collect_iter = 1;
#pragma unroll
		while(collect_iter < WARPS_PER_BLOCK){
			wmma::load_matrix_sync(fetch_c_frag, c_buff + collect_iter * 256, 16, wmma::mem_row_major);
#pragma unroll
			for(int i=0;i<C_frag.num_elements;i++)
				C_frag.x[i] = C_frag.x[i] + fetch_c_frag.x[i];
			collect_iter++;			
		}
		wmma::store_matrix_sync(c_buff, C_frag, 16, wmma::mem_row_major);//store block level res
	}
	
	__syncthreads();		
	if(threadIdx.x < M * N){
		int lane_cbuff_index = (threadIdx.x / N) * WMMA_N + (threadIdx.x % N);
		atomicAdd(output + threadIdx.x, c_buff[lane_cbuff_index]);
	}
	
}


#endif