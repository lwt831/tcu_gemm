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

template<int BLOCKS_PER_GRID, int WARPS_PER_BLOCK, int WMMA_M, int WMMA_N, int WMMA_K>
__global__ void compute_hgemm_kernel4_slow(half *A, half *B, float *output, int M, int N, int K, int tail_k){

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
	unsigned int kk = (WARPS_PER_BLOCK * blockId + warpId) * WMMA_K * WMMA_M / M;
	const unsigned int kk_per_it = BLOCKS_PER_GRID * WARPS_PER_BLOCK * WMMA_K * WMMA_M / M;
	//start point at shared mem of each warp
	const half *warp_a_buff = a_buff + (warpId * 256);
	const half *warp_b_buff = b_buff + (warpId * 256);
	//shared mem ptr of each lane
	copy_t *lane_shared_ptr_a = (copy_t *)(warp_a_buff + (laneId/16)*64 + laneId%16);
	copy_t *lane_shared_ptr_b = (copy_t *)(warp_b_buff + (laneId/16)*64 + laneId%16);
	//global ptr of each lane
	copy_t *lane_ptr_a = A + kk + laneId;
	copy_t *lane_ptr_b = B + kk + laneId;

	
	if(blockId == 0 && threadIdx.x < 256){
		if((threadIdx.x % 64) < K - tail_k){
			half *lane_ptr_taila = A + (threadIdx.x/64)*K + threadIdx.x % 64 + tail_k;
			half *lane_ptr_tailb = B + (threadIdx.x/64)*K + threadIdx.x % 64 + tail_k;
			tail_buffer[((threadIdx.x%64)/16)*64 + (threadIdx.x/64)*16 + threadIdx.x%16] =  *lane_ptr_taila;
			tail_buffer[((threadIdx.x%64)/16)*64 + (threadIdx.x/64)*16 + threadIdx.x%16 + 256] =  *lane_ptr_tailb;
		}
		__syncthreads();//必要
		if(warpId == 0){ 
			wmma::load_matrix_sync(A_frag, tail_buffer, 16);
			wmma::load_matrix_sync(B_frag, tail_buffer + 256, 16);
			wmma::mma_sync(C_frag, A_frag, B_frag, C_frag);
		}
	}

	while(kk < tail_k){ //if k = 2^20, on a grid with 80 * 8 = 640 warps, it'll be k/(WMMA_K*4)/640 = 26 iterations/warp
		for(int i = 0; i < M; i++){
			*lane_shared_ptr_a = *lane_ptr_a;
			*lane_shared_ptr_b = *lane_ptr_b;

			lane_shared_ptr_a += 16;
			lane_shared_ptr_b += 16;
			lane_ptr_a += K;
			lane_ptr_b += K;
		}

		lane_shared_ptr_a += 64;
		lane_shared_ptr_b += 64;
		lane_ptr_a = A + kk + laneId + 32;
		lane_ptr_b = B + kk + laneId + 32;

		for(int i = 0; i < M; i++){
			*lane_shared_ptr_a = *lane_ptr_a;
			*lane_shared_ptr_b = *lane_ptr_b;

			lane_shared_ptr_a += 16;
			lane_shared_ptr_b += 16;
			lane_ptr_a += K;
			lane_ptr_b += K;
		}

		wmma::load_matrix_sync(A_frag, warp_a_buff, 16);
		wmma::load_matrix_sync(B_frag, warp_b_buff, 16);
		wmma::mma_sync(C_frag, A_frag, B_frag, C_frag);


		kk += kk_per_it;
		lane_ptr_a = A + kk + laneId;
		lane_ptr_b = B + kk + laneId;
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
			wmma::load_matrix_sync(fetch_c_frag, c_buff + collect_iter*256, 16, wmma::mem_row_major);
#pragma unroll
			for(int i=0;i<C_frag.num_elements;i++)
				C_frag.x[i] = C_frag.x[i] + fetch_c_frag.x[i];
			collect_iter++;			
		}
		wmma::store_matrix_sync(c_buff, C_frag, 16, wmma::mem_row_major);//store block level res
	}
	
	__syncthreads();		
	if(threadIdx.x < 64){
		int lane_cbuff_index = ((threadIdx.x%16)/4)*16 + threadIdx.x%4 + (threadIdx.x/16)*68;
		atomicAdd(output + (threadIdx.x%16), c_buff[lane_cbuff_index]);
	}
	
}



//gemm kernel for dimension = 3
template<int BLOCKS_PER_GRID, int WARPS_PER_BLOCK, int WMMA_M, int WMMA_N, int WMMA_K>
__global__ void compute_hgemm_kernel3_slow(half *A, half *B, float *output, int M, int N, int K, int tail_k){

	const unsigned int blockId = blockIdx.x;
	const unsigned int warpId = threadIdx.x / WARP_SIZE;
	const unsigned int laneId = threadIdx.x % WARP_SIZE;

	//const unsigned int all_iters = K/WMMA_K;
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
	unsigned int kk = (WARPS_PER_BLOCK * blockId + warpId) * WMMA_K * (WMMA_M / M);
	const unsigned int kk_per_it = BLOCKS_PER_GRID * WARPS_PER_BLOCK * WMMA_K * (WMMA_M / M);
	//start point at shared mem of each warp
	const half *warp_a_buff = a_buff + (warpId * 256);
	const half *warp_b_buff = b_buff + (warpId * 256);
	//shared mem ptr of each lane
	copy_t *lane_shared_ptr_a = (copy_t *)(warp_a_buff + (laneId/16)*48 + laneId%16);
	copy_t *lane_shared_ptr_b = (copy_t *)(warp_b_buff + (laneId/16)*48 + laneId%16);
	//global ptr of each lane
	copy_t *lane_ptr_a = A + kk + laneId;
	copy_t *lane_ptr_b = B + kk + laneId;

	if(blockId == 0){
		if((threadIdx.x % 80 ) < K - tail_k && threadIdx.x < 240){
			half *lane_ptr_taila = A + (threadIdx.x/80)*K + threadIdx.x % 80 + tail_k;
			half *lane_ptr_tailb = B + (threadIdx.x/80)*K + threadIdx.x % 80 + tail_k;
			tail_buffer[((threadIdx.x%80)/16)*48 + (threadIdx.x/80)*16 + threadIdx.x%16] =  *lane_ptr_taila;
			tail_buffer[((threadIdx.x%80)/16)*48 + (threadIdx.x/80)*16 + threadIdx.x%16 + 256] =  *lane_ptr_tailb;
		}
		__syncthreads();//必要
		if(warpId == 0){ 
			wmma::load_matrix_sync(A_frag, tail_buffer, 16);
			wmma::load_matrix_sync(B_frag, tail_buffer + 256, 16);
			wmma::mma_sync(C_frag, A_frag, B_frag, C_frag);
		}
	}

    while(kk < tail_k){
		for(int i = 0; i < M; i++){
		    *lane_shared_ptr_a = *lane_ptr_a;
			*lane_shared_ptr_b = *lane_ptr_b;
			
			lane_shared_ptr_a += 16;
			lane_shared_ptr_b += 16;
			lane_ptr_a += K;
			lane_ptr_b += K;
		}		
		lane_shared_ptr_a += 48;
		lane_shared_ptr_b += 48;
		lane_ptr_a = A + kk + laneId + 32;
		lane_ptr_b = B + kk + laneId + 32;

		for(int i = 0; i < M; i++){
		    *lane_shared_ptr_a = *lane_ptr_a;
			*lane_shared_ptr_b = *lane_ptr_b;
			
			lane_shared_ptr_a += 16;
			lane_shared_ptr_b += 16;
			lane_ptr_a += K;
			lane_ptr_b += K;
		}
		lane_shared_ptr_a += 48;
		lane_shared_ptr_b += 48;
		lane_ptr_a = A + kk + laneId + 64;
		lane_ptr_b = B + kk + laneId + 64;

		for(int i = 0; i < M; i++){
			if(laneId < 16){
				*lane_shared_ptr_a = *lane_ptr_a;
				*lane_shared_ptr_b = *lane_ptr_b;
				
				lane_shared_ptr_a += 16;
				lane_shared_ptr_b += 16;
				lane_ptr_a += K;
				lane_ptr_b += K;
			}
		}

		wmma::load_matrix_sync(A_frag, warp_a_buff, 16);
		wmma::load_matrix_sync(B_frag, warp_b_buff, 16);
		wmma::mma_sync(C_frag, A_frag, B_frag, C_frag);

		kk += kk_per_it;
		lane_ptr_a = A + kk + laneId;
		lane_ptr_b = B + kk + laneId;
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
			wmma::load_matrix_sync(fetch_c_frag, c_buff + collect_iter*256, 16, wmma::mem_row_major);
#pragma unroll
			for(int i=0;i<C_frag.num_elements;i++)
				C_frag.x[i] = C_frag.x[i] + fetch_c_frag.x[i];
			collect_iter++;			
		}
		wmma::store_matrix_sync(c_buff, C_frag, 16, wmma::mem_row_major);//store block level res
	}
	
	__syncthreads();		
	if(threadIdx.x < 45){//mod 45 = 5 * 3 * 3
		int lane_cbuff_index = ((threadIdx.x%9)/3)*16 + threadIdx.x%3 + (threadIdx.x/9)*51;
        //atomicAdd(output+ ((threadIdx.x%9)/3)*3 + (threadIdx.x%9)%3, c_buff[lane_cbuff_index]);
        atomicAdd(output+ threadIdx.x%9, c_buff[lane_cbuff_index]);
	}
}



//gemm kernel for dimension = 8
template<int BLOCKS_PER_GRID, int WARPS_PER_BLOCK, int WMMA_M, int WMMA_N, int WMMA_K>
__global__ void compute_hgemm_kernel8_slow(half *A, half *B, float *output,int M, int N, int K, int tail_k){

	const unsigned int blockId = blockIdx.x;
	const unsigned int warpId = threadIdx.x / WARP_SIZE;
	const unsigned int laneId = threadIdx.x % WARP_SIZE;

	//const unsigned int all_iters = K/WMMA_K;
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
	unsigned int kk = (WARPS_PER_BLOCK * blockId + warpId) * WMMA_K * (WMMA_M / M);//32
	const unsigned int kk_per_it = BLOCKS_PER_GRID * WARPS_PER_BLOCK * WMMA_K * WMMA_M / M;
	//start point at shared mem of each warp
	const half *warp_a_buff = a_buff + (warpId * 256);
	const half *warp_b_buff = b_buff + (warpId * 256);
	//shared mem ptr of each lane
	copy_t *lane_shared_ptr_a = (copy_t *)(warp_a_buff + (laneId/16)*128 + laneId%16);//mod
	copy_t *lane_shared_ptr_b = (copy_t *)(warp_b_buff + (laneId/16)*128 + laneId%16);
	//global ptr of each lane
	copy_t *lane_ptr_a = A + kk + laneId;//mod
	copy_t *lane_ptr_b = B + kk + laneId;

    while(kk < K){ //if k    = 2^20, on a grid with 80 * 8 = 640 warps, it'll be k/(WMMA_K*4)/640 = 26 iterations/warp
		for(int i = 0; i < M; i++){
			*lane_shared_ptr_a = *lane_ptr_a;
			*lane_shared_ptr_b = *lane_ptr_b;

			lane_shared_ptr_a += 16;
			lane_shared_ptr_b += 16;
			lane_ptr_a += K;
			lane_ptr_b += K;
		}


		wmma::load_matrix_sync(A_frag, warp_a_buff, 16);
		wmma::load_matrix_sync(B_frag, warp_b_buff, 16);
		wmma::mma_sync(C_frag, A_frag, B_frag, C_frag);
		//wmma::mma_sync(C_frag, A_frag, B_frag, C_frag);

		kk += kk_per_it;
		lane_ptr_a = A + kk + laneId;//move global ptr
		lane_ptr_b = B + kk + laneId;
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
			wmma::load_matrix_sync(fetch_c_frag, c_buff + collect_iter*256, 16, wmma::mem_row_major);
#pragma unroll
			for(int i=0;i<C_frag.num_elements;i++)
				C_frag.x[i] = C_frag.x[i] + fetch_c_frag.x[i];
			collect_iter++;			
		}
		wmma::store_matrix_sync(c_buff, C_frag, 16, wmma::mem_row_major);//store block level res
	}
	
	__syncthreads();		
	if(threadIdx.x < 128){//mod 128 = 2 * 8 * 8
		int lane_cbuff_index = ((threadIdx.x%64)/8)*16 + threadIdx.x%8 + (threadIdx.x/64)*136;
        atomicAdd(output+ threadIdx.x%64, c_buff[lane_cbuff_index]);
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
__global__ void compute_hgemm_kernel4_pad(half *A, half *B, float *output, int M, int N, int K, int tail_k){

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
	copy_t *lane_ptr_b = B + kk + (laneId / 16) * K + (laneId % 16);

	
	while(kk < tail_k){ //if k = 2^20, on a grid with 80 * 8 = 640 warps, it'll be k/(WMMA_K*4)/640 = 26 iterations/warp
		for(int i = 0; i < 2; i++){
			*(lane_shared_ptr_a + i * 32) = *(lane_ptr_a + i * 2 * K);
			*(lane_shared_ptr_b + i * 32) = *(lane_ptr_b + i * 2 * K);
		}


		wmma::load_matrix_sync(A_frag, warp_a_buff, 16);
		wmma::load_matrix_sync(B_frag, warp_b_buff, 16);
		wmma::mma_sync(C_frag, A_frag, B_frag, C_frag);


		kk += kk_per_it;
		lane_ptr_a += kk_per_it;
		lane_ptr_b += kk_per_it;
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
	if(threadIdx.x < 16){
		int lane_cbuff_index = (threadIdx.x / 4) * 16 + (threadIdx.x % 4);
		atomicAdd(output + threadIdx.x, c_buff[lane_cbuff_index]);
	}
	
}


#endif