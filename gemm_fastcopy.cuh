#ifndef GEMM_FASTCOPY_H
#define GEMM_FASTCOPY_H

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


//gemm kernel for M = 4
template<int BLOCKS_PER_GRID, int WARPS_PER_BLOCK, int WMMA_M, int WMMA_N, int WMMA_K>
__global__ void compute_hgemm_kernel4(half *A, half *B, float *output, int M, int N, int K, int tail_k){

	const unsigned int blockId = blockIdx.x;
	const unsigned int warpId = threadIdx.x / WARP_SIZE;
	const unsigned int laneId = threadIdx.x % WARP_SIZE;

	//const unsigned int all_iters = K/WMMA_K;
	typedef int4 copy_t;

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
	copy_t *lane_shared_ptr_a = ((copy_t *)warp_a_buff) + (((laneId%8)/2)*8 + laneId/lane_copy_num*2 + laneId%2);
	copy_t *lane_shared_ptr_b = ((copy_t *)warp_b_buff) + (((laneId%8)/2)*8 + laneId/lane_copy_num*2 + laneId%2);
	//global ptr of each lane
	copy_t *lane_ptr_a = (copy_t *)(A + kk + (laneId/8)*K) + laneId%8;
	copy_t *lane_ptr_b = (copy_t *)(B + kk + (laneId/8)*K) + laneId%8;

	/*
	if(blockId == 0 && threadIdx.x < 256){
		if((threadIdx.x % 64) < K - tail_k){
			half *lane_ptr_taila = A + (threadIdx.x/64)*K + threadIdx.x % 64 + tail_k;
			half *lane_ptr_tailb = B + (threadIdx.x/64)*K + threadIdx.x % 64 + tail_k;
			//tail_buffer[((threadIdx.x%64)/16)*64 + (threadIdx.x/64)*16 + threadIdx.x%16] =  *lane_ptr_taila;
			//tail_buffer[((threadIdx.x%64)/16)*64 + (threadIdx.x/64)*16 + threadIdx.x%16 + 256] =  *lane_ptr_tailb;
		} 
		
	}*/

	while(kk < tail_k){ //if k = 2^20, on a grid with 80 * 8 = 640 warps, it'll be k/(WMMA_K*4)/640 = 26 iterations/warp

		// if(kk == tail_k){ //handle the tail(out of n*4*16 on K dimension)
		// 	wmma::load_matrix_sync(A_frag, tail_buffer, 16);
		// 	wmma::load_matrix_sync(B_frag, tail_buffer+256, 16);
		// 	wmma::mma_sync(C_frag, A_frag, B_frag, C_frag);
		// 	break;
		// }

		*lane_shared_ptr_a = *lane_ptr_a;
		*lane_shared_ptr_b = *lane_ptr_b;

		wmma::load_matrix_sync(A_frag, warp_a_buff, 16);
		wmma::load_matrix_sync(B_frag, warp_b_buff, 16);
		wmma::mma_sync(C_frag, A_frag, B_frag, C_frag);

		kk += kk_per_it;
		lane_ptr_a += kk_per_it / lane_copy_num;//move global ptr
		lane_ptr_b += kk_per_it / lane_copy_num;
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
		//wmma::store_matrix_sync(global_block_res + blockId*256, C_frag, 16, wmma::mem_row_major);
	}
	
	__syncthreads();		
	if(threadIdx.x < 64){
		int lane_cbuff_index = ((threadIdx.x%16)/4)*16 + threadIdx.x%4 + (threadIdx.x/16)*68;
		atomicAdd(output+ ((threadIdx.x%16)/4)*4 + (threadIdx.x%16)%4, c_buff[lane_cbuff_index]);
	}
	
/*
	cooperative_groups::this_grid().sync();
	//naive grid level reduction
	if(blockId == 0){
		if(warpId == 0){
			wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> fetch_c_frag;
			int global_collect_iter = 1;
			while(global_collect_iter < BLOCKS_PER_GRID){
				wmma::load_matrix_sync(fetch_c_frag, global_block_res + global_collect_iter*256, 16, wmma::mem_row_major);
#pragma unroll
				for(int i=0;i<C_frag.num_elements;i++)
					C_frag.x[i] = C_frag.x[i] + fetch_c_frag.x[i];
				global_collect_iter++;
			}
			wmma::store_matrix_sync(c_buff, C_frag, 16, wmma::mem_row_major);

			int lane_cbuff_index = (laneId/4)*16 + laneId%4;
			if(laneId < 16){//16 threads to reduce 4*16 elements to 16 elements				
				for(int i=1;i<4;i++)
					c_buff[lane_cbuff_index] += c_buff[lane_cbuff_index+i*68];//68 = 4*16+4
				output[laneId] = c_buff[lane_cbuff_index];
			}				
		}	
	}
*/	
}


//gemm kernel for dimension = 3
template<int BLOCKS_PER_GRID, int WARPS_PER_BLOCK, int WMMA_M, int WMMA_N, int WMMA_K>
__global__ void compute_hgemm_kernel3(half *A, half *B, float *output, int M, int N, int K){

	const unsigned int blockId = blockIdx.x;
	const unsigned int warpId = threadIdx.x / WARP_SIZE;
	const unsigned int laneId = threadIdx.x % WARP_SIZE;

	//const unsigned int all_iters = K/WMMA_K;
	typedef int4 copy_t;

	__shared__ half a_buff[WARPS_PER_BLOCK*16*16];
	__shared__ half b_buff[WARPS_PER_BLOCK*16*16];
	__shared__ float c_buff[WARPS_PER_BLOCK*16*16];

	wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> A_frag;
	wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> B_frag;
	wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> C_frag;
	wmma::fill_fragment(C_frag, 0.0f);

	//loop over K 
	unsigned int kk = (WARPS_PER_BLOCK * blockId + warpId) * WMMA_K * (WMMA_M / M);//80
	const unsigned int kk_per_it = BLOCKS_PER_GRID * WARPS_PER_BLOCK * WMMA_K * WMMA_M / M;
	//start point at shared mem of each warp
	const half *warp_a_buff = a_buff + (warpId * 256);
	const half *warp_b_buff = b_buff + (warpId * 256);
	//shared mem ptr of each lane
	copy_t *lane_shared_ptr_a = ((copy_t *)warp_a_buff) + (((laneId%10)/2)*6 + (laneId/10)*2 + laneId%2);//mod
	copy_t *lane_shared_ptr_b = ((copy_t *)warp_b_buff) + (((laneId%10)/2)*6 + (laneId/10)*2 + laneId%2);
	//global ptr of each lane
	copy_t *lane_ptr_a = (copy_t *)(A + kk + (laneId/10)*K) + laneId%10;//mod
	copy_t *lane_ptr_b = (copy_t *)(B + kk + (laneId/10)*K) + laneId%10;

    while(kk < K){ //if k = 2^20, on a grid with 80 * 8 = 640 warps, it'll be k/(WMMA_K*4)/640 = 26 iterations/warp
        if(laneId < 30){  
		    *lane_shared_ptr_a = *lane_ptr_a;
		    *lane_shared_ptr_b = *lane_ptr_b;
		}

		wmma::load_matrix_sync(A_frag, warp_a_buff, 16);
		wmma::load_matrix_sync(B_frag, warp_b_buff, 16);
		wmma::mma_sync(C_frag, A_frag, B_frag, C_frag);

		kk += kk_per_it;
		lane_ptr_a += kk_per_it / lane_copy_num;//move global ptr
		lane_ptr_b += kk_per_it / lane_copy_num;
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
        atomicAdd(output+ threadIdx.x%9, c_buff[lane_cbuff_index]);
	}
}




//gemm kernel for dimension = 8
template<int BLOCKS_PER_GRID, int WARPS_PER_BLOCK, int WMMA_M, int WMMA_N, int WMMA_K>
__global__ void compute_hgemm_kernel8(half *A, half *B, float *output,int M, int N, int K){

	const unsigned int blockId = blockIdx.x;
	const unsigned int warpId = threadIdx.x / WARP_SIZE;
	const unsigned int laneId = threadIdx.x % WARP_SIZE;

	//const unsigned int all_iters = K/WMMA_K;
	typedef int4 copy_t;

	__shared__ half a_buff[WARPS_PER_BLOCK*16*16];
	__shared__ half b_buff[WARPS_PER_BLOCK*16*16];
	__shared__ float c_buff[WARPS_PER_BLOCK*16*16];

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
	copy_t *lane_shared_ptr_a = ((copy_t *)warp_a_buff) + (((laneId%4)/2)*16 + (laneId/4)*2 + laneId%2);//mod
	copy_t *lane_shared_ptr_b = ((copy_t *)warp_b_buff) + (((laneId%4)/2)*16 + (laneId/4)*2 + laneId%2);
	//global ptr of each lane
	copy_t *lane_ptr_a = (copy_t *)(A + kk + (laneId/4)*K) + laneId%4;//mod
	copy_t *lane_ptr_b = (copy_t *)(B + kk + (laneId/4)*K) + laneId%4;

    while(kk < K){ //if k = 2^20, on a grid with 80 * 8 = 640 warps, it'll be k/(WMMA_K*4)/640 = 26 iterations/warp
		*lane_shared_ptr_a = *lane_ptr_a;
		*lane_shared_ptr_b = *lane_ptr_b;

		wmma::load_matrix_sync(A_frag, warp_a_buff, 16);
		wmma::load_matrix_sync(B_frag, warp_b_buff, 16);
		wmma::mma_sync(C_frag, A_frag, B_frag, C_frag);

		kk += kk_per_it;
		lane_ptr_a += kk_per_it / lane_copy_num;//move global ptr
		lane_ptr_b += kk_per_it / lane_copy_num;
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
        //atomicAdd(output+ ((threadIdx.x%9)/3)*3 + (threadIdx.x%9)%3, c_buff[lane_cbuff_index]);
        atomicAdd(output+ threadIdx.x%64, c_buff[lane_cbuff_index]);
	}
}

#endif