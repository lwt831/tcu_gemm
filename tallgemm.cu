#include <iostream>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cooperative_groups.h>
#include <mma.h>
#include <cublas.h>
#include <cublas_v2.h>
#include "checkerror.h"
#include "helper.h"

#define WARP_SIZE (32)
#define lane_copy_num (8)

using namespace nvcuda;

void half_cublas(half *A_h, half *B_h, float *output_h, int M, int N, int K){
	cublasHandle_t handle;
	cublasCreate(&handle);
	half *A_d;
	half *B_d;
	float *output_d;
	cudaMalloc((void**)&A_d, M*K*sizeof(half));
	cudaMalloc((void**)&B_d, N*K*sizeof(half));
	cudaMalloc((void**)&output_d, M*N*sizeof(float));
	
	cudaMemcpy(A_d, A_h, M*K*sizeof(half), cudaMemcpyHostToDevice);
	cudaMemcpy(B_d, B_h, N*K*sizeof(half), cudaMemcpyHostToDevice);
	
	float one = 1;
	float zero = 0;
	cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);//tensor core on(looks like only available for M, N == 8, 16, )

	cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
				 M, N, K,
				 &one, A_d, CUDA_R_16F, M, B_d, CUDA_R_16F, K,
				 &zero, output_d, CUDA_R_32F, M, CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
	cudaDeviceSynchronize();

	cudaMemcpy(output_h, output_d, M*N*sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(A_d);
	cudaFree(B_d);
	cudaFree(output_d);

}

//先用模板，之后改成常数
template<int BLOCKS_PER_GRID, int WARPS_PER_BLOCK, int WMMA_M, int WMMA_N, int WMMA_K>
__global__ void compute_hgemm_kernel(half *A, half *B, float *output, float *global_block_res, int M, int N, int K){

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
	unsigned int kk = (WARPS_PER_BLOCK * blockId + warpId) * WMMA_K * WMMA_M / M;
	const unsigned int kk_per_it = BLOCKS_PER_GRID * WARPS_PER_BLOCK * WMMA_K * WMMA_M / M;
	//start point at shared mem of each warp
	const half *warp_a_buff = a_buff + (warpId * WMMA_M * WMMA_K);
	const half *warp_b_buff = b_buff + (warpId * WMMA_N * WMMA_K);
	//shared mem ptr of each lane
	copy_t *lane_shared_ptr_a = ((copy_t *)warp_a_buff) + (((laneId%8)/2)*8 + laneId/lane_copy_num*2 + laneId%2);
	copy_t *lane_shared_ptr_b = ((copy_t *)warp_b_buff) + (((laneId%8)/2)*8 + laneId/lane_copy_num*2 + laneId%2);
	//global ptr of each lane
	copy_t *lane_ptr_a = (copy_t *)(A + kk*M + (laneId/8)*K) + laneId%8;
	copy_t *lane_ptr_b = (copy_t *)(B + kk*N + (laneId/8)*K) + laneId%8;
#pragma unroll
	while(kk < K){ //if k = 2^20, on a grid with 80 * 8 = 640 warps, it'll be k/(WMMA_K*4)/640 = 26 iterations/warp
		*lane_shared_ptr_a = *lane_ptr_a;
		*lane_shared_ptr_b = *lane_ptr_b;

		wmma::load_matrix_sync(A_frag, warp_a_buff, 16);
		wmma::load_matrix_sync(B_frag, warp_b_buff, 16);
		wmma::mma_sync(C_frag, A_frag, B_frag, C_frag);

		kk += kk_per_it;
		lane_ptr_a += kk_per_it * M / lane_copy_num;//move global ptr
		lane_ptr_b += kk_per_it * N / lane_copy_num;
	}
	wmma::store_matrix_sync(c_buff + warpId * 256, C_frag, 16, wmma::mem_row_major);
	

	//naive block level reduction
	if(warpId == 0){
		wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> fetch_c_frag;
		wmma::fill_fragment(A_frag, 0.0f);
		wmma::fill_fragment(B_frag, 0.0f);


		int collect_iter = 1;
#pragma unroll
		while(collect_iter < WARPS_PER_BLOCK){
			wmma::load_matrix_sync(fetch_c_frag, c_buff + collect_iter*256, 16, wmma::mem_row_major);
			//wmma::mma_sync(C_frag, A_frag, B_frag, fetch_c_frag);
#pragma unroll
			for(int i=0;i<C_frag.num_elements;i++)
				C_frag.x[i] = C_frag.x[i] + fetch_c_frag.x[i];
			collect_iter++;
		}
		wmma::store_matrix_sync(global_block_res + blockId*256, C_frag, 16, wmma::mem_row_major);//store block level res
		if(blockId == 0)
			for(int i=0;i<C_frag.num_elements;i++)
				printf("%3.2f, ", C_frag.x[i]);//////////////////////////////
	}
	//__syncthreads();
	cooperative_groups::this_grid().sync();

	//naive grid level reduction
	if(blockId == 0){
		if(warpId == 0){
			wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> fetch_c_frag;
			int global_collect_iter = 1;
			while(global_collect_iter < BLOCKS_PER_GRID){
				wmma::load_matrix_sync(fetch_c_frag, global_block_res + global_collect_iter*256, 16, wmma::mem_row_major);
				//wmma::mma_sync(C_frag, A_frag, B_frag, fetch_c_frag);
#pragma unroll
				for(int i=0;i<C_frag.num_elements;i++)
					C_frag.x[i] = C_frag.x[i] + fetch_c_frag.x[i];

				global_collect_iter++;
			}
			wmma::store_matrix_sync(c_buff, C_frag, 16, wmma::mem_row_major);

			if(laneId < 16){//16 threads to reduce 4*16 elements to 16 elements
				int lane_cbuff_index = (laneId/4)*16 + laneId%4;
				for(int i=1;i<4;i++)
					c_buff[lane_cbuff_index] += c_buff[lane_cbuff_index+i*68];//68 = 4*16+4
				output[laneId] = c_buff[lane_cbuff_index];
			}
		}	
	}
	
}

template<int BLOCKS_PER_GRID, int WARPS_PER_BLOCK>
void compute_hgemm(half *A_h, half *B_h, float *output_h, int M, int N, int K){

	half *A_d;
	half *B_d;
	float *output_d;
	float *global_block_res;//buffer for storing block level result

	cudaMalloc((void**)&A_d, M*K*sizeof(half));
	cudaMalloc((void**)&B_d, N*K*sizeof(half));
	cudaMalloc((void**)&output_d, M*N*sizeof(float));
	cudaMalloc((void**)&global_block_res, BLOCKS_PER_GRID*256*sizeof(float));
	
	cudaMemcpy(A_d, A_h, M*K*sizeof(half), cudaMemcpyHostToDevice);
	cudaMemcpy(B_d, B_h, N*K*sizeof(half), cudaMemcpyHostToDevice);

	checkKernelErrors( (compute_hgemm_kernel<BLOCKS_PER_GRID, WARPS_PER_BLOCK, 16, 16, 16><<<BLOCKS_PER_GRID, WARPS_PER_BLOCK*WARP_SIZE>>>(A_d, B_d, output_d, global_block_res, 4, 4, K)) );
	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaMemcpy(output_h, output_d, M*N*sizeof(float), cudaMemcpyDeviceToHost));
	cudaFree(A_d);
	cudaFree(B_d);
	cudaFree(output_d);
	cudaFree(global_block_res);
}

int main(){
	const int M = 4;
	const int N = 4;
	const int K = 1<<15;

	half *A = new half[M*K];
	float *A_f = new float[M*K];
	half *B = new half[N*K];
	float *B_f = new float[N*K];
	float *output = new float[M*N];

	init_input(A, A_f, M*K);
	init_input(B, B_f, N*K);
	
	//half_cublas(A, B, output, M, N, K);
	compute_hgemm<80, 8>(A, B, output, M, N, K);
	

	//print_matrix(output, M, N);

	delete []A;
	delete []A_f;
	delete []B;
	delete []B_f;
	delete []output;
	return 0;
}
