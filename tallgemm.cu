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
#include "helper.cuh"
#include "gemm_kernel.cuh" 
#include "gemm_fastcopy.cuh" 


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


	cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
				 M, N, K,
				 &one, A_d, CUDA_R_16F, K, B_d, CUDA_R_16F, K,
				 &zero, output_d, CUDA_R_32F, M, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
	cudaDeviceSynchronize();

	cudaMemcpy(output_h, output_d, M*N*sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(A_d);
	cudaFree(B_d);
	cudaFree(output_d);

}


template<int BLOCKS_PER_GRID, int WARPS_PER_BLOCK>
void compute_hgemm(half *A_h, half *B_h, float *output_h, int M, int N, int K){

	half *A_d;
	half *B_d;
	float *output_d;
	float *global_block_res;//buffer for storing block level result

	checkCudaErrors(cudaMalloc((void**)&A_d, M*K*sizeof(half)));
	checkCudaErrors(cudaMalloc((void**)&B_d, N*K*sizeof(half)));
	checkCudaErrors(cudaMalloc((void**)&output_d, M*N*sizeof(float)));
	//checkCudaErrors(cudaMalloc((void**)&global_block_res, BLOCKS_PER_GRID*256*sizeof(float)));
	
	checkCudaErrors(cudaMemcpy(A_d, A_h, M*K*sizeof(half), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(B_d, B_h, N*K*sizeof(half), cudaMemcpyHostToDevice));

	int tail_k = K - (K % ((16 / M) * 16)); 
	std::cout<<"tail_k"<<tail_k<<std::endl;
	if(M == 3)
		checkKernelErrors( (compute_hgemm_kernel3_slow<BLOCKS_PER_GRID, WARPS_PER_BLOCK, 16, 16, 16><<<BLOCKS_PER_GRID, WARPS_PER_BLOCK*WARP_SIZE>>>(A_d, B_d, output_d, 3, 3, K, tail_k)) );
		//checkKernelErrors( (compute_hgemm_kernel3<BLOCKS_PER_GRID, WARPS_PER_BLOCK, 16, 16, 16><<<BLOCKS_PER_GRID, WARPS_PER_BLOCK*WARP_SIZE>>>(A_d, B_d, output_d, 3, 3, K, tail_k)) );
	if(M == 4)
		checkKernelErrors( (compute_hgemm_kernel4_slow<BLOCKS_PER_GRID, WARPS_PER_BLOCK, 16, 16, 16><<<BLOCKS_PER_GRID, WARPS_PER_BLOCK*WARP_SIZE>>>(A_d, B_d, output_d, 4, 4, K, tail_k)) );
		//checkKernelErrors( (compute_hgemm_kernel4<BLOCKS_PER_GRID, WARPS_PER_BLOCK, 16, 16, 16><<<BLOCKS_PER_GRID, WARPS_PER_BLOCK*WARP_SIZE>>>(A_d, B_d, output_d, 4, 4, K, tail_k)) );
	if(M == 8)
		checkKernelErrors( (compute_hgemm_kernel8<BLOCKS_PER_GRID, WARPS_PER_BLOCK, 16, 16, 16><<<BLOCKS_PER_GRID, WARPS_PER_BLOCK*WARP_SIZE>>>(A_d, B_d, output_d, 8, 8, K)) );
	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaMemcpy(output_h, output_d, M*N*sizeof(float), cudaMemcpyDeviceToHost));
	cudaFree(A_d);
	cudaFree(B_d);
	cudaFree(output_d);
	//cudaFree(global_block_res);
}

int main(){
	const int M = 3;
	const int N = 3;
	const int K = 6400;

	half *A = new half[M*K];
	float *A_f = new float[M*K];
	half *B = new half[N*K];
	float *B_f = new float[N*K];
	float *output = new float[M*N];
	float *output_cublas = new float[M*N];

	//init_input_const(A, A_f, M*K/2, 1.0f);
	init_input(A, A_f, M*K);
	init_input(B, B_f, N*K);

	std::cout<<"result with my kernel:"<<std::endl;
	compute_hgemm<64, 8>(A, B, output, M, N, K);//num of warps should >= 4	
	print_matrix(output, M, N);

	std::cout<<"result with cublas:"<<std::endl;
	half_cublas(A, B, output_cublas, M, N, K);
	print_matrix(output_cublas, M, N);

	delete []A;
	delete []A_f;
	delete []B;
	delete []B_f;
	delete []output;
	return 0;
}
