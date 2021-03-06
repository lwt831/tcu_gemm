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
#include "test_kernel.cuh"


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
				 N, M, K,
				 &one, B_d, CUDA_R_16F, N, A_d, CUDA_R_16F, K,
				 &zero, output_d, CUDA_R_32F, N, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);//CUBLAS_GEMM_DEFAULT, CUBLAS_GEMM_DEFAULT_TENSOR_OP, CUBLAS_GEMM_ALGO0_TENSOR_OP to CUBLAS_GEMM_ALGO15_TENSOR_OP
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
	float *output_d1;

	checkCudaErrors(cudaMalloc((void**)&A_d, M*K*sizeof(half)));
	checkCudaErrors(cudaMalloc((void**)&B_d, N*K*sizeof(half)));
	checkCudaErrors(cudaMalloc((void**)&output_d, M*N*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&output_d1, M*N*sizeof(float)));

	checkCudaErrors(cudaMemcpy(A_d, A_h, M*K*sizeof(half), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(B_d, B_h, N*K*sizeof(half), cudaMemcpyHostToDevice));

	int tail_k = K - (K % ((16 / M) * 16)); 
	//std::cout<<"tail_k"<<tail_k<<std::endl;
	for(int i = 0; i < 100; i++){
		if(N == 1)
			dense_mv<BLOCKS_PER_GRID, WARPS_PER_BLOCK><<<BLOCKS_PER_GRID, WARPS_PER_BLOCK*WARP_SIZE>>>(A_d, B_d, output_d, M, K);
		if(M == 3)
			checkKernelErrors( (compute_hgemm_kernel3_slow<BLOCKS_PER_GRID, WARPS_PER_BLOCK, 16, 16, 16><<<BLOCKS_PER_GRID, WARPS_PER_BLOCK*WARP_SIZE>>>(A_d, B_d, output_d, 3, 3, K, tail_k)) );
		if(M == 4){
			checkKernelErrors( (compute_hgemm_kernel4_slow<BLOCKS_PER_GRID, WARPS_PER_BLOCK, 16, 16, 16><<<BLOCKS_PER_GRID, WARPS_PER_BLOCK*WARP_SIZE>>>(A_d, B_d, output_d, 4, 4, K, tail_k)) );
			checkKernelErrors( (compute_hgemm_kernel4_pad<BLOCKS_PER_GRID, WARPS_PER_BLOCK, 16, 16, 16><<<BLOCKS_PER_GRID, WARPS_PER_BLOCK*WARP_SIZE>>>(A_d, B_d, output_d1, 4, 4, K, tail_k)) );
		}
		if(M == 8)
			checkKernelErrors( (compute_hgemm_kernel8_slow<BLOCKS_PER_GRID, WARPS_PER_BLOCK, 16, 16, 16><<<BLOCKS_PER_GRID, WARPS_PER_BLOCK*WARP_SIZE>>>(A_d, B_d, output_d, 8, 8, K, tail_k)) );
	}
	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaMemcpy(output_h, output_d, M*N*sizeof(float), cudaMemcpyDeviceToHost));
	cudaFree(A_d);
	cudaFree(B_d);
	cudaFree(output_d);
	//cudaFree(global_block_res);
}

int main(){
	const int M = 4;
	const int N = 4;
	const int K = 80000;

	half *A = new half[M*K];
	float *A_f = new float[M*K];
	half *B = new half[N*K];
	float *B_f = new float[N*K];
	float *output = new float[M*N];
	float *output_cublas = new float[M*N];

	srand((int)time(0));
	init_input(A, A_f, M*K);

	srand((int)time(0) + 1);
	init_input(B, B_f, M*K);

	std::cout<<"result with my kernel:"<<std::endl;
	compute_hgemm<68, 8>(A, B, output, M, N, K);//num of warps should >= 4	
	print_matrix(output, M, N);

	std::cout<<"result with cublas:"<<std::endl;
	half_cublas(A, B, output_cublas, M, N, K);
	print_matrix(output_cublas, M, N);

	// for(int i = 0; i < 100; i++)
	// wmma_once<<<1, 256>>>();
	
	// for(int i = 0; i < 100; i++)
	// 	mmm_once<<<1, 256>>>();



	checkCudaErrors(cudaDeviceSynchronize());


	delete []A;
	delete []A_f;
	delete []B;
	delete []B_f;
	delete []output;
	return 0;
}
