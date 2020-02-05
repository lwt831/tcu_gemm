#ifndef HELPER_H
#define HELPER_H

#include <time.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

void init_input(half *input_half, float *input_float,size_t size){
	srand((int)time(0));
	for(int i=0;i<size;i++){
	  input_float[i] = (float)(rand() % 3);
	  //input_half[i] = __float2half(((float)(input_float[i])-1.0f+0.25f));
	  input_half[i] = __float2half(((float)(input_float[i])));
	}
}

void init_input_const(half *input_half, float *input_float,size_t size, float val){
	for(int i=0;i<size;i++){
	  input_float[i] = val;
	  //input_half[i] = __float2half(((float)(input_float[i])-1.0f+0.25f));
	  input_half[i] = __float2half(val);
	}
}

void print_matrix(float *mat, size_t M, size_t N){
	printf("\n");	
	for(int m=0;m<M;m++){
		for(int n=0;n<N;n++)
			printf("%f, ", mat[m*M+n]);
		printf("\n");
	}
}

void print_matrix(half *mat, size_t M, size_t N){
	printf("\n");	
	for(int m=0;m<M;m++){
		for(int n=0;n<N;n++)
			printf("%f, ", (float)mat[m*M+n]);
		printf("\n");
	}
}

__device__ void print_matrix_device(half *mat, size_t M, size_t N){
	if(blockIdx.x == 0 && threadIdx.x == 0)
		for(int i=0;i<M;i++){
			for(int j=0;j<N;j++)
				printf("%5.2f, ", (float)mat[i*16+j]);
			printf("\n");
		}
}

__device__ void print_matrix_device(float *mat, size_t M, size_t N){
	if(blockIdx.x == 0 && threadIdx.x == 0)
		for(int i=0;i<M;i++){
			for(int j=0;j<N;j++)
				printf("%5.2f, ", mat[i*16+j]);
			printf("\n");
		}
}


struct GpuTimer
{
  cudaEvent_t start;
  cudaEvent_t stop;

  GpuTimer()
  {
    // create events 
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
  }

  ~GpuTimer()
  {
    // delete events 
    cudaEventDestroy(start);
	cudaEventDestroy(stop);
  }

  void Start()
  {
    // start event
    cudaEventRecord(start,0);
  }

  void Stop()
  {
    // stop event
    cudaEventRecord(stop,0);
  }

  float Elapsed()
  {
    // elapsed time 
	cudaEventSynchronize(stop);
	float elapsed;
    cudaEventElapsedTime(&elapsed, start, stop); 
    return elapsed;
  }
};


#endif
