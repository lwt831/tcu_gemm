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
	  input_float[i] = (float)(rand() % 100);
	  //input_half[i] = __float2half(((float)(input_float[i])-1.0f+0.25f));
	  input_half[i] = __float2half(((float)(input_float[i]))/100.0f);
	  input_half[i] = __float2half(1.0f);
	}
}

void print_matrix(float *mat, size_t M, size_t N){
	
	for(int m=0;m<M;m++){
		for(int n=0;n<N;n++)
			printf("%f,", mat[m*M+n]);
		printf("\n");
	}
}

#endif
