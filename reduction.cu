
/*
    Reduction summation algorithm (with sequential addressing)
    made by: Carrick McClain

    Sources:
        http://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/projects/reduction/doc/reduction.pdf
        some guidance from https://stackoverflow.com
*/

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdlib>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string.h>
#include <assert.h>

#define NUM_BLOCKS 2
#define BLOCK_WIDTH 8
#define BLOCK_SIZE (BLOCK_WIDTH * BLOCK_WIDTH)
#define NUM_FLOATS 100

using namespace std;

inline void gpu_handle_error( cudaError_t err, const char* file, int line, int abort = 1 )
{
	if (err != cudaSuccess)
	{
		fprintf (stderr, "gpu error %s, %s, %d\n", cudaGetErrorString (err), file, line);
		if (abort)
			exit (EXIT_FAILURE);
	}
}
#define gpu_err_chk(e) {gpu_handle_error( e, __FILE__, __LINE__ );}



__global__ void reduction_add (float* X, float* Y)
{
    __shared__ float XY[NUM_FLOATS];

    unsigned int tx = threadIdx.x;
    unsigned int i = blockIdx.x * (2 * blockDim.x) + threadIdx.x;
    XY[tx] = X[i] + X[i + blockDim.x];
    __syncthreads();

    for (unsigned int stride = blockDim.x/2; stride > 0; stride >>= 2)
    {
        if (tx < stride)
            XY[tx] += XY[tx + stride];
        __syncthreads();
    }
    if (tx == 0)
        Y[blockIdx.x] = XY[0];
}



int main (int argc, char** argv)
{
    cudaError_t err;
    int idx = 0;
    int sum = 0;
    char chars[11];
    float* h_input_data = (float*)malloc (NUM_FLOATS * sizeof(float));
    float* h_output_data = (float*)malloc (NUM_FLOATS * sizeof(float));
    float* d_input_data;
    float* d_output_data;
    ifstream infile;
    
    //get data from floats.csv
    infile.open("floats.csv", ifstream::in);
    if (infile.is_open())
    {
        while (infile.good())
        {
            infile.getline(chars, 256, ',');
            h_input_data[idx] = (float)(strtod(chars, NULL));
            idx++;
        }
        infile.close();
    }
    else cout << "Error opening file";

    assert ((sizeof(h_input_data) / sizeof(float)) == 100);

    err = cudaMalloc ((void**) &d_input_data, NUM_FLOATS * sizeof(float));
    gpu_err_chk(err);
    err = cudaMalloc ((void**) &d_output_data, NUM_FLOATS * sizeof(float));
    gpu_err_chk(err);
    err = cudaMemcpy (d_input_data, h_input_data,
                      NUM_FLOATS * sizeof(float), cudaMemcpyHostToDevice);
    gpu_err_chk(err);

    dim3 dimGrid (NUM_BLOCKS);
    dim3 dimBlock (BLOCK_SIZE);
    reduction_add<<<dimGrid, dimBlock>>> (d_input_data, d_output_data);
    err = cudaGetLastError();
    gpu_err_chk(err);

    err = cudaMemcpy(   h_output_data, d_output_data,
                        NUM_FLOATS * sizeof(float),
                        cudaMemcpyDeviceToHost  );
    gpu_err_chk(err);

    idx = 0;
    while (h_output_data != NULL)
    {
        sum += h_output_data[idx];
        idx++;
    }

    cout << "Sum of floats: " << sum;

    
    return 0;
}