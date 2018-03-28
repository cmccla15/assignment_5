
/*
    Reduction summation algorithm (with sequential addressing)
    made by: Carrick McClain

    Sources:
        http://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/projects/reduction/doc/reduction.pdf
*/

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

#define BLOCK_SIZE 1024

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
    extern __shared__ float XY[];

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
    if (ty == 0)
        Y[blockIdx.x] = XY[0];
}



int main (int argc, char** argv)
{
    if (argc < 2)
    {
        cerr << "expected args: input file name" << endl;
        exit (-1);
    }

    cudaError_t err;
    int idx = 0;
    int numFloats = 100;
    char chars[11];
    float* h_input_data = malloc (numFloats * sizeof(float));
    float* h_output_data = malloc (numFloats * sizeof(float));
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
            h_input_data[idx] = (float)(chars);
            idx++;
        }
        infile.close();
    }
    else cout << "Error opening file";

    err = cudaMalloc ((void**) &d_input_data, h_input_data, numFloats * sizeof(float));
    gpu_err_chk(err);
    err = cudaMalloc ((void**) &d_output_data, h_output_data, numFloats * sizeof(float));
    gpu_err_chk(err);
    err = cudaMemcpy (d_input_data, h_input_data,
                      numFloats * sizeof(float), cudaMemcpyHostToDevice);
    gpu_err_chk(err);

    dim3 dimGrid ();
    dim3 dimBlock ();
    reduction_add<<<>>>();


    return 0;
}