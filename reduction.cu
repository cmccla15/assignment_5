
/*
    Reduction summation algorithm (with sequential addressing)
    made by: Carrick McClain

    Sources:
        http://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/projects/reduction/doc/reduction.pdf
*/

#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

#define BLOCK_SIZE 16

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



    return 0;
}