//reduction.cu
//made by Carrick McClain

#define SECTION_SIZE 10
#define BLOCK_SIZE 16

__global__ void reduction_add ()
{
    __shared__ float XY[]

    for (unsigned int stride = 1; stride <= BLOCK_SIZE; stride *= 2)
    {
        int index = (threadIdx.x + 1) * stride * 2 - 1;
        if (index < 2 * BLOCK_SIZE)
            XY[index] += XY[index - stride];
        __syncthreads();
    }
}