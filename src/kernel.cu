#include <stdio.h>

#define BLOCK_SIZE 16

__global__ void matAdd(unsigned int num_elements, const int a, const int b, float* result) {

         int i = threadIdx.x + blockIdx.x * blockDim.x;
        float num = 0;
        if(i < num_elements)
        {
        num = (i * (float)(b-a)/num_elements) + a;
        result[i] = (2*num); //CHANGE ME Example: 2*num will result in the program integrating "2x"
        }

}


__global__ void optimizedReduction(float *out, float *in, unsigned size)
{
    /********************************************************************
    Load a segment of the input vector into shared memory
    Traverse the reduction tree
    Write the computed sum to the output vector at the correct index
    ********************************************************************/
    
    // OPTIMIZED REDUCTION IMPLEMENTATION
    __shared__ float partial[512*2];
    unsigned int x = threadIdx.x;
    unsigned int start = 2*blockIdx.x*blockDim.x;
    partial[x] = in[start + x];
    partial[blockDim.x+x] = in[start+blockDim.x+x];

    for(unsigned int i = 1; i <= blockDim.x; i *= 2)
    {
      __syncthreads();
      if(x < (((2*512)/(2*i))))
      {
        partial[x] += partial[x + ((((2*512)/(2*i))))];
      }
    }
    __syncthreads();
    out[blockIdx.x] = partial[0]; 
}

void basicMatAdd(unsigned int num_elements, const int a, const int b, float* result, float* out, unsigned int out_elements)
{
    // Initialize thread block and kernel grid dimensions ---------------------

    
	
    /*************************************************************************/
    //INSERT CODE HERE
    dim3 dim_grid((num_elements-1)/16 + 1,(num_elements-1)/16+1,1);
    dim3 dim_block(BLOCK_SIZE,BLOCK_SIZE,1);
    /*************************************************************************/
    
	// Invoke CUDA kernel -----------------------------------------------------

    /*************************************************************************/
    //INSERT CODE HERE
    matAdd<<<dim_grid,dim_block>>>(num_elements,a,b,result);
    
    dim_block.x = 512; dim_block.y = dim_block.z = 1;
    dim_grid.x = out_elements; dim_grid.y = dim_grid.z = 1;
    optimizedReduction<<<dim_grid, dim_block>>>(out, result, num_elements);
    /*************************************************************************/

}

