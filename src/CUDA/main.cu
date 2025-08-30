#include <stdio.h>
#include <stdlib.h>
#include "kernel.cu"
#include "support.h"

int main (int argc, char *argv[])
{

    Timer timer;
    cudaError_t cuda_ret;

    // Initialize host variables ----------------------------------------------

    printf("\nSetting up the problem..."); fflush(stdout);
    startTime(&timer);

    float *A_h, *out_h;
    float *A_d, *out_d;
    unsigned int num_elements = 20000;
    unsigned int out_elements = 0;
    int a = 0;
    int b = 0;
 

    dim3 dim_grid, dim_block;

     if (argc == 3) {
        a = atoi(argv[1]); 
        b = atoi(argv[2]);
    } else {
        printf("\n    Invalid input parameters!\n");
        exit(0);
    }
   

    initVector(&A_h, num_elements);

   // A_h = (float*)malloc(num_elements * sizeof(float));
    //if(A_h == NULL) FATAL("Unable to allocate host");

    out_elements = num_elements / (512<<1);
    if(num_elements % (512<<1)) out_elements++;
    
    out_h = (float*)malloc(out_elements * sizeof(float));
    if(out_h == NULL) FATAL("Unable to allocate host");


    stopTime(&timer); printf("%f s\n", elapsedTime(timer));
   

    // Allocate device variables ----------------------------------------------

   

        //cudaMalloc((void **) &A_d, size);
        
       cuda_ret = cudaMalloc((void**)&A_d, num_elements * sizeof(float));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory"); 

    cuda_ret = cudaMalloc((void**)&out_d, out_elements * sizeof(float));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");

        
        


    printf("Allocating device variables..."); fflush(stdout);
    startTime(&timer);

    /*************************************************************************/
    //INSERT CODE HERE

    /*************************************************************************/
    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Copy host variables to device ------------------------------------------
    printf("Copying data from host to device..."); fflush(stdout);
    startTime(&timer);
	
    /*************************************************************************/
    //INSERT CODE HERE  
    cuda_ret = cudaMemcpy(A_d, A_h, num_elements * sizeof(float),cudaMemcpyHostToDevice);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to the device");

    cuda_ret = cudaMemset(out_d, 0, out_elements * sizeof(float));
    if(cuda_ret != cudaSuccess) FATAL("Unable to set device memory");

    /*************************************************************************/
    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Launch kernel using standard mat-add interface -------------------------
    printf("Launching kernel..."); fflush(stdout);
    startTime(&timer);
    basicMatAdd(num_elements, a, b, A_d, out_d, out_elements);


    cuda_ret = cudaDeviceSynchronize();
    if(cuda_ret != cudaSuccess) printf("Unable to launch kernel");
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Copy device variables to host ------------------------------------------
    printf("Copying data from device to host..."); fflush(stdout);
    startTime(&timer);

    /*************************************************************************/
    //INSERT CODE HERE
    cuda_ret = cudaMemcpy(out_h, out_d, out_elements * sizeof(float),
    cudaMemcpyDeviceToHost);
	if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to host");    /*************************************************************************/
    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Verify correctness -----------------------------------------------------

    printf("Verifying results...\n"); fflush(stdout);

    startTime(&timer);

   for(int i=1; i<out_elements; i++) {
		out_h[0] += out_h[i];
		//printf("%d: %0.3f ", i, out_h[i]);
	}
    out_h[0] = ((2*a) + (out_h[0]*2) + (2*b))*(((float)(b - a)/(num_elements))/2);
    verify(a,b,num_elements,out_h[0]);
    
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    
    // Free memory ------------------------------------------------------------

    free(A_h);


    /*************************************************************************/
    //INSERT CODE HERE
    cudaFree(A_d);
    /*************************************************************************/
    return 0;
}

