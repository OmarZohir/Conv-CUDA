#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <iostream>
#include <cmath>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#include <chrono>// timer library

//feel free to change these parameters when you do experiments
#define DATA_SIZE 1024
#define BLOCK_SIZE  256
#define MASK_WIDTH 155      // we assune the mask width should be odd value

#define TEST_ROUNDS 1000

__constant__ float mask_const[MASK_WIDTH];


void convolution_1D_host(float *in, float *m, float *out, int Mask_Width, int Width) {
   //i : index in the main array
   //j : index in the mask
    for(int i = 0; i<Width;i++){
        int radius = Mask_Width/2;
        float Res = 0;
        int startIdx = i - radius;
        for (int j = 0; j < Mask_Width; j++) {
            if (startIdx + j >= 0 && startIdx + j < Width) {
                Res += in[startIdx + j]*m[j];
            }
        }
        out[i] = Res;
    }
}  

// Naive implementation
// 
__global__ void convolution_1D_basic_kernel(float *in, float *m, float *out, int Mask_Width, int Width) {

    //Global Thread ID calculation
    int gindex = threadIdx.x + blockIdx.x * blockDim.x;  //global index
    
    // Mask radius calculation (for readibility)
    int radius = MASK_WIDTH/2;
    
    // Calculating starting point of the calculation
    int startIdx = gindex - radius;
    float result = 0;
    for(int i = 0; i < Mask_Width ;i++){
        if( startIdx + i >= 0 && startIdx + i < Width ) { // boundary check
            result += in[startIdx+i]*m[i];
        }
    }
    out[gindex] = result;
}
  
//Assumption: Padding is always less than the size of 1 block
// Tiled convolution  Implement Your self
__global__ void convolution_1D_basic_tiled_kernel(float *in, float *m, float *out, int Mask_Width, int Width) {

    //Global thread ID calculation
    int gindex = threadIdx.x + blockIdx.x * blockDim.x;

    // Store all elements needed to compute output in shared memory
    extern __shared__ float s_array[];
    
    // Mask radius calculation (for readibility) = number of padded elements on either side of the input array
    int radius = Mask_Width/2;
    int Pad = 2*radius;
    
    //Padded array size, i.e., size of in
    int n_padded = Width + Pad;
   
   // Offset for the second set of loads in shared memory
   int offset = threadIdx.x + blockDim.x; 

   // Global offset for the array in DRAM = gindex + blockDim.x 
   //int g_offset = blockDim.x * blockIdx.x + offset;
    
    // Load the memory block into the shared memory
    if (gindex < DATA_SIZE)
        s_array[threadIdx.x] = in[gindex];

    // Load the remaining elements needed for the padding into the shared memory
    if (threadIdx.x < 2*radius && gindex + blockDim.x < n_padded){ 
            s_array[offset] = in[gindex + blockDim.x];
    } 
    __syncthreads();
    
    //Calculation part, Each thread calculates one element of the output array

    float result = 0;

    for (int i = 0; i < MASK_WIDTH; i++){
        result += s_array[threadIdx.x + i] * m[i]; 
    }

    //write back the results
    if (gindex < DATA_SIZE){
        out[gindex] = result;
    }
}

// Implement yourself, tiled with constant memory
__global__ void convolution_1D_tiled_const_kernel(float *in, float *out, int Mask_Width, int Width) {
    //Global thread ID calculation
    int gindex = threadIdx.x + blockIdx.x * blockDim.x;

    // Store all elements needed to compute output in shared memory
    extern __shared__ float s_array[];
    
    // Mask radius calculation (for readibility) = number of padded elements on either side of the input array
    int radius = Mask_Width/2;
    int Pad = 2*radius;
    
    //Padded array size, i.e., size of in
    int n_padded = Width + Pad;
   
   // Offset for the second set of loads in shared memory
   int offset = threadIdx.x + blockDim.x; 

   // Global offset for the array in DRAM = gindex + blockDim.x 
   //int g_offset = blockDim.x * blockIdx.x + offset;
    
    // Load the memory block into the shared memory
    if (gindex < DATA_SIZE)
        s_array[threadIdx.x] = in[gindex];

    // Load the remaining elements needed for the padding into the shared memory
    if (threadIdx.x < 2*radius && gindex + blockDim.x < n_padded){ 
            s_array[offset] = in[gindex + blockDim.x];
    } 
    __syncthreads();
    
    //Calculation part, Each thread calculates one element of the output array

    float result = 0;

    for (int i = 0; i < MASK_WIDTH; i++){
        result += s_array[threadIdx.x + i] * mask_const[i]; 
    }

    //write back the results
    if (gindex < DATA_SIZE){
        out[gindex] = result;
    }
}

// Initialize arrays of size DATA_SIZE into random numbers of type float
void initial(float *in){
    float LO = -1;
    float HI = 1;
    for (int i = 0; i < DATA_SIZE;i++){
        in[i] = LO + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(HI-LO)));
    }
}

//Initialize arrays of size MASK_SIZE into random numbers of type float
void initial_mask(float *mask){
    float LO = -1;
    float HI = 1;
    for (int i =0; i < MASK_WIDTH;i++){
        mask[i] = LO + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(HI-LO)));
    }
}

void print_mask(float* mask){
    std::cout<<"[";
    for(int i =0; i < MASK_WIDTH; i++){
        
        std::cout<<mask[i]<<",";
    }
    std::cout<<"]"<<std::endl;
}

void printOut(float *out){
    std::cout<<"[";
    for(int i =0; i < DATA_SIZE; i++){
        
        std::cout<<out[i]<<",";
    }
    std::cout<<"]"<<std::endl;
}


void check_errors(float* h_out, float* d_out, int size){
    float errors = 0;
    for(int i =0;i < size;i++){
        errors += abs(h_out[i] - d_out[i]);
    }
    float avg_err = errors/size;
    //std::cout << "average errors = " << errors/size<<std::endl;
    if(avg_err > 0.001){
        std::cout << "average errors = " << avg_err<<std::endl;
        std::cout << " error: Check your CUDA implementation! the result is not numerically correct compared to C program" << std::endl;
    }
}


void run_C(){


    float in[DATA_SIZE]; 
    float mask[MASK_WIDTH]; 
    float out[DATA_SIZE];


    //convolution_1D_host(float *in, float *m, float *out, int Mask_Width, int Width)

    initial(in);
    initial_mask(mask);
    float total = 0;
    for(int i=0; i<TEST_ROUNDS;i++){

        auto begin = std::chrono::high_resolution_clock::now();
        convolution_1D_host(in,mask,out,MASK_WIDTH,DATA_SIZE);
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<float, std::milli> fp_ms = end - begin;

        total += fp_ms.count();
    }



    std::cout << "C program time  : " << total/TEST_ROUNDS << " ms " << std::endl;



}

void run_Naive_CUDA(){
    float in[DATA_SIZE]; 
    float mask[MASK_WIDTH]; 
    float out[DATA_SIZE];

    float host_out[DATA_SIZE];

    float *d_in, *d_out, *d_mask;
    int size = sizeof(float) * DATA_SIZE;
    int size_mask = sizeof(float) * MASK_WIDTH;

    cudaMalloc( (void **) &d_in, size);
    cudaMalloc( (void **) &d_out, size);
    cudaMalloc( (void **) &d_mask, size_mask ); 
 
    // host data initialization
    initial(in);
    initial_mask(mask);
    //print_mask(mask);
    // std::cout<<"Input = "<< std::endl;
    // printOut(in);

    // CUDA memory copy function, from host to device
    cudaMemcpy(d_in, in, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, mask, size_mask, cudaMemcpyHostToDevice);

    //create CUDA event to measure the execution time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // do compuation on device(GPU)

    float total = 0;

    // do compuation on device(GPU)
    for (int i = 0;i < TEST_ROUNDS;i++ ){

        cudaEventRecord(start);

        convolution_1D_basic_kernel<<<(DATA_SIZE+BLOCK_SIZE-1)/BLOCK_SIZE,BLOCK_SIZE>>>(d_in,d_mask,d_out,MASK_WIDTH,DATA_SIZE);
    
        cudaDeviceSynchronize();
    
        cudaEventRecord(stop);
    
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        total += milliseconds;

    }

    

    std::cout << "Naive CUDA Kernel Execution time(without data transfer time) = " << total/TEST_ROUNDS  << " ms"<< std::endl;
    
    
    // Copy result back to host (CPU)
    cudaMemcpy(out, d_out, size, cudaMemcpyDeviceToHost);
    //error check and report any possible GPU errors

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cout<< " CUDA failed "<< std::endl;
        std::cout << "Error: " << cudaGetErrorString(err) << std::endl;
    }


    convolution_1D_host(in, mask, host_out,  MASK_WIDTH,  DATA_SIZE);

    check_errors(host_out,out,DATA_SIZE);

    cudaFree(d_out);
    cudaFree(d_in);
    cudaFree(d_mask);

}


void run_tiled_CUDA(){

    //Array padding
    int r = MASK_WIDTH/2; //radius of the mask as an integer, needed for reuse
    int n_padded = DATA_SIZE + r*2;

    //Size of the padded input array, output array, and the mask array in bytes
    size_t size_padded = sizeof(float) * n_padded;
    int size = sizeof(float) * DATA_SIZE;
    int size_mask = sizeof(float) * MASK_WIDTH;

    //Allocate the input array, with padded elements
    float* in = new float[size_padded];
    
    //Initialize the padded elements to zero, and initialize the rest of the elements using
    //the function provided
    for (int i = 0; i < n_padded; i++) {
        if ((i < r) || (i >= (DATA_SIZE + r))) {
          in[i] = 0;
        } 
    }
    initial((float *)(in + r));

    // Allocate the mask and the output arrays at the host
    float mask[MASK_WIDTH]; 
    float out[DATA_SIZE];
    float host_out[DATA_SIZE];
    
    // Mask initialization
    initial_mask(mask);

    // Allocate memory on the device for the input array, output array, and the mask
    float *d_in, *d_out, *d_mask;
    cudaMalloc( (void **) &d_in, size_padded);
    cudaMalloc( (void **) &d_out, size);
    cudaMalloc( (void **) &d_mask, size_mask); 

    //print_mask(mask);
    // std::cout<<"Input = "<< std::endl;
    // printOut(in);

    // CUDA memory copy function, from host to device
    cudaMemcpy(d_in, in, size_padded, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, mask, size_mask, cudaMemcpyHostToDevice);
    //create CUDA event to measure the execution time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float total = 0;
    int GridSize = (DATA_SIZE+BLOCK_SIZE-1)/BLOCK_SIZE; 

    //Amount of space per block of shared memory
    //Design decision: Shared memory size = 1 BLOCK size
    size_t SHMEM = sizeof(float) * (BLOCK_SIZE + r*2);

    // do compuation on device(GPU)
    for (int i = 0;i < TEST_ROUNDS;i++ ){

        cudaEventRecord(start);

        convolution_1D_basic_tiled_kernel<<<GridSize,BLOCK_SIZE,SHMEM>>>(d_in,d_mask,d_out,MASK_WIDTH,DATA_SIZE);
    
        cudaDeviceSynchronize();
    
        cudaEventRecord(stop);
    
    
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        total += milliseconds;

    }

    

    std::cout << "Tiled Kernel Execution time(without data transfer time) = " << total/TEST_ROUNDS  << " ms"<< std::endl;

    // Copy result back to host (CPU)
    cudaMemcpy(out, d_out, size, cudaMemcpyDeviceToHost);
    //error check and report any possible GPU errors

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cout<< " CUDA failed "<< std::endl;
        std::cout << "Error: " << cudaGetErrorString(err) << std::endl;
    }

    convolution_1D_host((float*)(in + r), mask, host_out,  MASK_WIDTH,  DATA_SIZE);

    check_errors(host_out,out,DATA_SIZE);

    delete[] in;
    cudaFree(d_out);
    cudaFree(d_in);
    cudaFree(d_mask);

}



void run_tiled_const_CUDA(){
    
    //Array padding
    int r = MASK_WIDTH/2; //radius of the mask as an integer, needed for reuse
    int n_padded = DATA_SIZE + r*2;
    size_t size_padded = sizeof(float) * n_padded;

    //Allocate the input array, with padded elements
    float* in = new float[size_padded];
    
    //Initialize the padded elements to zero, and initialize the rest of the elements using
    //the function provided
    for (int i = 0; i < n_padded; i++) {
        if ((i < r) || (i >= (DATA_SIZE + r))) {
          in[i] = 0;
        } 
    }
    initial((float *)(in + r));

    float mask[MASK_WIDTH]; 
    float out[DATA_SIZE];

    float host_out[DATA_SIZE];

    float *d_in, *d_out;
    int size = sizeof(float) * DATA_SIZE; //size of output only, not the input array
    int size_mask = sizeof(float) * MASK_WIDTH;

    cudaMalloc( (void **) &d_in, size_padded);
    cudaMalloc( (void **) &d_out, size);
 
    // Mask initialization
    initial_mask(mask);

    //print_mask(mask);
    // std::cout<<"Input = "<< std::endl;
    // printOut(in);

    // CUDA memory copy function, from host to device
    cudaMemcpy(d_in, in, size_padded, cudaMemcpyHostToDevice);

    // constant memory
    cudaMemcpyToSymbol(mask_const,mask,size_mask);

    //create CUDA event to measure the execution time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float total = 0;

    int GridSize = (DATA_SIZE+BLOCK_SIZE-1)/BLOCK_SIZE;

    //Amount of space per block of shared memory
    //Design decision: Shared memory size = 1 BLOCK size
    size_t SHMEM = sizeof(float) * (BLOCK_SIZE + r*2);


    // do compuation on device(GPU)
    for (int i = 0;i < TEST_ROUNDS;i++ ){

        cudaEventRecord(start);

        convolution_1D_tiled_const_kernel<<<GridSize,BLOCK_SIZE,SHMEM>>>(d_in,d_out,MASK_WIDTH,DATA_SIZE); 
    
        cudaDeviceSynchronize();
    
        cudaEventRecord(stop);
    
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        total += milliseconds;

    }

    std::cout << "Tiled + Const Mem CUDA Kernel Execution time(without data transfer time) = " << total/TEST_ROUNDS  << " ms" << std::endl;
    

    // Copy result back to host (CPU)
    cudaMemcpy(out, d_out, size, cudaMemcpyDeviceToHost);
    //error check and report any possible GPU errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cout<< " CUDA failed "<< std::endl;
        std::cout << "Error: " << cudaGetErrorString(err) << std::endl;
    }

    // compute C
    convolution_1D_host((float*)(in + r), mask, host_out,  MASK_WIDTH,  DATA_SIZE);

    check_errors(host_out,out,DATA_SIZE);

    delete[] in;
    cudaFree(d_out);
    cudaFree(d_in);
}



int main() {
    run_C();
    run_Naive_CUDA();
    run_tiled_CUDA();
    run_tiled_const_CUDA();



    return 0;
}