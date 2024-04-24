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

#define CHECK_BANK_CONFLICTS 1
#if CHECK_BANK_CONFLICTS
#define TILE(i, j) cutilBankChecker(((float*)&Tile[0][0]), ((BLOCK_SIZE+MASK_SIZE-1) * i + j))
#else
#define TILE(i, j) Tile[i][j]
#endif


#define IN_SIZE 240
#define MASK_SIZE 7 // or 3 or 7
#define TEST_ROUNDS 1000
#define KERNEL_ROUNDS 1000
#define BLOCK_SIZE 16

__constant__ float mask_const[MASK_SIZE*MASK_SIZE];


// C implementation
void conv2d_host(float* input,float* mask,float* output,int in_size,int mask_size){

    for (int out_h = 0;out_h<in_size;out_h++ ){
        for(int out_w = 0; out_w < in_size; out_w++ ){
            float res = 0;
            for(int mask_h = 0; mask_h < mask_size;mask_h++){
                for(int mask_w = 0; mask_w < mask_size;mask_w++){
                    
                    int input_h = out_h - mask_size/2 + mask_h;
                    int input_w = out_w - mask_size/2 + mask_w;
                    
                    if(input_h  >= 0   && input_h  < in_size  && input_w >=0 && input_w < in_size ){
                        res += mask[mask_w + mask_h * mask_size] * input[input_h*in_size + input_w];
                    }
                }
            }
            output[out_w + out_h*in_size] = res;
        }
    }
}


/*****************You CUDA kernel implementation*************************/

/*
in: input array of dimensions BLOCK_SIZE * BLOCK_SIZE, i.e., a square matrix
out: output array, of dimensions BLOCK_SIZE * BLOCK_SIZE
m: mask array, each dimension of the matrix is of size MASK_SIZE, which is smaller than to BLOCK_SIZE
N: Length of one dimension of the input square matrix
maskN: Length of one dimension of the mask square matrix
*/
__global__ void  conv2d_Naive_CUDA(float *in, float *out, float *m, int N, int maskN){
    // Accumulate row i of A and column j of B
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Starting index for calculation
    int start_r = row - int(maskN/2);
    int start_c = col - int(maskN/2);

    //variable for accumulating the result
    float accu = 0.0;

    // Iterate over all the rows
    for (int i = 0; i < maskN; i++) {
        // Go over each column
        for (int j = 0; j < maskN; j++) {
            // Range check for rows
            if ((start_r + i) >= 0 && (start_r + i) < N) {
                // Range check for columns
                if ((start_c + j) >= 0 && (start_c + j) < N) {
                    // Accumulate result
                    accu += in[(start_r + i) * N + (start_c + j)] * m[i * maskN + j];
                }
            }
        }
    }

  // Write back the result
  out[row * N + col] = accu;
}


__global__ void  conv2d_tiled_constant_mem(float* in, float* out, int N) {

    // Declaration of the shared memory array Tile used to
    // store the sub-matrix of In (i.e., the tile)
    __shared__ float Tile[BLOCK_SIZE + MASK_SIZE - 1][BLOCK_SIZE + MASK_SIZE - 1];

    int radius = int(MASK_SIZE / 2);
    int padding = 2 * radius;

    // Block and thread indices
    int bx = blockIdx.x;  int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    // Finding current row and column being calculated by that thread
    int row = by * blockDim.y + ty;
    int col = bx * blockDim.x + tx;

    // Starting index for calculation
    int start_r = row - int(MASK_SIZE/2);
    int start_c = col - int(MASK_SIZE/2);

    //First Load into shared memory (including padding)
    if (start_r >= 0 && start_r < N && start_c >= 0 && start_c < N) {
        TILE(ty, tx)= in[start_r * N + start_c];
    }
    else {
        //Adding padding on the sides of the main array
        TILE(ty, tx) = 0;
    }

    int offsetX = blockDim.x; int offsetY = blockDim.y;
    int Xidx = start_r * N + start_c + offsetX;
    int yidx = (start_r + offsetY) * N + start_c;
    int NextRow = start_r + blockDim.y;
    int NextCol = start_c + blockDim.x;

    //2nd set of loads into shared memory to allow independent calculations of threads
    if (tx < padding){
        //Add an extra element from the end of the row
        //if start_r is -1 (i.e., the first padding row), just add a zero to the end of the row
        if (start_r >= 0 && NextCol < N)
            TILE(ty, tx + offsetX) = in[Xidx];
        else
            TILE(ty, tx + offsetX) = 0;
    }

    if (ty < padding){
        if (start_c >= 0 && NextRow < N)
            //Load an extra element from the end of the column
            TILE(ty + offsetY, tx) = in[yidx];
        else
            TILE(ty + offsetY, tx) = 0;
    }
    
    // Load the remaining (BLOCK_SIZE - MASK_SIZE) * (BLOCK_SIZE - MASK_SIZE) into the padded tile 
    if (tx < padding && ty < padding) {
        if (NextCol < N && NextRow < N)
            TILE(ty + offsetY, tx + offsetX) = in[(start_r + offsetY) * N + start_c + offsetX];
        else
            TILE(ty + offsetY, tx + offsetX) = 0;
    }

    __syncthreads();

    //Multiply the mask by the tile 
    float accu = 0;

    // Iterate over all the rows
    for (int i = 0; i < MASK_SIZE; i++){
        // Go over each column
        for (int j = 0; j < MASK_SIZE; j++) {
            // Accumulate result
            accu += TILE(ty + i, tx + j) * mask_const[i * MASK_SIZE + j];
        }
    }
   
    if (row < N && col < N)
        out[row*N + col] = accu;
}


/*****************You CUDA kernel implementation*************************/



void initial_input(float* input, int in_size){
    float LO = -1;
    float HI = 1;
    for(int h = 0; h < in_size;h++){
        for(int w = 0; w< in_size;w++){
            input[h*in_size + w] =  LO + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(HI-LO)));
        }
    }
}

void initial_mask(float* mask, int mask_size){
    float LO = -1;
    float HI = 1;
    for (int mask_h = 0; mask_h < mask_size; mask_h++){
        for (int mask_w = 0; mask_w < mask_size; mask_w++){
            mask[mask_h*mask_size + mask_w] = LO + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(HI-LO)));
        }
    }
}


void print(float* in, int size){

    std::cout<<"[";
    for(int i =0; i < size; i++){
        for(int  j = 0; j < size; j++){
            std::cout<<in[j + i*size]<<",";
        }
        std::cout<<std::endl;
    }
    std::cout<<"]"<<std::endl;

}


void check_errors(float* h_out, float* d_out, int size){
    float errors = 0;
    //FOR DEBUGGING 
    /*float errorIdx[1000];
    int errorCount = 0;*/
    for(int i =0;i < size;i++){
        errors += abs(h_out[i] - d_out[i]);
        /*if (abs(h_out[i] - d_out[i]) > 0.001)
        {
            errorIdx[errorCount] = i;
            errorCount++;
        }*/
    }
    float avg_err = errors/size;
    //std::cout << "average errors = " << errors/size<<std::endl;
    if(avg_err > 0.001){
        std::cout << "average errors = " << avg_err<<std::endl;
        std::cout << " error: Check your CUDA implementation! the result is not numerically correct compared to C program" << std::endl;
    }
    /*std::cout << "Error indices are ";
    
    for (int i = 0; i < errorCount; i++) {
        std::cout << errorIdx[i] << ", ";
    }
    std::cout << std::endl;*/
}


void run_c(){

    int in_size = IN_SIZE;
    int mask_size = MASK_SIZE;
    float input[IN_SIZE * IN_SIZE];

    float mask[MASK_SIZE * MASK_SIZE];

    float output[IN_SIZE * IN_SIZE];

    initial_input(input,in_size);
    initial_mask(mask,mask_size);

    float total = 0;

    for(int i=0;i<TEST_ROUNDS;i++){
        auto begin = std::chrono::high_resolution_clock::now();
        conv2d_host(input,mask,output,in_size,mask_size);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float, std::milli> fp_ms = end - begin;

        total += fp_ms.count();


    }

    std::cout << "C program time  : " << total/TEST_ROUNDS << " ms " << std::endl;
}



void run_Naive_CUDA(){

    int in_size = IN_SIZE;
    int mask_size = MASK_SIZE;


    float h_input[IN_SIZE * IN_SIZE];

    float h_mask[MASK_SIZE * MASK_SIZE];

    float h_output[IN_SIZE * IN_SIZE];

    float ref_out[IN_SIZE * IN_SIZE];

    float *d_in, *d_out, *d_mask;
    int size = sizeof(float) * in_size*in_size;
    int size_mask = sizeof(float) * mask_size*mask_size;


    cudaMalloc( (void **) &d_in, size);
    cudaMalloc( (void **) &d_out, size);
    cudaMalloc( (void **) &d_mask, size_mask ); 


    
    initial_input(h_input,in_size);
    initial_mask(h_mask,mask_size);


    // CUDA memory copy function, from host to device
    cudaMemcpy(d_in, h_input, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, h_mask, size_mask, cudaMemcpyHostToDevice);

    //create CUDA event to measure the execution time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);


    // Calculate grid dimensions
    int GridSize = (IN_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Dimension launch arguments
    dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_dim(GridSize, GridSize);

    float total = 0;


    for (int i = 0;i <KERNEL_ROUNDS;i++ ){

        cudaEventRecord(start);

        //TODO: invoke your kernel  conv2d_Naive_CUDA
        conv2d_Naive_CUDA<<<grid_dim, block_dim>>>(d_in, d_out, d_mask,IN_SIZE,MASK_SIZE);
    
        cudaDeviceSynchronize();
    
        cudaEventRecord(stop);
    
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        total += milliseconds;

    }

    std::cout << "Naive CUDA Kernel Execution time(without data transfer time) = " << total/KERNEL_ROUNDS << " ms"<< std::endl;
    
    
    // Copy result back to host (CPU)
    cudaMemcpy(h_output, d_out, size, cudaMemcpyDeviceToHost);
    //error check and report any possible GPU errors

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cout<< " CUDA failed "<< std::endl;
        std::cout << "Error: " << cudaGetErrorString(err) << std::endl;
    }

    // void conv2d_host(float* input,float* mask,float* output,int in_size,int mask_size)
    conv2d_host(h_input, h_mask, ref_out,  in_size,  mask_size);

    check_errors(ref_out,h_output,in_size*in_size);

    cudaFree(d_out);
    cudaFree(d_in);
    cudaFree(d_mask);
}


void run_tiled_constant_mem(){

    int in_size = IN_SIZE;
    int mask_size = MASK_SIZE;

    float h_input[IN_SIZE * IN_SIZE];

    float h_mask[MASK_SIZE * MASK_SIZE];

    float h_output[IN_SIZE * IN_SIZE];

    float ref_out[IN_SIZE * IN_SIZE];

    float *d_in, *d_out;
    int size = sizeof(float) * in_size*in_size;
    int size_mask = sizeof(float) * mask_size*mask_size;


    cudaMalloc( (void **) &d_in, size);
    cudaMalloc( (void **) &d_out, size);
    //cudaMalloc( (void **) &d_mask, size_mask ); 


    
    initial_input(h_input,in_size);
    initial_mask(h_mask,mask_size);



    // CUDA memory copy function, from host to device
    cudaMemcpy(d_in, h_input, size, cudaMemcpyHostToDevice);
    //cudaMemcpy(d_mask, h_mask, size_mask, cudaMemcpyHostToDevice);


    // constant memory
    cudaMemcpyToSymbol(mask_const,h_mask,size_mask);

    // Calculate grid dimensions
    int GridSize = (IN_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Dimension launch arguments
    dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_dim(GridSize, GridSize);

    //FOR DEBUGGING
    //print(h_input, IN_SIZE);
    //print(h_mask, MASK_SIZE);

    //create CUDA event to measure the execution time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float total = 0;


    for (int i = 0;i < KERNEL_ROUNDS;i++ ){

        cudaEventRecord(start);

        //TODO invoke your kernel  conv2d_const_tiled_CUDA
        conv2d_tiled_constant_mem <<<grid_dim, block_dim>>>(d_in,d_out, IN_SIZE);
        cudaDeviceSynchronize();
    
        cudaEventRecord(stop);
    
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        total += milliseconds;

    }



    std::cout << "Tiled + Const Mem CUDA Kernel Execution time(without data transfer time) = " << total/ KERNEL_ROUNDS << " ms"<< std::endl;
    
    
    // Copy result back to host (CPU)
    cudaMemcpy(h_output, d_out, size, cudaMemcpyDeviceToHost);
    //error check and report any possible GPU errors

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cout<< " CUDA failed "<< std::endl;
        std::cout << "Error: " << cudaGetErrorString(err) << std::endl;
    }

    // void conv2d_host(float* input,float* mask,float* output,int in_size,int mask_size)
    conv2d_host(h_input, h_mask, ref_out,  in_size,  mask_size);
    check_errors(ref_out,h_output,in_size*in_size);


    cudaFree(d_out);
    cudaFree(d_in);
    // cudaFree(d_mask);

}


int main() {

    run_c();
    run_Naive_CUDA();
    run_tiled_constant_mem();

/************* Your CUDA program, you can follow the steps *******************/

/*************step 1 : create host/device arrary pointer********************/

/**************step 2 : allocate host/device memory ***********************/

/**************step 3 set up GPU event timer **************************/


/**************step 4 Kernel configurations & kernel launch **********************/


/**************step 5: check the GPU results with CPU baseline ****************/


/**************step 6: report GPU kernel time & free the allocated memory ****************/


    return 0;
}