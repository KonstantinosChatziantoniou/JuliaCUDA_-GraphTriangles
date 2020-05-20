#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <stdlib.h>

#include "../headers/kernel_row_per_th.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line);


__global__ void gpu_kernel_1(int* col_indx,  int* csr_rows, int nnz, int rows, int* out_sum){
    // Indexing for block/threads
    int bid = blockIdx.x + blockIdx.y*gridDim.x + blockIdx.z*gridDim.x*gridDim.y;
    int tid = threadIdx.x + threadIdx.y*blockDim.x;
    int stride = blockDim.x*blockDim.y;

    if(bid >= rows){
        return;
    }
    // ------- Dynamic memory allocations -------
    extern __shared__ int sh_arr[];
    int* sh_sum = sh_arr;
    int* sh_row = &sh_arr[stride];
    int* sh_len = &sh_arr[stride*2];
    int* sh_cols = &sh_arr[stride*3];
    sh_sum[tid] = 0;
    // ------- Read main row -------
    /* The thread 0 of each wrap will read one
        element from the global memory and shuffle it
        to the rest.                                   
    */
    int start_row;
    int end_row;
    if(threadIdx.x == 0){
        start_row = csr_rows[bid]; 
        end_row = bid==(rows-1) ? nnz : csr_rows[bid+1];
    }
     
    start_row = __shfl_sync(0xffffffff, start_row, 0);
    end_row = __shfl_sync(0xffffffff, end_row, 0);
   
    int len = end_row - start_row;
    if(tid < len){
        sh_row[tid] = col_indx[start_row + tid];
    }
    

    __syncthreads();

    // ------- Read all the columns -------
    for(int i = 0; i < len; i++){
        int col = sh_row[i];
        start_row = csr_rows[col];
        end_row = col==(rows-1)? nnz : csr_rows[col+1];
        int templen = end_row - start_row;
        if(tid == 0)
            sh_len[i] = templen;
        if(tid < templen){
            sh_cols[i*stride + tid] = col_indx[start_row + tid];
        } 
    }

    __syncthreads();
    /* Each thread, will check every element of one column
        against the elements of the main row */
    int sum = 0;
    if(tid < len){
        int a = 0; 
        int b = 0;
        int blen = sh_len[tid];
        while(1){
            if(a == len || b == blen)
                break;
            int b1 = sh_row[a] == sh_cols[tid*stride + b];
            int b2 = sh_row[a] > sh_cols[tid*stride + b];
            int b3 = sh_row[a] < sh_cols[tid*stride + b];

            a = a + b1 + b3;
            b = b + b1 + b2;
            sum = sum + b1;
            
        }
    }
   __syncthreads();
    for(int w = 16; w > 0; w = w/2){
        sum += __shfl_down_sync(0xffffffff, sum, w, 32);
    }
    if(threadIdx.x == 0)
        sh_sum[threadIdx.y] = sum;
    __syncthreads();
    if(tid == 0){
        for(int i = 1; i < blockDim.y; i++){
            sh_sum[0] += sh_sum[i];
        } 
        out_sum[bid] = sh_sum[0];
    }
}

int CountTriangles_row_per_th(int rows, int nnz, int max_per_row, int* intg_csr_rows, int* col_indx){
    int *dev_cols, *dev_csr_rows;
    int* dev_sum;
    int* res_sum = (int*)malloc(rows*sizeof(int));
    gpuErrchk(cudaMalloc((void**)&dev_cols, nnz*sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&dev_csr_rows, rows*sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&dev_sum, rows*sizeof(int)));

    gpuErrchk(cudaMemcpy((void*)dev_cols, col_indx, nnz*sizeof(int),cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy((void*)dev_csr_rows, intg_csr_rows, rows*sizeof(int),cudaMemcpyHostToDevice))
    int blocks = rows/(512*512) + 1;
    int thread_groups = 0;
    /* We need more threads than the maximum number of non zero elements per row */
    while(1){
        thread_groups += 1;
        if(thread_groups*32 >= max_per_row){
            break;
        }
    }
    int threads = 32*thread_groups;
    /* Each shared memory array must have a multiple of 32 elements to avoid
        bank conflicts */
    int shmem = (max_per_row + 3)*threads*sizeof(int);
    printf("smem threads %d %d %d\n",threads, shmem, sizeof(int));
    gpu_kernel_1<<<dim3(512,512,blocks),dim3(32, thread_groups,1),shmem>>>(dev_cols, dev_csr_rows, nnz, rows, dev_sum);
    gpuErrchk(cudaMemcpy(res_sum, dev_sum, rows*sizeof(int),cudaMemcpyDeviceToHost))
    int s = 0;
    for(int i = 0; i < rows; i++){
        s += res_sum[i];
    }

    return s;


}

inline void gpuAssert(cudaError_t code, const char *file, int line)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      exit(code);
   }
}