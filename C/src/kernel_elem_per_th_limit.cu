#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#include "../headers/kernel_elem_per_th_limit.h"


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line);


__global__ void gpu_kernel_3(int* col_indx,  int* csr_rows, int nnz, int rows,
                                int* out_sum, int max_per_row, int max_mem_nnz){
    int bid = blockIdx.x + blockIdx.y*gridDim.x + blockIdx.z*gridDim.x*gridDim.y;
    int tid = threadIdx.x + threadIdx.y*blockDim.x;
    int stride = blockDim.x*blockDim.y;

    if(bid >= rows){
        return;
    }
    extern __shared__ int sh_arr[];
    int* sh_row = sh_arr;
    int* sh_sum = &sh_arr[max_mem_nnz];
    int* sh_cols = &sh_arr[max_mem_nnz + stride];
    sh_sum[tid] = 0;
    // Read main row -------------------------------------------
    int start_row = 1;
    int end_row = 2;
    if(threadIdx.x == 0){
        start_row = csr_rows[bid]; 
        end_row = bid==(rows-1) ? nnz : csr_rows[bid+1];
    }
     
    start_row = __shfl_sync(0xffffffff, start_row, 0);
    end_row = __shfl_sync(0xffffffff, end_row, 0);
   
    int len = end_row - start_row;
    for(int r = 0; r < len; r+=stride){
        if(tid+r < len){
            sh_row[tid+r] = col_indx[start_row + tid  + r];
        }
    }

    __syncthreads();
    /* Read the elements of the columns and stores them contigiously in
    memory. When there are enough elements (more than threads), each thread
    will take an element and compare it to all the elements of the main row */
    int total_elems = 0;
    int sum = 0;
    for(int i = 0; i < len; i++){
        int col = sh_row[i];
        start_row = csr_rows[col];
        end_row = col==(rows-1)? nnz : csr_rows[col+1];
        int temp_len = end_row-start_row;
        for(int r = 0; r < len; r+=stride){
            if(tid+r < temp_len){
                sh_cols[total_elems + tid + r] = col_indx[start_row + tid + r];
            }
        }
        total_elems += temp_len;
        __syncthreads();
        if(total_elems >= stride || i == (len-1)){
            for(int s = 0; s < total_elems; s += stride){
                for(int l = 0; l < len; l++){
                    if(tid + s < total_elems){
                        sum += (sh_row[(tid+l)%len] == sh_cols[s + tid]);
                    }
                }
            }
            total_elems = 0;
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

int CountTriangles_elem_per_th_limit(int rows, int nnz, int max_per_row, int* intg_csr_rows, int* col_indx, int thread_groups){
    int *dev_cols, *dev_csr_rows;
    int* dev_sum;
    int* res_sum = (int*)malloc(rows*sizeof(int));
    gpuErrchk(cudaMalloc((void**)&dev_cols, nnz*sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&dev_csr_rows, rows*sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&dev_sum, rows*sizeof(int)));

    gpuErrchk(cudaMemcpy((void*)dev_cols, col_indx, nnz*sizeof(int),cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy((void*)dev_csr_rows, intg_csr_rows, rows*sizeof(int),cudaMemcpyHostToDevice))
    int blocks = rows/(512*512) + 1;
    //int thread_groups = 2;
    int threads = 32*thread_groups;
    int main_rows_smem = 0;
    if(max_per_row%32 == 0){
        main_rows_smem = max_per_row;
    }else{
        main_rows_smem = 32*(max_per_row/32 + 1);
    } 
    int cols_smem = threads > max_per_row ? threads : max_per_row;
    int shmem = (main_rows_smem + threads + 2*cols_smem)*sizeof(int);
    printf("SHMEME %d maxpr %d\n", shmem, main_rows_smem);

    gpu_kernel_3<<<dim3(512,512,blocks),dim3(32,thread_groups,1),shmem>>>(dev_cols, dev_csr_rows, nnz, rows,
                 dev_sum, max_per_row ,main_rows_smem);
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