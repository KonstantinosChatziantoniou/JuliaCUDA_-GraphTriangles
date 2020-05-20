#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>
//https://proofwiki.org/wiki/Product_of_Triangular_Matrices
#include "../headers/preprocess.h"
#include "../headers/helper.h"
#include "../headers/kernel_row_per_th.h"
#include "../headers/kernel_elem_per_th.h"
#include "../headers/kernel_elem_per_th_limit.h"



int main(int argc, char** argv){

    if(argc != 2){
        printf("Invalid number of arguments\n");
        return 1;
    }


    // "auto.mtx"; // "data.csv"; //  "great-britain_osm.mtx"; // "delaunay_n22.mtx"; //
    printf("Dataset: %s\n",argv[1]);
    int rows,cols,nnz,max_per_row;
    int *col_indx, *csr_rows, *intg_csr_rows;
    int *res_sum;
    struct timeval start,end;
    /* Read Data in COO format and transform to 0 based index */
    gettimeofday(&start,NULL);
    readData(argv[1],&csr_rows,&col_indx,&nnz,&rows,&cols);
    max_per_row = GetMaxNNZperRow(csr_rows, rows);
    intg_csr_rows = GetIntgrCSR(csr_rows, rows);
    gettimeofday(&end,NULL);
    printTime(start,end, "Read and Preprocess Data");


    int res = 0;

    res = CountTriangles_row_per_th(rows, nnz, max_per_row, intg_csr_rows, col_indx);
    printf("Res row per thread %d.\n",res);

    res = 0;
    res = CountTriangles_elem_per_th(rows, nnz, max_per_row, intg_csr_rows, col_indx);
    printf("Res elem per thread %d.\n",res);


    res = 0;
    res = CountTriangles_elem_per_th_limit(rows, nnz, max_per_row, intg_csr_rows, col_indx, 1);
    printf("Res elem per thread with limited threads 32 %d.\n",res);



    res = 0;
    res = CountTriangles_elem_per_th_limit(rows, nnz, max_per_row, intg_csr_rows, col_indx, 2 );
    printf("Res elem per thread with limited threads 64 %d.\n",res);


}