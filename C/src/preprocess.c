#include <stdlib.h>
#include <stdio.h>

#include "../headers/preprocess.h"

void readData(const char data[], int **csr_rows, int **col_indx, int* nnz, int * rows, int* cols){
    char * line = NULL;
    size_t len = 0;
    size_t read;
    FILE *f = fopen(data,"r");
    while (read = getline(&line, &len , f) != -1)
    {
        if(line[0] == '%'){
            continue;
        }
        sscanf(line, "%d %d %d", rows, cols, nnz);
        break;
    }
    printf("-READ %d %d %d\n",*rows,*cols,*nnz);
    col_indx[0] = (int*)malloc((*nnz)*sizeof(int));
    csr_rows[0] = (int*)malloc((*rows)*sizeof(int));
    for(int i = 0; i < *nnz; i++){
        int temp_r;
        fscanf(f, "%d %d", &col_indx[0][i] , &temp_r);
        // data have 1 base index
        // transform to 0-based index
        col_indx[0][i]--;
        csr_rows[0][temp_r-1]++;
    }

    fclose(f);
}

int GetMaxNNZperRow(int* csr_rows, int len){
    int max = 0;
    for(int i = 0; i < len; i++){
        if(max < csr_rows[i]){
            max = csr_rows[i];
        }
    }

    return max;
}

int* GetIntgrCSR(int* csr_rows, int rows){
    // initialize
    int* intg_csr_rows = (int*)malloc(rows*sizeof(int));
    intg_csr_rows[0] = 0;
    for(int i = 1; i < rows; i++){
        intg_csr_rows[i] = intg_csr_rows[i-1] + csr_rows[i-1];
    }

    return intg_csr_rows;
}