#ifndef H_PREPRO
#define H_PREPRO

#ifdef __cplusplus
extern "C" 
#endif 
void readData(const char data[], int **csr_rows, int **col_indx, int* nnz, int * rows, int* cols);

#ifdef __cplusplus
extern "C" 
#endif 
int GetMaxNNZperRow(int* csr_rows, int len);

#ifdef __cplusplus
extern "C" 
#endif
int* GetIntgrCSR(int* csr_rows, int rows);

#endif