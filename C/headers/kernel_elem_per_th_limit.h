#ifndef H_KERNEL_EL
#define H_KERNEL_EL



#ifdef __cplusplus
extern "C" 
#endif 
int CountTriangles_elem_per_th_limit(int rows, int nnz, int max_per_row, int* intg_csr_rows, int* col_indx, int thread_groups);








#endif