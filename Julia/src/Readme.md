## preprocessing.jl

Reads the data from file in COO format.
Converts the data to CSR format and calculated the maximum non zero elements per row

# kernels

## kernel.jl - method 1

Each block reads a row, and all the columns pointed by the row elements.
Each thread takes a column and "multiplies it(not really)" with the row.
Block size is equal to the max non zero elements per row.

## kernel2.jl - method 2

Each block reads a row, and all the columns pointed by the row elements.
This time the elements of the columns are stored side by side in an 1D array.
Each thread takes one element from the columns array and compares it with
all the elements of the row. This is done in a strided fashion.
The best block size is 32, as shown in the figures.

## kernel3.jl - method 3
Same as kernel 2, but the loop for checking one element against the row exits
early if the row[i] > element.
The best block size is 32, as shown in the figures.

# Comparisons

## kernel2 vs kernel3

kernel3 is consistently slower than kernel2 because of the additional
check instruction and the wrap-threads divergence.

## kernel vs kernel2

checking loop of `kernel`:  (row length -> r,  column length -> c)
      max time: r+c
      min time: min(r,c)
active threads: r


checking loop of `kernel2`:  (row length -> r,  column_array length -> ca)
      constant time of `r`
active threads: min(total_threads, ca - (stride_iter)\*total_threads )

although the `kernel`'s loop can be shorter, `kernel2` is significantly faster.

The problems of the `kernel` are:
            1. Thread divergence in the loop
            2. If r is small, the occupancy is small

The occupancy of the `kernel2` is lower only at the last stride iteration
