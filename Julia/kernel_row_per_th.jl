using CUDA
# Each block is assigned a row.
# Each thread checks the main row with one of the columns pointed by the row.
function countTri_row_per_th(csr_rows, col_indx, out_sum)
    ## Indexing for threads and blocks.
    bid::Int32 = blockIdx().x + (blockIdx().y-1)*gridDim().x + (blockIdx().z-1)*gridDim().x*gridDim().y
    rows::Int32 = size(csr_rows)[1]
    nnz::Int32 = size(col_indx)[1]
    tid::Int32 = threadIdx().x + (threadIdx().y-1)*blockDim().x
    stride::Int32 = blockDim().x*blockDim().y
    if bid > rows
        return nothing
    end
    ## Dynamic Memory for all threads.
    sh_sum = @cuDynamicSharedMem(Int32, stride)
    sh_row = @cuDynamicSharedMem(Int32, stride, stride*sizeof(Int32))
    sh_len = @cuDynamicSharedMem(Int32, stride, 2*stride*sizeof(Int32))
    sh_cols = @cuDynamicSharedMem(Int32, (stride,stride), 3*stride*sizeof(Int32))
    @inbounds sh_sum[tid] = 0
    ## Info for the main row.
    @inbounds row_start::Int32 =  csr_rows[bid]
    @inbounds row_end::Int32 = (bid == rows) ?  nnz+1 : csr_rows[bid + 1]

    len::Int32 = row_end-row_start
    if tid <= len
        @inbounds sh_row[tid] = col_indx[row_start + tid - 1]
    end
    sync_threads()

    # #Read all the other rows
    for i::Int32 = 1:len
            # @inbounds col::Int32 = sh_row[i]
            # @inbounds row_start = csr_rows[col]
            # @inbounds row_end = (col == rows) ?  nnz+1 : csr_rows[col + 1]
            if threadIdx().x == 1
                @inbounds col::Int32 = sh_row[i]
                @inbounds row_start = csr_rows[col]
                @inbounds row_end = (col == rows) ?  nnz+1 : csr_rows[col + 1]
            end
            row_start = shfl_sync(0xffffffff, row_start, 1)
            row_end = shfl_sync(0xffffffff, row_end, 1)
            templen::Int32 =  row_end-row_start
            (tid==1) && (sh_len[i] = templen)
            if tid  <= templen
                @inbounds sh_cols[i, tid] = col_indx[row_start + tid - 1]
            end
    end
    sync_threads()
    # Sum
    sum::Int32 = 0
    if tid <= len
        a::Int32  = 1
        b::Int32  = 1
        @inbounds blen::Int32 = sh_len[tid]
        while true
            if a > len || b > blen
                break;
            end

            @inbounds b1::Int32 = sh_row[a] == sh_cols[tid,b];
            @inbounds b2::Int32 = sh_row[a] > sh_cols[tid,b];
            @inbounds b3::Int32 = sh_row[a] < sh_cols[tid,b];

            b = b + b1 + b2
            a = a + b1 + b3
            sum = sum + b1
        end
    end
    w::Int32 = 16
    while w > 0
        sum += shfl_down_sync(0xff, sum, w)
        w = floor(w/2)
    end
    sync_threads()
    if threadIdx().x == 1
        sh_sum[threadIdx().y] = sum
    end
    if tid == 1
        for i::Int32 = 2:blockDim().y
            @inbounds sum += sh_sum[i]
        end
        @inbounds out_sum[bid] = sum
    end
    return nothing
end


function CountTriangles_RowPerTh(csr_rows, col_indx, max_per_row)
    dev_csr_rows = CuArray(csr_rows)
    dev_col_indx = CuArray(col_indx)
    dev_out_sum = CUDA.zeros(Int32, size(csr_rows)[1])

    blocks = Int64(floor((size(csr_rows)[1])/(512*512)+ 1))
    ## Threads MUST be more then the max nnz elements per row
    th_groups = 1
    threads = 0
    while 32*th_groups < max_per_row
        th_groups += 1
    end
    threads = 32*th_groups
    @cuda(blocks=(512,512,blocks), threads=(32, th_groups,1),
        shmem=(sizeof(Int32)*(max_per_row+3)*threads),
        countTri_row_per_th(dev_csr_rows, dev_col_indx, dev_out_sum))

    out_sum = Array(dev_out_sum)
    return reduce(+, out_sum, init=0)


end
