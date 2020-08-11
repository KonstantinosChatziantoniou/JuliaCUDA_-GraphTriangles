using CUDAnative, CUDAdrv, CuArrays



# Each block is assigned a row.
# Each thread checks  the main row
# with one of the element of the columns pointed by the row.
function countTri_elem_per_th_nt(csr_rows, col_indx, out_sum, max_nnz)
    bid = blockIdx().x + (blockIdx().y-1)*gridDim().x + (blockIdx().z-1)*gridDim().x*gridDim().y
    rows = size(csr_rows)[1]
    nnz = size(col_indx)[1]
    tid = threadIdx().x + (threadIdx().y-1)*blockDim().x
    stride = blockDim().x*blockDim().y
    if bid > rows
        return nothing
    end
    sh_row = @cuDynamicSharedMem(Int32, stride)
    sh_sum = @cuDynamicSharedMem(Int32, stride, stride*sizeof(Int32))
    sh_cols = @cuDynamicSharedMem(Int32, max_nnz*max_nnz, sizeof(Int32)*2*stride)

    @inbounds sh_sum[tid] = 0
    @inbounds row_start =  csr_rows[bid]
    @inbounds row_end = (bid == rows) ?  nnz+1 : csr_rows[bid + 1]
    len = row_end-row_start
    if tid <= len
        @inbounds sh_row[tid] = col_indx[row_start + tid - 1]
    end

    sync_threads()

    # #Read all the other rows
    start_save = 0
    for i = 1:len
        @inbounds col = sh_row[i]
        @inbounds row_start = csr_rows[col]
        @inbounds row_end = (col == rows) ?  nnz+1 : csr_rows[col + 1]
        templen =  row_end-row_start
        if tid  <= templen
            @inbounds sh_cols[start_save+tid] = col_indx[row_start+tid-1]
        end
        start_save += templen
    end
    sync_threads()
    sum = 0
    for i = 0:stride:(start_save-1)
        if tid + i <= start_save
            for j = 1:len
                @inbounds sum += sh_row[((tid+j-2)%len)+1] == sh_cols[i + tid]
            end
        end
    end
    sync_threads()
    w = 16
    while w > 0
        sum += shfl_down_sync(0xffffffff, sum, w)
        w = floor(w/2)
    end
    if threadIdx().x == 1
        @inbounds sh_sum[threadIdx().y] = sum
    end
    sync_threads()
    if tid == 1
        for i = 2:blockDim().y
            @inbounds sum += sh_sum[i]
        end
        @inbounds out_sum[bid] = sum
    end
    return nothing
end


function CountTriangles_ElemPerTh_nt(csr_rows, col_indx, max_per_row)
    dev_csr_rows = CuArray(csr_rows)
    dev_col_indx = CuArray(col_indx)
    dev_out_sum = CuArrays.zeros(Int32, size(csr_rows)[1])
    th_groups = 1
    threads = 0
    while 32*th_groups < max_per_row
        th_groups += 1
    end
    threads = 32*th_groups
    blocks = Int64(floor((size(csr_rows)[1])/(512*512)+ 1))
    println("2 Total blocks: ", 512*512*blocks, " vs rows: ", size(csr_rows)[1])
    println("\t Threads: ", threads)
    @cuda(blocks=(512,512,blocks), threads=(32,th_groups,1),
        shmem=sizeof(Int32)*(threads*2 + max_per_row*max_per_row),
        countTri_elem_per_th_nt(dev_csr_rows, dev_col_indx, dev_out_sum, max_per_row))
    out_sum = Array(dev_out_sum)
    # println(dev_out_sum)
    return reduce(+, out_sum, init=0)
end
