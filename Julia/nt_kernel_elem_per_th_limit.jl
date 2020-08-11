using CUDA


# Each block is assigned a row.
# Each thread checks  the main row
# with one of the element of the columns pointed by the row.
# The columns aren't read at once but in strided fashion.
function countTri_elem_per_th_limit_nt(csr_rows, col_indx, out_sum, max_mem_nnz, max_nnz)
#function countTri(out_sum)
    bid = blockIdx().x + (blockIdx().y-1)*gridDim().x + (blockIdx().z-1)*gridDim().x*gridDim().y
    rows = size(csr_rows)[1]
    nnz = size(col_indx)[1]
    tid = threadIdx().x + (threadIdx().y-1)*blockDim().x
    stride = blockDim().x*blockDim().y
    if bid > rows
        return nothing
    end

    sh_row = @cuDynamicSharedMem(Int32, max_mem_nnz)
    sh_cols = @cuDynamicSharedMem(Int32, max_nnz + CUDA.max(Int32(stride), Int32(max_nnz)),
                                sizeof(Int32)*max_mem_nnz)
    sh_sum = @cuDynamicSharedMem(Int32, blockDim().y,
            (max_nnz + CUDA.max(Int32(stride), Int32(max_nnz)) + max_mem_nnz)*sizeof(Int32))

    @inbounds row_start =  csr_rows[bid]
    @inbounds row_end = (bid == rows) ?  nnz+1 : csr_rows[bid + 1]

    len = row_end-row_start
    for s = 0:stride:(len-1)
        if tid + s <= len
            @inbounds sh_row[tid+s] = col_indx[row_start + tid - 1 + s]
        end
    end
    sync_threads()
    #
    #Read all the other rows
    total_elems = 0
    sum = 0
    for i = 1:len
        @inbounds col = sh_row[i]
        @inbounds row_start = csr_rows[col]
        @inbounds row_end = (col == rows) ?  nnz+1 : csr_rows[col + 1]
        templen = row_end-row_start
        sync_threads()
        for s = 0:stride:(templen-1)
            if s + tid <= templen
                @inbounds sh_cols[total_elems + s + tid] = col_indx[row_start + s + tid - 1]
            end
        end
        total_elems  += templen
        sync_threads()
        if total_elems >= stride || i == len
            for s = 0:stride:(total_elems-1)
                if tid + s <= total_elems
                    for l = 1:len
                        # sum += sh_row[(l%len)+1] == sh_cols[s + tid]
                        @inbounds sum += sh_row[((tid+l-2)%len)+1] == sh_cols[s + tid]
                    end
                end
            end
            total_elems = 0
        end
    end

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


function CountTriangles_ElemPerThLimit_nt(csr_rows, col_indx, max_per_row; th_groups=2)
    dev_csr_rows = CuArray(csr_rows)
    dev_col_indx = CuArray(col_indx)
    dev_out_sum = CUDA.zeros(Int32, size(csr_rows)[1])
    thread_groups = th_groups
    max_mem_nnz = 0
    while max_mem_nnz < max_per_row
        max_mem_nnz += 32
    end
    shmem = sizeof(Int32)*(thread_groups + max_mem_nnz + max_per_row + max(thread_groups*32,max_per_row))
    blocks = Int64(floor((size(csr_rows)[1])/(512*512)+ 1))
    println("3 Total blocks: ", 512*512*blocks, " vs rows: ", size(csr_rows)[1])
    println("\t Threads: 32*", thread_groups)
    @cuda(blocks=(512,512,blocks), threads=(32, thread_groups),
        shmem=shmem,
        countTri_elem_per_th_limit_nt(dev_csr_rows, dev_col_indx, dev_out_sum,
                                max_mem_nnz, max_per_row))
    out_sum = Array(dev_out_sum)
    # println(dev_out_sum)
    return reduce(+, out_sum, init=0)
end
