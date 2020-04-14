using CUDAnative, CUDAdrv, CuArrays



function countTri(csr_rows, col_indx, out_sum, max_nnz)
#function countTri(out_sum)
    bid = (blockIdx().x +
            (blockIdx().y-1)*gridDim().x +
            (blockIdx().z-1)*gridDim().x*gridDim().y)
    rows = size(csr_rows)[1]
    if bid > rows
        return nothing
    end
    tid = threadIdx().x #+ (threadIdx().y-1)*blockDim().x
    threads = blockDim().x #*blockDim().y
    sh_sum = @cuDynamicSharedMem(Int32, threads)
    sh_row = @cuDynamicSharedMem(Int32, max_nnz, sizeof(Int32)*(threads))
    sh_len = @cuDynamicSharedMem(Int32, max_nnz, sizeof(Int32)*(threads + max_nnz))
    sh_cols = @cuDynamicSharedMem(Int32, max_nnz*max_nnz, sizeof(Int32)*(threads + 2*max_nnz))
    @inbounds sh_sum[tid] = 0
    row_start = csr_rows[bid]
    row_end = Int32(0)
    if bid == rows
        row_end = size(col_indx)[1]
    else
        @inbounds row_end = csr_rows[bid + 1] - 1             # include the row end to loop
    end
    # Read the current row of the block
    sync_threads()
    len = row_end-row_start + 1
    for j = 0:32:(len-1)
        if j + tid  <= len
            @inbounds sh_row[j+tid] = col_indx[row_start+j+tid-1]
        end
    end
    sync_threads()

    # #Read all the other rows
    start_save = 0
    for i = 1:len
            col = sh_row[i]
            row_start = csr_rows[col]
            row_end = Int32(0)
            if col == rows
                row_end = size(col_indx)[1]
            else
                row_end = csr_rows[col + 1] - 1
            end
            temp_len = row_end-row_start + 1
            sync_threads()
            for j = 0:32:(temp_len-1)
                if row_start + j + tid - 1 <= row_end
                    sh_cols[start_save + j + tid] = col_indx[row_start + j + tid - 1]
                end
            end
            sync_threads()
            start_save += temp_len
    end


    for i = 0:threads:(start_save-1)
        if i + tid > start_save
            break
        end
        for j = 1:len
            sh_sum[tid] += sh_row[j] == sh_cols[i + tid]
        end
    end
    sync_threads()
    if tid == 1
        for i = 2:threads
            sh_sum[1] += sh_sum[i];
        end
        out_sum[bid] = sh_sum[1];
    end
    return nothing
end


function CountTriangles(csr_rows, col_indx, max_per_row; th=32)
    dev_csr_rows = CuArray(csr_rows)
    dev_col_indx = CuArray(col_indx)
    dev_out_sum = CuArrays.zeros(Int32, size(csr_rows)[1])
    threads = th
    blocks = Int64(floor((size(csr_rows)[1])/(512*512)+ 1))
    println("2 Total blocks: ", 512*512*blocks, " vs rows: ", size(csr_rows)[1])
    println("\t Threads: ", threads)
    @cuda(blocks=(512,512,blocks), threads=threads,
        shmem=sizeof(Int32)*(threads + (max_per_row+2)*max_per_row),
        countTri(dev_csr_rows, dev_col_indx, dev_out_sum, max_per_row))
    out_sum = Array(dev_out_sum)
    # println(dev_out_sum)
    return reduce(+, out_sum, init=0)
end
