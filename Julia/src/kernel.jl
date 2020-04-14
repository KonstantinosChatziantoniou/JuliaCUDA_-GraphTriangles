using CUDAnative, CUDAdrv, CuArrays

function countTri(csr_rows, col_indx, out_sum)
#function countTri(out_sum)
    bid = blockIdx().x + (blockIdx().y-1)*gridDim().x + (blockIdx().z-1)*gridDim().x*gridDim().y
    rows = size(csr_rows)[1]
    if bid > rows
        return nothing
    end
    tid = threadIdx().x #+ (threadIdx().y-1)*blockDim().x
    max_nnz = blockDim().x#*blockDim().y
    sh_sum = @cuDynamicSharedMem(Int32, max_nnz)
    sh_row = @cuDynamicSharedMem(Int32, max_nnz, max_nnz*sizeof(Int32))
    sh_len = @cuDynamicSharedMem(Int32, max_nnz, 2*max_nnz*sizeof(Int32))
    sh_cols = @cuDynamicSharedMem(Int32, (max_nnz,max_nnz), 3*max_nnz*sizeof(Int32))
    sh_sum[tid] = 0
    row_start = csr_rows[bid]
    row_end = Int32(0)
    if bid == rows
        row_end = size(col_indx)[1]
    else
        row_end = csr_rows[bid + 1] - 1             # include the row end to loop
    end
    # Read the current row of the block
    sync_threads()
    sh_row[tid] = -10
    if row_start + tid - 1 <= row_end
        sh_row[tid] = col_indx[row_start + tid - 1]
    end
    sync_threads()
    len = row_end-row_start + 1

    # #Read all the other rows
    for i = 1:len
            col = sh_row[i]

            row_start = csr_rows[col]
            row_end = Int32(0)
            if col == rows
                row_end = size(col_indx)[1]
            else
                row_end = csr_rows[col + 1] - 1
            end
            sh_len[i] = row_end-row_start + 1
            sync_threads()
            if row_start + tid - 1 <= row_end
                sh_cols[i, tid] = col_indx[row_start + tid - 1]
            end
            sync_threads()
    end

    # Sum
    if tid <= len
        a  = 1;
        b  = 1;
        sum = 0;
        while true
            if a > len || b > sh_len[tid]
                break;
            end

            b1 = sh_row[a] == sh_cols[tid,b];
            b2 = sh_row[a] > sh_cols[tid,b];
            b3 = sh_row[a] < sh_cols[tid,b];

            a = a + b1 + b3
            b = b + b1 + b2
            sum = sum + b1
        end

        sh_sum[tid] = sum;
    end
    if tid == 1
        sum = (0);
        for i = 1:len
            sum += sh_sum[i];
        end
        out_sum[bid] = sum;
    end
    return nothing
end


function CountTriangles(csr_rows, col_indx, max_per_row)Int64
    dev_csr_rows = CuArray(csr_rows)
    dev_col_indx = CuArray(col_indx)
    dev_out_sum = CuArrays.zeros(Int32, size(csr_rows)[1])

    blocks = Int64(floor((size(csr_rows)[1])/(512*512)+ 1))
    println("Total blocks: ", 512*512*blocks, " vs rows: ", size(csr_rows)[1])
    println("\t Threads: ", max_per_row)
    @cuda(blocks=(512,512,blocks), threads=max_per_row,
        shmem=(sizeof(Int32)*(max_per_row+2)*max_per_row),
        countTri(dev_csr_rows, dev_col_indx, dev_out_sum))
    out_sum = Array(dev_out_sum)
    #println(dev_out_sum)
    return reduce(+, out_sum, init=0)


end
