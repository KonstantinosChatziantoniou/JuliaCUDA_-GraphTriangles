using Statistics

"""
    ReadData(f)

Reads a .mtx file that contains a
sparse undirected graph.

Returns the number of non zero elements per row
and the column index of the non zero elements.
"""
function ReadData(f)
    println("Reading: ",f)
    dfile = open(f)
    header = readline(dfile)
    # Skipping comments
    while true
        if header[1] == '%'
            header = readline(dfile)
        else
            break
        end
    end
    header =  map(x -> parse(Int32,x),split(header))
    col_indx = Array{Int32,1}(undef,header[3])
    csr_rows = zeros(Int32, header[1])
    for i  = 1:header[3]
        line = (readline(dfile))
        mid = findfirst(" ", line)[1]
        col_indx[i] = parse(Int32, line[1:(mid-1)])
        csr_rows[parse(Int32, line[(mid+1):end])] += 1
    end
    return csr_rows, col_indx
end

function SparseReadData(f)
    println("Reading: ",f)
    dfile = open(f)
    header = readline(dfile)
    # Skipping comments
    while true
        if header[1] == '%'
            header = readline(dfile)
        else
            break
        end
    end
    header =  map(x -> parse(Int32,x),split(header))
    col_indx = Array{Int32,1}(undef,header[3])
    row_indx = Array{Int32,1}(undef,header[3])
    for i  = 1:header[3]
        line = (readline(dfile))
        mid = findfirst(" ", line)[1]
        col_indx[i] = parse(Int32, line[1:(mid-1)])
        row_indx[i] = parse(Int32, line[(mid+1):end])
    end
    return row_indx, col_indx
end
function GetIntegralCsrRows(csr_rows)
    int_csr_rows = similar(csr_rows)
    int_csr_rows[1] = 1

    for i = 2:length(csr_rows)
        @inbounds int_csr_rows[i] = csr_rows[i-1] + int_csr_rows[i-1]
    end
    return int_csr_rows
end












#
# function SlowReadData(f="auto.mtx")
#     println("Slow Reading: ",f)
#     dfile = open(f)
#     header = map(x -> parse(Int32,x),split(readline(dfile)))
#     col_indx = Array{Int32,1}(undef,header[3])
#     csr_rows = zeros(Int32, header[1])
#     for i  = 1:header[3]
#         t = map(x -> parse(Int32,x),split(readline(dfile)))
#         col_indx[i] = t[1]
#         csr_rows[t[2]] += 1
#     end
#
#     return csr_rows, col_indx
# end
