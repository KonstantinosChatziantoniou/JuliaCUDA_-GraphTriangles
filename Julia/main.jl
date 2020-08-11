using Statistics
include("./preprocessing.jl")
include("./kernel_row_per_th.jl")
include("./kernel_elem_per_th.jl")
include("./kernel_elem_per_th_limit.jl")
include("./nt_kernel_row_per_th.jl")
include("./nt_kernel_elem_per_th.jl")
include("./nt_kernel_elem_per_th_limit.jl")
df = "../auto.mtx"
if length(ARGS) >= 1
    df = ARGS[1]
end
r, c = ReadData(df)
max_nnz_per_row = max(r...)
mean_nnz = mean(r)
intg_r = GetIntegralCsrRows(r)

# add @profile if you use nvprof/nsys or @benchmark for measuring time.
println("Max: ", max_nnz_per_row, " Mean: ", mean_nnz)
t = CountTriangles_RowPerTh(intg_r, c, max_nnz_per_row)
println("tr rpt: ",t)

t = CountTriangles_ElemPerTh(intg_r, c, max_nnz_per_row)
println("tr ept: ",t)

t = CountTriangles_ElemPerThLimit(intg_r, c, max_nnz_per_row; th_groups=1)
println("tr eptl32: ",t)

t = CountTriangles_ElemPerThLimit(intg_r, c, max_nnz_per_row; th_groups=2)
println("tr eptl64: ",t)

## NO TYPES
t = CountTriangles_RowPerTh_nt(intg_r, c, max_nnz_per_row)
println("tr rpt: ",t)

t = CountTriangles_ElemPerTh_nt(intg_r, c, max_nnz_per_row)
println("tr ept: ",t)

t = CountTriangles_ElemPerThLimit_nt(intg_r, c, max_nnz_per_row; th_groups=1)
println("tr eptl32: ",t)

t = CountTriangles_ElemPerThLimit_nt(intg_r, c, max_nnz_per_row; th_groups=2)
println("tr eptl64: ",t)
