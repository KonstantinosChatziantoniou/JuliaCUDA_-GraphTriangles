#using Statistics, BenchmarkTools
include("./src/preprocessing.jl")
include("./src/kernel2.jl")


r, c = ReadData("./data/auto.mtx")
max_nnz_per_row = max(r...)
intg_r = GetIntegralCsrRows(r)

# add @profile if you use nvprof/nsys or @benchmark for measuring time.
CountTriangles(intg_r, c, max_nnz_per_row)
