using SparseArrays
include("./src/preprocessing.jl")
r, c = SparseReadData("./data/auto.mtx")

sr = sparse(r, c, ones(length(r)), max(c...),max(c...))
num_tr = sum((sr*sr).*sr)
