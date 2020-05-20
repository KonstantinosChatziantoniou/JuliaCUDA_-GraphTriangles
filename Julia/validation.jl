using SparseArrays
include("./src/preprocessing.jl")
r, c = SparseReadData("../auto.mtx")

sr = sparse(r, c, ones(length(r),1), max(c...),max(c...))
num_tr = sum((sr*sr).*sr)
