# JuliaCUDA_GraphTriangles

An implementation of the graph triangle algorithm for gpgpu, using `CUDAnative` and `CuArrays` for `Julia`.
The aim is to show that Julia for CUDA programming  can achieve not only faster development time, but also equivelent or even
better execution time than `C`(the goto language for CUDA kernel programming along with C++).

## Implementation
This implementation is based on the formula `sum((A*A).*A)/6, where A is the adjacency matrix` for counting the triangles.
We assume that the graph is symmetric and non weighted.
The above formula can be simplified to simple sparse vector multiplication and tracking only the sum of the result.
[https://github.com/KonstantinosChatziantoniou/GraphTrianglesCounting/blob/master/main.pdf] contains the mathematical analysis for the above simplification.



## File structure

```Julia/```

+ `main.jl` Preprocesses the points and calls the kernels. Modify for different dataset and differrent kernel implementation. 
Returns Distances and Indices for each Nearest Neighbour

+ `validation.jl` Does everything `main.jl` does. It validates the results using the `NearestNeighbour` package.

```Julia/Code/```

Contains the preprocessing function definitions and the kernel definitions.


## C implementaion

The `C` implementation of the grid knn algorithm is taken from this repo [https://github.com/KonstantinosChatziantoniou/GraphTrianglesCounting]


```Notebook```

Containg a Jyputer notebook to be executed in google colab for execution time benchmarking. The notebook includes scripts for 
installing cuda and julia in goole colab.


## Execution

`Julia/main.jl` and `Julia/validation.jl` are ready to be executed as standalone scripts.

For the`C` implementation

                      run `make`
                      run `./mainProgram N B` where N is the number of points (2^N) 
                          and B is the number of blocks per dimension (2^B,2^B,2^B)

For the `jl.ipynb`, upload as a new Notebook to google colab and follow the instructions inside

## Results
Google colab: Tesla P4
![res.png](https://raw.githubusercontent.com/KonstantinosChatziantoniou/JuliaCUDA_GraphTriangles/tree/master/Res/rel_speed.png)
![res2.png](https://raw.githubusercontent.com/KonstantinosChatziantoniou/JuliaCUDA_GraphTriangles/tree/master/Res/time_auto.png)
![res3.png](https://raw.githubusercontent.com/KonstantinosChatziantoniou/JuliaCUDA_GraphTriangles/tree/master/Res/time_del.png)
