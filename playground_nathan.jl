using LinearOperators
using NLPModels
using NLPModelsModifiers
using ProximalOperators
using RegularizedProblems
using ShiftedProximalOperators
using SolverCore
using TSVD
using LinearAlgebra
using Logging
using Printf
using ADNLPModels

include("src/utils.jl")
include("src/input_struct.jl")
include("src/R2N.jl")
include("src/R2_alg.jl")
include("src/R2DH.jl")

nlp = ADNLPModel(x -> (1-x[1])^2 + 100(x[1]-x[2]^2)^2, [-1., 1.])
h = NormL1(1.0)
options = ROSolverOptions()
res_r2n = R2N(nlp, h, options)
res_r2n.solution