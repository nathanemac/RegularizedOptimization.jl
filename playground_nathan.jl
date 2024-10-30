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
using ProxTV

# include("src/utils.jl")
include("src/input_struct.jl")
include("src/R2N.jl")
include("src/iR2N.jl")
include("src/utils_iR2N.jl")
include("src/R2_alg.jl")
include("src/R2DH.jl")
include("src/utils.jl")

nlp = ADNLPModel(x -> (1-x[1])^2 + 100(x[1]-x[2]^2)^2, [-1., 1.])

h1 = NormL2(1.0)
options = ROSolverOptions()
res_r2n = R2N(nlp, h1, options)
res_r2n.solution


options = ROSolverOptions()
res_ir2n_1 = iR2N(nlp, h1, options)
res_ir2n_1.solution

hp = NormLp(1.0, 2.0)
options = ROSolverOptions()
res_ir2n_p = iR2N(nlp, hp, options)
res_ir2n_p.solution

# TODO next: understand why neg_tol hits with p ≠ something known






h = NormLp(0.1, 1.0)
ψ = shifted(h, [1.39, -1.])
shift!(ψ, [0.5, 1.5])
x0 = [1., 1.]
y = similar(x0)
prox!(y, ψ, x0, 1.0)

h = NormL1(0.1)
ψ = shifted(h, [1., 1.])
shift!(ψ, [0.5, 1.5])
x0 = [1., 1.]
y = similar(x0)
prox!(y, ψ, x0, 1.0)
