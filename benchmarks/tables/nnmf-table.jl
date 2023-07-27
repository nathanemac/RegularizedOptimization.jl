include("regulopt-tables.jl")

Random.seed!(1234)
m, n, k = 100, 50, 5
model, nls_model, A, selected = nnmf_model(m, n, k)
f = LBFGSModel(model)
λ = 1.0e-1 # norm(grad(model, rand(model.meta.nvar)), Inf) / 100
h = NormL0(λ)
ν = 1.0e-3
ϵ = 1.0e-5
ϵi = 1.0e-3
ϵri = 1.0e-6
maxIter = 500
maxIter_inner = 100
verbose = 0 #10
options =
  ROSolverOptions(ν = ν, ϵa = ϵ, ϵr = ϵ, verbose = verbose, maxIter = maxIter, spectral = true)
options_nrTR = ROSolverOptions(
  ν = ν,
  ϵa = ϵ,
  ϵr = ϵ,
  verbose = verbose,
  maxIter = maxIter,
  spectral = true,
  reduce_TR = false,
)
options2 = ROSolverOptions(spectral = false, psb = true, ϵa = ϵi, ϵr = ϵri, maxIter = maxIter_inner)
options2_nrTR = ROSolverOptions(
  spectral = false,
  psb = true,
  ϵa = ϵi,
  ϵr = ϵri,
  maxIter = maxIter_inner,
  reduce_TR = false,
)
options3 =
  ROSolverOptions(spectral = false, psb = false, ϵa = ϵi, ϵr = ϵri, maxIter = maxIter_inner)
options3_nrTR = ROSolverOptions(
  spectral = false,
  psb = false,
  ϵa = ϵi,
  ϵr = ϵri,
  maxIter = maxIter_inner,
  reduce_TR = false,
)
options4 = ROSolverOptions(spectral = true, ϵa = ϵi, ϵr = ϵri, maxIter = maxIter_inner)
options4_nrTR =
  ROSolverOptions(spectral = true, ϵa = ϵi, ϵr = ϵri, maxIter = maxIter_inner, reduce_TR = false)
options5 = ROSolverOptions(
  ν = ν,
  ϵa = ϵ,
  ϵr = ϵ,
  verbose = verbose,
  maxIter = maxIter,
  spectral = false,
  psb = true,
)
options5_nrTR = ROSolverOptions(
  ν = ν,
  ϵa = ϵ,
  ϵr = ϵ,
  verbose = verbose,
  maxIter = maxIter,
  spectral = false,
  psb = true,
  reduce_TR = false,
)
options6 = ROSolverOptions(
  ν = ν,
  ϵa = ϵ,
  ϵr = ϵ,
  verbose = verbose,
  maxIter = maxIter,
  spectral = false,
  psb = false,
)
options6_nrTR = ROSolverOptions(
  ν = ν,
  ϵa = ϵ,
  ϵr = ϵ,
  verbose = verbose,
  maxIter = maxIter,
  spectral = false,
  psb = false,
  reduce_TR = false,
)
options7 = ROSolverOptions(
  ν = ν,
  ϵa = ϵ,
  ϵr = ϵ,
  verbose = verbose,
  maxIter = maxIter,
  spectral = false,
  psb = false,
  andrei = false
)

options7_nrTR = ROSolverOptions(
  ν = ν,
  ϵa = ϵ,
  ϵr = ϵ,
  verbose = verbose,
  maxIter = maxIter,
  spectral = false,
  psb = false,
  andrei = false,
  reduce_TR = false,
)

options8 = ROSolverOptions(
  ν = ν,
  ϵa = ϵ,
  ϵr = ϵ,
  verbose = verbose,
  maxIter = maxIter,
  spectral = false,
  psb = false,
  andrei = false,
  wolk = true
)

options8_nrTR = ROSolverOptions(
  ν = ν,
  ϵa = ϵ,
  ϵr = ϵ,
  verbose = verbose,
  maxIter = maxIter,
  spectral = false,
  psb = false,
  andrei = false,
  wolk = true,
  reduce_TR = false,
)

solvers = [:R2, :R2DH, :R2DH, :R2DH, :TRDH, :TRDH, :TRDH, :TRDH, :TRDH, :TRDH, :TRDH, :TRDH, :TRDH, :TRDH, :TR, :TR, :TR, :TR, :TR, :TR, :TR, :TR, :TR, :TR, :TR, :TR, :TR, :TR]
subsolvers =
  [:None, :None, :None, :None, :None, :None, :None, :None, :None, :None, :None, :None, :None, :None, :R2, :TRDH, :TRDH, :TRDH, :TRDH, :TRDH, :TRDH, :TRDH, :TRDH, :TRDH, :TRDH, :R2DH, :R2DH, :R2DH]
solver_options = [
  options,
  options,
  options7,
  options8,
  options,
  options_nrTR,
  options5,
  options5_nrTR,
  options6,
  options6_nrTR,
  options7,
  options7_nrTR,
  options8,
  options8_nrTR,
  options,
  options,
  options,
  options,
  options,
  options,
  options,
  options,
  options,
  options,
  options,
  options,
  options,
  options,
]
subsolver_options = [
  options2,
  options2,
  options2,
  options2,
  options2,
  options2,
  options2,
  options2,
  options2,
  options2,
  options2,
  options2,
  options2,
  options2,
  options2,
  options2,
  options2_nrTR,
  options3,
  options3_nrTR,
  options4,
  options4_nrTR,
  options7,
  options7_nrTR,
  options8,
  options8_nrTR,
  options8,
  options7,
  options,
] # n'importe lequel si subsolver = :None
subset = 1:length(solvers)

benchmark_table(
  f,
  selected,
  [],
  h,
  λ,
  solvers[subset],
  subsolvers[subset],
  solver_options[subset],
  subsolver_options[subset],
  "NNMF with m = $m, n = $n, k = $k, ν = $ν, λ = $λ",
  tex = false,
);
