using LinearOperators
using NLPModels
using NLPModelsModifiers
using ProximalOperators
using RegularizedProblems
using ShiftedProximalOperators
using SolverCore
using LinearAlgebra
using Logging
using Printf
using Arpack
#added :
using ADNLPModels
using ProxTV
using Images
using Random
using Plots
using DataFrames

# include("src/utils.jl")
include("src/input_struct.jl")
include("src/R2N.jl")
include("src/iR2N.jl")
include("src/utils_iR2N.jl")
include("src/R2_alg.jl")
include("src/R2DH.jl")
include("src/utils.jl")

nlp = ADNLPModel(x -> (1-x[1])^2 + 100(x[1]-x[2]^2)^2, [-1., 1.])

h1 = NormL1(1.0)
options = ROSolverOptions()
res_r2n = R2N(nlp, h1, options)
res_r2n.solution


options = ROSolverOptions(dualGap = 1e-8)
hp = NormLp(1.0, 1.59)
res_ir2n_1 = iR2N(nlp, hp, options)
res_ir2n_1.solution

hp = NormLp(1.0, 1.14)
options = ROSolverOptions(neg_tol = 0.001, dualGap = 1e-9)
res_ir2n_p = iR2N(nlp, hp, options)
res_ir2n_p.solution

hp = NormTVp(1.0, 1.14)
options = ROSolverOptions(neg_tol = 0.0001, dualGap = 1e-6)
res_ir2n_p = iR2N(nlp, hp, options)
res_ir2n_p.solution


# tests sur des regularized RegularizedProblems
Random.seed!(123)
model, nls_model, sol = bpdn_model()
bpdn_bfgs = LBFGSModel(model)

h1 = NormL1(0.01)
options = ROSolverOptions(verbose=100, ϵa = 1e-6, ϵr = 1e-6)
res_r2n = R2N(bpdn_bfgs, h1, options)

h1p = NormLp(0.01, 1.0)
options = ROSolverOptions(verbose=100, ϵa = 1e-6, ϵr = 1e-6, dualGap = 1e-6)
res_ir2n = iR2N(bpdn_bfgs, h1p, options)

h1p =  NormLp(0.01, 1.0)
options = ROSolverOptions(verbose=10, ϵa = 1e-6, ϵr = 1e-6, dualGap = 1e-6)
res_ir2n_p = iR2N(bpdn_bfgs, h1p, options)

isapprox(res_r2n.solution, res_ir2n.solution, atol=1e-5)
isapprox(res_r2n.solution, res_ir2n_p.solution, atol=1e-5)
isapprox(res_ir2n.solution, res_ir2n_p.solution, atol=1e-5)


# en faisant varier dualGap
model, nls_model, sol = bpdn_model()
bpdn_bfgs = LBFGSModel(model)
h1 = NormLp(0.01, 1.43)
results = DataFrame(dualGap=Float64[], iter=Union{Int, String}[], objective=Union{Float64, String}[])

# Boucle sur les différentes valeurs de dualGap
for dG in [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
  options = ROSolverOptions(verbose=2, ϵa=1e-5, ϵr=1e-5, dualGap=dG)
  try
      # Appel de la fonction iR2N
      res_ir2n = iR2N(bpdn_bfgs, h1, options)
      
      # Ajout des résultats dans le dataframe
      push!(results, (dualGap=dG, iter=res_ir2n.iter, objective=res_ir2n.objective))
  catch e
      # En cas d'erreur, ajout d'une ligne avec "ERROR" pour les valeurs
      push!(results, (dualGap=dG, iter="ERROR", objective="ERROR"))
  end
end

# Affichage du dataframe (optionnel, sinon pour sauvegarder dans un fichier CSV)
println(results)



# Demo sur un sinus bruité
using Random
using Plots
Random.seed!(123)
n = 100
t = range(0, 2π, length=n)
y = sin.(t)
y_noisy = sin.(t) + 0.1 * randn(n)
plot(t, y, label="Original signal")
plot!(t, y_noisy, label="Noisy signal")

H = NormTVp(0.01, 1.0)
nlp = ADNLPModel(x ->  0.5 * norm(x - y)^2, y_noisy)

options = ROSolverOptions(verbose=10, ϵa = 1e-6, ϵr = 1e-6, dualGap = 1e-6)
res_r2n = R2N(nlp, H, options)
plot!(t, res_r2n.solution, label="R2N - denoised signal")

res_ir2n = iR2N(nlp, H, options)
plot!(t, res_ir2n.solution, label="iR2N - denoised signal")

# Hp = NormTVp(0.01, 1.53)
# res_ir2n = iR2N(nlp, Hp, options)
# plot!(t, res_ir2n.solution, label="iR2N - denoised signal")


# Avec une image
img = load("/Users/nathanallaire/Desktop/cameraman.jpg")
gray_img = channelview(colorview(Gray, img))
display(img)
img1d = Float64.(vec(gray_img))

sigma = 0.1  # Écart-type du bruit
noise = sigma * randn(size(gray_img))  # Générer un bruit gaussien de la même taille que l'image
noisy_img = clamp01.(gray_img .+ noise)  # Clamp pour rester dans les valeurs valides [0, 1]
display(noisy_img)
img1d_noisy = Float64.(vec(noisy_img))

H = NormTVp(0.01, 1.0)
nlp = ADNLPModel(x ->  0.5 * norm(x - img1d)^2, img1d_noisy)


# tests
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
