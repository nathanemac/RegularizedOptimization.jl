abstract type InexactShiftedProximableFunction end

### NormLp and ShiftedNormLp Implementation

"""
    NormLp(λ::Real or AbstractArray, p::Real)

Represents the Lp norm with parameter `p` and scaling factor `λ`.
"""
struct NormLp{T1,T2} 
    λ::T1
    p::T2

    function NormLp(λ::T1, p::T2) where {T1,T2}
        if λ isa Real
            λ < 0 && error("λ must be nonnegative")
        elseif λ isa AbstractArray
            eltype(λ) <: Real || error("Elements of λ must be real")
            any(λ .< 0) && error("All elements of λ must be nonnegative")
        else
            error("λ must be a real scalar or array")
        end

        p >= 1 || error("p must be greater than or equal to one")
        new{T1,T2}(λ, p)
    end
end

"""
    prox!(y, h::NormLp, q, ν; dualGap=1e-5)

Evaluates inexactly the proximity operator of a Lp norm object.
The duality gap at the solution is guaranteed to be less than `dualGap`.

Inputs:
    - `y`: Array in which to store the result.
    - `h`: NormLp object.
    - `q`: Vector to which the proximity operator is applied.
    - `ν`: Scaling factor.
    - `dualGap`: Desired quality of the solution in terms of duality gap (default `1e-5`).
"""
function prox!(
        y::AbstractArray,
        h::NormLp,
        q::AbstractArray,
        ν::Real;
        dualGap::Real = 1e-5,
)
    
    n = length(y)
    ws = ProxTV.newWorkspace(n)

    # Allocate info array (based on C++ code)
    info = zeros(Float64, 3)

    # Adjust lambda to account for ν (multiply λ by ν)
    lambda_scaled = h.λ * ν

    positive = Int32(all(v -> v >= 0, y) ? 1 : 0)

    ProxTV.PN_LPp(q, lambda_scaled, y, info, n, h.p, ws, positive, dualGap)

    return y
end

# Allows NormLp objects to be called as functions
function (h::NormLp)(x::AbstractArray)
    return h.λ * ProxTV.LPnorm(x, length(x), h.p)
end

"""
    ShiftedNormLp

A mutable struct representing a shifted NormLp function.
"""
mutable struct ShiftedNormLp{
    R<:Real,
    T<:Real,
    V0<:AbstractVector{R},
    V1<:AbstractVector{R},
    V2<:AbstractVector{R},
} <: InexactShiftedProximableFunction
    h::NormLp{R,T}
    xk::V0
    sj::V1
    sol::V2
    shifted_twice::Bool
    xsy::V2

    function ShiftedNormLp(
        h::NormLp{R,T},
        xk::AbstractVector{R},
        sj::AbstractVector{R},
        shifted_twice::Bool,
    ) where {R<:Real,T<:Real}
        sol = similar(xk)
        xsy = similar(xk)
        new{R,T,typeof(xk),typeof(sj),typeof(sol)}(h, xk, sj, sol, shifted_twice, xsy)
    end
end

"""
    shifted(h::NormLp, xk::AbstractVector)

Creates a ShiftedNormLp object with initial shift `xk`.
"""
shifted(h::NormLp{R,T}, xk::AbstractVector{R}) where {R<:Real,T<:Real} =
    ShiftedNormLp(h, xk, zero(xk), false)

"""
    shifted(ψ::ShiftedNormLp, sj::AbstractVector)

Creates a ShiftedNormLp object by adding a second shift `sj`.
"""
shifted(
    ψ::ShiftedNormLp{R,T,V0,V1,V2},
    sj::AbstractVector{R},
) where {
    R<:Real,
    T<:Real,
    V0<:AbstractVector{R},
    V1<:AbstractVector{R},
    V2<:AbstractVector{R},
} = ShiftedNormLp(ψ.h, ψ.xk, sj, true)

# Functions to get the name, expression, and parameters of the function
fun_name(ψ::ShiftedNormLp) = "shifted Lp norm"
fun_expr(ψ::ShiftedNormLp) = "t ↦ λ * ‖xk + sj + t‖ₚ"
fun_params(ψ::ShiftedNormLp) = "xk = $(ψ.xk)\n" * " "^14 * "sj = $(ψ.sj)"

"""
    shift!(ψ::ShiftedNormLp, shift::AbstractVector)

Updates the shift of a ShiftedNormLp object.
"""
function shift!(ψ::ShiftedNormLp, shift::AbstractVector{R}) where {R<:Real}
    if ψ.shifted_twice
        ψ.sj .= shift
    else
        ψ.xk .= shift
    end
    return ψ
end

# Allows ShiftedNormLp objects to be called as functions
function (ψ::ShiftedNormLp)(y::AbstractVector)
    @. ψ.xsy = ψ.xk + ψ.sj + y
    return ψ.h(ψ.xsy)
end

"""
    prox!(y, ψ::ShiftedNormLp, q, ν; dualGap=1e-5)

Evaluates inexactly the proximity operator of a shifted Lp norm.
The duality gap at the solution is guaranteed to be less than `dualGap`.

Inputs:
    - `y`: Array in which to store the result.
    - `ψ`: ShiftedNormLp object.
    - `q`: Vector to which the proximity operator is applied.
    - `ν`: Scaling factor.
    - `dualGap`: Desired quality of the solution in terms of duality gap (default `1e-5`).
"""
function prox!(
    y::AbstractArray,
    ψ::ShiftedNormLp,
    q::AbstractArray,
    ν::Real;
    dualGap::Real = 1e-5,
)
    n = length(y)
    ws = ProxTV.newWorkspace(n)

    # Allocate info array (based on C++ code)
    info = zeros(Float64, 3)

    # Compute y_shifted = xk + sj + q
    y_shifted = ψ.xk .+ ψ.sj .+ q

    # Adjust lambda to account for ν (multiply λ by ν)
    lambda_scaled = ψ.h.λ * ν

    # Allocate the x vector to store the intermediate solution
    x = zeros(n)

    positive = Int32(all(v -> v >= 0, y_shifted) ? 1 : 0)

    if ψ.h.p == 1
        ProxTV.PN_LP1(y_shifted, lambda_scaled, x, info, n)
    elseif ψ.h.p == 2
        ProxTV.PN_LP2(y_shifted, lambda_scaled, x, info, n)
    elseif ψ.h.p == Inf
        ProxTV.PN_LPi(y_shifted, lambda_scaled, x, info, n, ws)
    else
        ProxTV.PN_LPp(y_shifted, lambda_scaled, x, info, n, ψ.h.p, ws, positive, dualGap)
    end
    # Compute s = x - xk - sj
    s = x .- ψ.xk .- ψ.sj

    # Store the result in y
    y .= s

    return y
end



### NormTVp and ShiftedNormTVp Implementation

"""
    NormTVp(λ::Real or AbstractArray, p::Real)

Represents the Total Variation (TV) norm with parameter `p` and scaling factor `λ`.
"""
struct NormTVp{T1,T2}
    λ::T1
    p::T2

    function NormTVp(λ::T1, p::T2) where {T1,T2}
        if λ isa Real
            λ < 0 && error("λ must be nonnegative")
        elseif λ isa AbstractArray
            eltype(λ) <: Real || error("Elements of λ must be real")
            any(λ .< 0) && error("All elements of λ must be nonnegative")
        else
            error("λ must be a real scalar or array")
        end

        p >= 1 || error("p must be greater than or equal to one")
        new{T1,T2}(λ, p)
    end
end

"""
    TVp_norm(x::AbstractArray, p::Real)

Computes the TVp norm of vector `x` with parameter `p`.
"""
function TVp_norm(x::AbstractArray, p::Real)
    n = length(x)
    tvp_sum = 0.0
    for i = 1:(n-1)
        tvp_sum += abs(x[i+1] - x[i])^p
    end
    return tvp_sum^(1 / p)
end

"""
    prox!(y, h::NormTVp, q, ν; dualGap=1e-5)

Evaluates inexactly the proximity operator of a TVp norm object.
The duality gap at the solution is guaranteed to be less than `dualGap`.

Inputs:
    - `y`: Array in which to store the result.
    - `h`: NormTVp object.
    - `q`: Vector to which the proximity operator is applied.
    - `ν`: Scaling factor.
    - `dualGap`: Desired quality of the solution in terms of duality gap (default `1e-5`).
"""
function prox!(
        y::AbstractArray,
        h::NormTVp,
        q::AbstractArray,
        ν::Real;
        dualGap::Real = 1e-5,
)
    
    n = length(y)
    ws = ProxTV.newWorkspace(n)

    # Allocate info array (based on C++ code)
    info = zeros(Float64, 3)

    # Adjust lambda to account for ν (multiply λ by ν)
    lambda_scaled = h.λ * ν

    ProxTV.TV(q, lambda_scaled, y, info, n, h.p, ws, objGap = dualGap)

    return y
end

# Allows NormTVp objects to be called as functions
function (h::NormTVp)(x::AbstractArray)
    return h.λ * TVp_norm(x, h.p)
end

"""
    ShiftedNormTVp

A mutable struct representing a shifted NormTVp function.
"""
mutable struct ShiftedNormTVp{
    R<:Real,
    T<:Real,
    V0<:AbstractVector{R},
    V1<:AbstractVector{R},
    V2<:AbstractVector{R},
} <: InexactShiftedProximableFunction
    h::NormTVp{R,T}
    xk::V0
    sj::V1
    sol::V2
    shifted_twice::Bool
    xsy::V2

    function ShiftedNormTVp(
        h::NormTVp{R,T},
        xk::AbstractVector{R},
        sj::AbstractVector{R},
        shifted_twice::Bool,
    ) where {R<:Real,T<:Real}
        sol = similar(xk)
        xsy = similar(xk)
        new{R,T,typeof(xk),typeof(sj),typeof(sol)}(h, xk, sj, sol, shifted_twice, xsy)
    end
end

"""
    shifted(h::NormTVp, xk::AbstractVector)

Creates a ShiftedNormTVp object with initial shift `xk`.
"""
shifted(h::NormTVp{R,T}, xk::AbstractVector{R}) where {R<:Real,T<:Real} =
    ShiftedNormTVp(h, xk, zero(xk), false)

"""
    shifted(ψ::ShiftedNormTVp, sj::AbstractVector)

Creates a ShiftedNormTVp object by adding a second shift `sj`.
"""
shifted(
    ψ::ShiftedNormTVp{R,T,V0,V1,V2},
    sj::AbstractVector{R},
) where {
    R<:Real,
    T<:Real,
    V0<:AbstractVector{R},
    V1<:AbstractVector{R},
    V2<:AbstractVector{R},
} = ShiftedNormTVp(ψ.h, ψ.xk, sj, true)

# Functions to get the name, expression, and parameters of the function
fun_name(ψ::ShiftedNormTVp) = "shifted TVp norm"
fun_expr(ψ::ShiftedNormTVp) = "t ↦ λ * TVp(xk + sj + t)"
fun_params(ψ::ShiftedNormTVp) = "xk = $(ψ.xk)\n" * " "^14 * "sj = $(ψ.sj)"

"""
    shift!(ψ::ShiftedNormTVp, shift::AbstractVector)

Updates the shift of a ShiftedNormTVp object.
"""
function shift!(ψ::ShiftedNormTVp, shift::AbstractVector{R}) where {R<:Real}
    if ψ.shifted_twice
        ψ.sj .= shift
    else
        ψ.xk .= shift
    end
    return ψ
end

# Allows ShiftedNormTVp objects to be called as functions
function (ψ::ShiftedNormTVp)(y::AbstractVector)
    @. ψ.xsy = ψ.xk + ψ.sj + y
    return ψ.h(ψ.xsy)
end

"""
    prox!(y, ψ::ShiftedNormTVp, q, σ; dualGap=1e-5)

Evaluates inexactly the proximity operator of a shifted TVp norm.
The duality gap at the solution is guaranteed to be less than `dualGap`.

Inputs:
    - `y`: Array in which to store the result.
    - `ψ`: ShiftedNormTVp object.
    - `q`: Vector to which the proximity operator is applied.
    - `ν`: Scaling factor.
    - `dualGap`: Desired quality of the solution in terms of duality gap (default `1e-5`).

Although `dualGap` can be specified, the TVp proximity operator uses a fixed objective gap of `1e-5` as defined in the C++ code. A warning will be emitted the first time this function is called.

"""
function prox!(y::AbstractArray, ψ::ShiftedNormTVp, q::AbstractArray, ν::Real; dualGap=1e-5)
    n = length(y)
    ws = ProxTV.newWorkspace(n)

    # Allocate info array (based on C++ code)
    info = zeros(Float64, 3)

    # Compute y_shifted = xk + sj + q
    y_shifted = ψ.xk .+ ψ.sj .+ q

    # Adjust lambda to account for σ (multiply λ by σ)
    lambda_scaled = ψ.h.λ * ν

    # Allocate the x vector to store the intermediate solution
    x = similar(y)

    # Call the TV function from ProxTV package
    ProxTV.TV(y_shifted, lambda_scaled, x, info, n, ψ.h.p, ws, objGap = dualGap)

    # Compute s = x - xk - sj
    s = x .- ψ.xk .- ψ.sj

    # Store the result in y
    y .= s

    return y
end



### general utility functions

"""
    shifted(h::Union{NormLp, NormTVp}, xk::AbstractVector)

Creates a shifted version of `h` depending on its type.
If `h` is of type `NormLp`, it returns a `ShiftedNormLp`.
If `h` is of type `NormTVp`, it returns a `ShiftedNormTVp`.
"""
function shifted(h::Union{NormLp, NormTVp}, xk::AbstractVector)
    if h isa NormLp
        return ShiftedNormLp(h, xk, zero(xk), false)
    elseif h isa NormTVp
        return ShiftedNormTVp(h, xk, zero(xk), false)
    else
        throw(ArgumentError("The function h must be either NormLp or NormTVp"))
    end
end

"""
    prox!(y, ψ::Union{InexactShiftedProximableFunction, ShiftedProximableFunction}, q, ν; dualGap=nothing)

Evaluates the proximity operator of a shifted regularizer, choosing between exact and inexact calculations based on the type of `ψ` and the presence of `dualGap`.

- If `ψ` is a `ShiftedProximableFunction` and `dualGap` is not provided, computes the **exact** proximity operator.
- If `ψ` is an `InexactShiftedProximableFunction` and `dualGap` is provided, computes the **inexact** proximity operator with a guaranteed duality gap below `dualGap`.

Inputs:
    - `y`: Array in which to store the result.
    - `ψ`: Either a `ShiftedProximableFunction` (for exact prox) or an `InexactShiftedProximableFunction` (for inexact prox).
    - `q`: Vector to which the proximity operator is applied.
    - `ν`: Scaling factor.
    - `dualGap`: Desired quality of the solution in terms of duality gap for inexact prox (default is `nothing`, indicating exact prox).

Outputs:
    - The solution is stored in the input vector `y`, which is also returned.

Errors:
    - Raises an error if `ψ` is of type `ShiftedProximableFunction` and `dualGap` is provided, or if `ψ` is of type `InexactShiftedProximableFunction` and `dualGap` is not provided.
"""
function prox!(y, ψ::Union{InexactShiftedProximableFunction, ShiftedProximableFunction}, q, ν; dualGap=nothing)
    if dualGap === nothing && ψ isa ShiftedProximableFunction
        # Call to exact prox!() if dualGap is not defined
        return prox!(y, ψ, q, ν)
    elseif dualGap !== nothing && ψ isa InexactShiftedProximableFunction
        # Call to inexact prox!() if dualGap is defined
        return prox!(y, ψ, q, ν; dualGap=dualGap)
    else
        error("Combination of ψ::$(typeof(ψ)) and dualGap::$(typeof(dualGap)) is not a valid call to prox!.
        Please provide dualGap::Real for InexactShiftedProximableFunction or omit it for ShiftedProximableFunction.")
    end
end


function check_condition_xi!(s, ψ::Union{InexactShiftedProximableFunction, ShiftedProximableFunction}, q, ν, κξ, ξ, mk, hk, k, dualGap)
    while dualGap > (1-κξ) / κξ * ξ
        # @info " -> iR2N: dualGap condition not satisfied, recomputing prox at iteration $k."
        dualGap = (1-κξ) / κξ * ξ
        prox!(s, ψ, q, ν; dualGap=dualGap)
        ξ = hk - mk(s) + max(1, abs(hk)) * 10 * eps()
    end
    return s, dualGap, ξ
end
