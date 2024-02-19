abstract type AbstractQDIM  <: ContinuousUnivariateDistribution end

"""
    QDIM{T<:Real} <: AbstractQDIM

A model object for the Quantum Prisoner's Dilemma Model. The QDIM has four basis states:
    
1. win first gamble, accept second gamble 
2. win first gamble, decline second gamble 
3. lose first gamble, accept second gamble 
4. lose first gamble, decline second gamble 

The bases are orthonormal and in standard form. The model generates predictions for three conditions:

1. Accept second gamble after winning first gamble 
2. Accept second gamble after losing first gamble
3. Plan to accept second gamble before observing outcome

# Fields 

- `α::T`: utility curvature parameter where α < 1 is risk averse and α > 1 is risk seeking
- `λ::T`: loss aversion parameter 
- `w₁:T`: decision weight for the first outcome
- `γ::T`: entanglement parameter for beliefs and actions 

# Constructors 

    QDIM(; α, λ, w₁, γ)

    QDIM(α, λ, w₁, γ)

# Example 

```julia
using QuantumDynamicInconsistencyModels
model = QDIM(; α = .9, λ = 2, w₁ = .5, γ = -1.74)
```

# References 

Busemeyer, J. R., Wang, Z., & Shiffrin, R. M. (2015). Bayesian model comparison favors quantum over standard decision theory account of dynamic inconsistency. Decision, 2(1), 1.
"""
struct QDIM{T<:Real} <: AbstractQDIM
    α::T
    λ::T
    w₁::T
    γ::T
end

QDIM(; α, λ, w₁, γ) = QDIM(α, λ, w₁, γ)

function QDIM(α, λ, w₁, γ) 
    return QDIM(promote(α, λ, w₁, γ)...)
end

Base.broadcastable(dist::AbstractQDIM) = Ref(dist)