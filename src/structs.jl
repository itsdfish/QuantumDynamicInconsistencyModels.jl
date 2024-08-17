abstract type AbstractQDIM <: ContinuousUnivariateDistribution end

"""
    QDIM{T<:Real} <: AbstractQDIM

A model object for the Quantum Prisoner's Dilemma Model. The QDIM has four basis states:
    
1. win first gamble, accept second gamble 
2. win first gamble, decline second gamble 
3. lose first gamble, accept second gamble 
4. lose first gamble, decline second gamble 

The bases are orthonormal and in standard form. The model returns the joint choice distribution for the planned and final decision of the 
second gamble conditioned on outcome of first gamble. 

1. probability of planning to accept second gamble and accepting second gamble
2. probability of planning to accept second gamble and rejecting second gamble
3. probability of planning to reject second gamble and accepting second gamble
4. probability of planning to reject second gamble and rejecting second gamble

# Fields 

- `α::T`: utility curvature parameter where α < 1 is risk averse and α > 1 is risk seeking
- `λ::T`: loss aversion parameter 
- `w₁:T`: decision weight for the first outcome
- `m::T`: the probability remembering and repeating the first response
- `γ::T`: entanglement parameter for beliefs and actions 

# Constructors 

    QDIM(; α, λ, w₁, m, γ)

    QDIM(α, λ, w₁, m, γ)

# Example 

```julia
using QuantumDynamicInconsistencyModels
model = QDIM(; α = .9, λ = 2, w₁ = .5, m, = .6, γ = -1.74)
```

# References 

Busemeyer, J. R., Wang, Z., & Shiffrin, R. M. (2015). Bayesian model comparison favors quantum over standard decision theory account of dynamic inconsistency. Decision, 2(1), 1.
"""
struct QDIM{T <: Real} <: AbstractQDIM
    α::T
    λ::T
    w₁::T
    m::T
    γ::T
end

QDIM(; α, λ, w₁, m, γ) = QDIM(α, λ, w₁, m, γ)

function QDIM(α, λ, w₁, m, γ)
    return QDIM(promote(α, λ, w₁, m, γ)...)
end

Base.broadcastable(dist::AbstractQDIM) = Ref(dist)
