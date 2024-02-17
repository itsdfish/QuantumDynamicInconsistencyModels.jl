abstract type AbstractQDIM  <: ContinuousUnivariateDistribution end

"""

    QDIM{T<:Real} <: AbstractQDIM

A model object for the Quantum Prisoner's Dilemma Model. The QDIM has four basis states:
    
1. opponent defects and you defect 
2. opponent defects and you cooperate 
3. opponent cooperates and you defect 
4. opponent cooperates and you cooperate

The bases are orthonormal and in standard form. The model assumes three conditions:

1. Player 2 is told that player 1 defected
2. Player 2 is told that player 1 cooperated
3. Player 2 is not informed of the action of player 1


Model inputs and outputs are assumed to be in the order above. 

# Fields 

- `μd`: utility for defecting 
- `μc`: utility for cooperating 
- `γ`: entanglement parameter for beliefs and actions 

# Example 

```julia
using QuantumDynamicInconsistencyModels
model = QDIM(;μd=.51, γ=2.09)
```

# References 

Pothos, E. M., & Busemeyer, J. R. (2009). A quantum probability explanation for violations of ‘rational’decision theory. Proceedings of the Royal Society B: Biological Sciences, 276(1665), 2171-2178.
"""
struct QDIM{T<:Real} <: AbstractQDIM
    α::T
    λ::T
    w_win::T
    γ::T
end

QDIM(; α, λ, w_win, γ) = QDIM(α, λ, w_win, γ)

function QDIM(α, λ, w_win, γ) 
    return QDIM(promote(α, λ, w_win, γ)...)
end

Base.broadcastable(dist::AbstractQDIM) = Ref(dist)