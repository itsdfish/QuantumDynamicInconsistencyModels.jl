"""
    predict(model::AbstractQDIM; t = π / 2)

Returns predicted response probability for the following conditions:

1. Player 2 is told that player 1 defected
2. Player 2 is told that player 1 cooperated
3. Player 2 is not informed of player 1's action
    
# Arguments

- `model::AbstractQDIM`

# Keywords

- `t = π / 2`: time of decision

# Example 

```julia 
using QuantumDynamicInconsistencyModels 
model = QDIM(;μd=.51, γ=2.09)
predict(model)
```
"""
function predict(model::AbstractQDIM, outcomes1, outcomes2; t = π / 2)
    (;γ) = model
    # utility difference between accepting/rejecting second gamble given a (1) win,
    # or (2) loss in the first gamble 
    d_win,d_loss = get_utility_diffs(model, outcomes1, outcomes2)
    # rotates belief in favor of cooperation of defection 
    H1 = make_H1(d_win, d_loss)
    # hamiltonian matrix for reducing cognitive dissonance
    # aligns action with belief about opponent's action 
	H2 = make_H2(γ)
    # combine both hamiltonian matrices so that time evolution reflects their joint contribution
	H = H1 .+ H2
    # unitary transformation matrix
    U = exp(-im * t * H)

    # cognitive state after observing win in first gamble 
    ψw = [√(.5), √(.5), 0, 0]
    # cognitive state after observing loss in first gamble 
	ψl = [0, 0, √(.5), √(.5)]
    # cognitive state when outcome of first gamble is unknown
    ψ0 = fill(.5, 4)

    # projection matrix for accepting second gamble  
    M = Diagonal([1.0,0.0,1.0,0.0])

    # compute probability of accepting second gamble given a win in the first gamble 
	proj_w = M * U * ψw
	p_w = real(proj_w' * proj_w)

    # compute probability of accepting second gamble given a loss in the first gamble 
	proj_l = M * U * ψl
	p_l = real(proj_l' * proj_l)

    # compute probability of defecting given no knowledge of opponent's action
	proj = M * U * ψ0
	p = real(proj' * proj)
    return [p_w,p_l,p]
end

"""
    make_H1(μd, μc)

Creates a Hamiltonian matrix which rotates in favor of defecting or cooperating depending on 
μd and μd. 

# Arguments 

- `μd`: utility for defecting 
- `μc`: utility for cooperating
"""
function make_H1(d_win, d_loss)
    hw = tanh(.5 * d_win)
    hl = tanh(.5 * d_loss)

	return [hw / √(1 + hw^2) 1 / √(1 + hw^2) 0 0;
         1 / √(1 + hw^2) -hw / √(1 + hw^2) 0 0;
         0 0 hl / √(1 + hl^2) 1 / √(1 + hl^2);
         0 0 1 / √(1 + hl^2) -hl / √(1 + hl^2)]
end

"""
    make_H2(γ)

Creates a Hamiltonian matrix which represents cognitive dissonance or wishful thinking. The matrix can be decomposed
into two components. The components rotate beliefs about the other player to be more consistent with planned actions.  
For example, if the other player defected, the matrix will rotate actions towards defection. 

# Arguments 

- `γ`: entanglement parameter which aligns beliefs and actions
"""
function make_H2(γ)
	v = -γ / √(2)
	H = [v 0 v 0;
		 0 -v 0 v;
		 v 0 -v 0;
		 0 v 0 v]
	return H
end

function rand(model::AbstractQDIM, outcomes1, outcomes2; t = π / 2)
    return rand(model, outcomes1, outcomes2, 1; t)
end

"""
    rand(model::AbstractQDIM, n::Int; t = π / 2)

Generates simulated data for the following conditions:

1. Player 2 is told that player 1 defected
2. Player 2 is told that player 1 cooperated
3. Player 2 is not informed of player 1's action

# Arguments

- `model::AbstractQDIM`
- `n`: the number of trials per condition 

# Keywords

- `t = π / 2`: time of decision

# Example 

```julia 
using QuantumDynamicInconsistencyModels 
model = QDIM(;μd=.51, γ=2.09)
data = rand(model, 100)
```
"""
function rand(model::AbstractQDIM, outcomes1::Vector{<:Number}, outcomes2::Vector{<:Number}, n::Int; t = π / 2)
    Θ = predict(model, outcomes1, outcomes2; t)
    return @. rand(Binomial(n, Θ))
end

function rand(model::AbstractQDIM, outcomes1, outcomes2, n::Int; t = π / 2)
    return map(x -> rand(model, x..., n; t), zip(outcomes1, outcomes2))
end

function rand(model::AbstractQDIM, outcomes1, outcomes2, n; t = π / 2)
    return map(x -> rand(model, x...; t), zip(outcomes1, outcomes2, n))
end

"""
    pdf(model::AbstractQDIM, n::Int, n_d::Vector{Int}; t = π / 2)

Returns the joint probability density given data for the following conditions:

1. Player 2 is told that player 1 defected
2. Player 2 is told that player 1 cooperated
3. Player 2 is not informed of player 1's action
    

# Arguments

- `model::AbstractQDIM`
- `n`: the number of trials per condition 
- `n_d`: the number of defections in each condition 

# Keywords

- `t = π / 2`: time of decision
"""
function pdf(model::AbstractQDIM, n::Int, n_d::Vector{Int}; t = π / 2)
    Θ = predict(model; t)
    return prod(@. pdf(Binomial(n, Θ), n_d)) 
end

"""
    logpdf(model::AbstractQDIM, n::Int, n_d::Vector{Int}; t = π / 2)

Returns the joint log density given data for the following conditions:

1. Player 2 is told that player 1 defected
2. Player 2 is told that player 1 cooperated
3. Player 2 is not informed of player 1's action

# Arguments

- `model::AbstractQDIM`: a subtype of `AbstractQDIM`
- `n`: the number of trials per condition 
- `n_d`: the number of defections in each condition 

# Keywords

- `t = π / 2`: time of decision

# Example 

```julia 
using QuantumDynamicInconsistencyModels 
model = QDIM(;μd=.51, γ=2.09)
n_trials = 100
data = rand(model, n_trials)
logpdf(model, n_trials, data)
```
"""
function logpdf(
        model::AbstractQDIM, 
        outcomes1::Vector{<:Number}, 
        outcomes2::Vector{<:Number},
        data::Vector{<:Number}, 
        n::Int; 
        t = π / 2
    )
    Θ = predict(model, outcomes1, outcomes2; t)
    return sum(@. logpdf(Binomial(n, Θ), data))
end

function logpdf(model::AbstractQDIM, outcomes1, outcomes2, data, n::Int; t = π / 2)
    return mapreduce(x -> logpdf(model, x..., n; t), +, zip(outcomes1, outcomes2, data))
end

function logpdf(model::AbstractQDIM, outcomes1, outcomes2, data, n; t = π / 2)
    return mapreduce(x -> logpdf(model, x...; t), +, zip(outcomes1, outcomes2, data, n))
end

loglikelihood(d::AbstractQDIM, data::Tuple) = logpdf(d, data...)

logpdf(model::AbstractQDIM, x::Tuple) = logpdf(model, x...)

function get_expected_utility(model::AbstractQDIM, vals::Vector{<:Number})
    (;λ,α,w_win) = model
    utils = get_utility.(vals, α, λ)
    w = [w_win,1-w_win]
    return utils' * w 
end

function get_utility(v::Number, α, λ)
    return  v < 0 ? -λ * abs(v)^α : v^α
end

"""
    get_utility_diffs(model::AbstractQDIM, outcomes1, outcomes2)

Computes the utility differences for chosing accepting gamble given a win and given a loss
during the previous stage. 

# Arguments

- `model::AbstractQDIM`: a subtype of `AbstractQDIM`
- `outcomes1`: the win loss outcomes for stage 1 
- `outcomes2`: the win loss outcomes for stage 2

# Returns 

- `util_diffs`: utility diff between taking second gamble and rejecting second gamble. The first entanglement
is conditioned on winning in the first stage and the second element is conditioned on losing in the first stage.
"""
function get_utility_diffs(model::AbstractQDIM, outcomes1, outcomes2)
    (;α,λ) = model
    vals = map(x -> outcomes2 .+ x,  outcomes1)
    # utility of gamble 1 outcomes 
    u1 = get_utility.(outcomes1, α, λ)
    # gamble 2 expected utility 
    u2 = [get_expected_utility(model, vals[i]) for i ∈ 1:length(vals)]
    return u2 - u1
end