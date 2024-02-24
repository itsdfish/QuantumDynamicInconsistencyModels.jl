"""
    predict(model::AbstractQDIM, outcomes1, outcomes2; t = π / 2)

Returns predicted response probabilities for the following conditions 

1. Accept second gamble after winning first gamble 
2. Accept second gamble after losing first gamble
3. Plan to accept second gamble before observing outcome

# Arguments

- `model::AbstractQDIM`: a subtype of `AbstractQDIM``
- `outcomes1::Vector{<:Number}`: outcomes for the first gamble
- `outcomes2::Vector{<:Number}`: outcomes for the second gamble 

# Keywords

- `t = π / 2`: time of decision

# Example 

```julia 
using QuantumDynamicInconsistencyModels
model = QDIM(; α = .9, λ = 2, w₁ = .5, γ = -1.74)
outcomes1 = [2,-1]
outcomes2 = [2,-1]
preds = predict(model, outcomes1, outcomes2)
```
"""
function predict(model::AbstractQDIM, outcomes1, outcomes2, win_first; t = π / 2)
    (;γ, m) = model
    p_plan, p_final = win_first ? predict_win(model, outcomes1, outcomes2; t) : predict_loss(model, outcomes1, outcomes2; t)
    return predict_joint_probs(model, p_plan, p_final)
end

"""
    make_H1(d_win, d_loss)  

Creates a Hamiltonian matrix which rotates in favor of accepting second gamble based on outcome of 
first gamble. 

# Arguments 

- `d_win`: utility difference given a win in the first gamble 
- `d_loss`: utility difference given a loss in the first gamble 
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
    rand(
        model::AbstractQDIM, 
        outcomes1::Vector{<:Number}, 
        outcomes2::Vector{<:Number}, 
        n::Int; 
        t = π / 2
    )

Generates simulated data for the following conditions:

1. Accept second gamble after winning first gamble 
2. Accept second gamble after losing first gamble
3. Plan to accept second gamble before observing outcome

# Arguments

- `model::AbstractQDIM`: a subtype of `AbstractQDIM``
- `outcomes1::Vector{<:Number}`: outcomes for the first gamble
- `outcomes2::Vector{<:Number}`: outcomes for the second gamble 
- `n`: the number of trials per condition 

# Keywords

- `t = π / 2`: time of decision

# Example 

```julia 
using QuantumDynamicInconsistencyModels
model = QDIM(; α = .9, λ = 1, w₁ = .5, γ = -1.74)
outcomes1 = [2,-1]
outcomes2 = [2,-1]
n_trials = 10
data = rand(model, outcomes1, outcomes2, n_trials)
```
"""
function rand(
        model::AbstractQDIM, 
        outcomes1::Vector{<:Number}, 
        outcomes2::Vector{<:Number}, 
        n::Int; 
        t = π / 2
    )
    Θ = predict(model, outcomes1, outcomes2; t)
    return @. rand(Binomial(n, Θ))
end

function rand(model::AbstractQDIM, outcomes1, outcomes2, n::Int; t = π / 2)
    return map(x -> rand(model, x..., n; t), zip(outcomes1, outcomes2))
end

"""
    rand(model::AbstractQDIM, outcomes1, outcomes2, n; t = π / 2)

Generates simulated data for the following conditions:

1. Accept second gamble after winning first gamble 
2. Accept second gamble after losing first gamble
3. Plan to accept second gamble before observing outcome

# Arguments

- `model::AbstractQDIM`: a subtype of `AbstractQDIM``
- `outcomes1`: a vector of vectors where each subvector contains outcomes for the first gamble of a given trial
- `outcomes2`: a vector of vectors where each subvector contains outcomes for the second gamble of a given trial
- `n`: a vector containing the number of trials per condition per gamble

# Keywords

- `t = π / 2`: time of decision

# Example 

```julia 
using QuantumDynamicInconsistencyModels
model = QDIM(; α = .9, λ = 1, w₁ = .5, γ = -1.74)
outcomes1 = [[2,-1],[3,-2]]
outcomes2 = [[2,-1],[3,-2]]
n_trials = [10,20]
data = rand(model, outcomes1, outcomes2, n_trials)
```
"""
function rand(model::AbstractQDIM, outcomes1, outcomes2, n; t = π / 2)
    return map(x -> rand(model, x...; t), zip(outcomes1, outcomes2, n))
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

"""
    rand(model::AbstractQDIM, outcomes1, outcomes2, n; t = π / 2)

Generates simulated data for the following conditions:

1. Accept second gamble after winning first gamble 
2. Accept second gamble after losing first gamble
3. Plan to accept second gamble before observing outcome

# Arguments

- `model::AbstractQDIM`: a subtype of `AbstractQDIM``
- `outcomes1`: a vector of vectors where each subvector contains outcomes for the first gamble of a given trial
- `outcomes2`: a vector of vectors where each subvector contains outcomes for the second gamble of a given trial
- `n::Int`: the number of trials per condition per gamble

# Keywords

- `t = π / 2`: time of decision

# Example 

```julia 
using QuantumDynamicInconsistencyModels
model = QDIM(; α = .9, λ = 1, w₁ = .5, γ = -1.74)
outcomes1 = [[2,-1],[3,-2]]
outcomes2 = [[2,-1],[3,-2]]
n_trials = 10
data = rand(model, outcomes1, outcomes2, n_trials)
```
"""
function logpdf(model::AbstractQDIM, outcomes1, outcomes2, data, n::Int; t = π / 2)
    return mapreduce(x -> logpdf(model, x..., n; t), +, zip(outcomes1, outcomes2, data))
end

function logpdf(model::AbstractQDIM, outcomes1, outcomes2, data, n; t = π / 2)
    return mapreduce(x -> logpdf(model, x...; t), +, zip(outcomes1, outcomes2, data, n))
end

loglikelihood(d::AbstractQDIM, data::Tuple) = logpdf(d, data...)

logpdf(model::AbstractQDIM, x::Tuple) = logpdf(model, x...)

function get_expected_utility(model::AbstractQDIM, vals::Vector{<:Number})
    (;λ,α,w₁) = model
    utils = get_utility.(vals, α, λ)
    w = [w₁,1-w₁]
    return utils' * w 
end

get_utility(v, α, λ) = v < 0 ? -λ * abs(v)^α : v^α

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

function predict_win(model::AbstractQDIM, outcomes1, outcomes2; t = π / 2)
    (;γ) = model
    # utility difference between accepting/rejecting second gamble given a (1) win,
    # or (2) loss in the first gamble 
    d_win,d_loss = get_utility_diffs(model, outcomes1, outcomes2)
    # rotates belief in favor of accepting gamble 
    H1 = make_H1(d_win, d_loss)
    # hamiltonian matrix for reducing cognitive dissonance
    # aligns action with belief about winning gamble
	H2 = make_H2(γ)
    # combine both hamiltonian matrices so that time evolution reflects their joint contribution
	H = H1 .+ H2
    # unitary transformation matrix
    U = exp(-im * t * H)

    # cognitive state after observing win in first gamble 
    ψw = [√(.5), √(.5), 0, 0]
    # cognitive state when outcome of first gamble is unknown
    ψ0 = fill(.5, 4)

    # basic vectors 
    # 1. win first gamble, accept second gamble 
    # 2. win first gamble, decline second gamble 
    # 3. lose first gamble, accept second gamble 
    # 4. lose first gamble, decline second gamble 

    # projection matrix for accepting second gamble  
    Ma = Diagonal([1.0,0.0,1.0,0.0])
    # projection matrix for winning first gamble and accepting second gamble  
    Maw = Diagonal([1.0,0.0,0.0,0.0])
    # projection matrix for winning first gamble 
    Mw = Diagonal([1.0,1.0,0.0,0.0])

    # compute probability of planning to accept second gamble given a win in the first gamble 
	proj_w = Ma * U * ψw
	p_w = real(proj_w' * proj_w)

    # compute probability of planning to accept second gamble and winning first gamble
	proj = Maw * U * ψ0
	p_p_aw = real(proj' * proj)

    # compute probability of winning first gamble when making a plan 
    proj = Mw * U * ψ0
	p_p_w = real(proj' * proj)

    # proj = Mw * U * ψ0
    # ψpw = proj ./ norm(proj)
    # proj = Maw * ψpw
	# p_p_w1 = real(proj' * proj)
    # println(p_p_w1)

    return [p_p_aw / p_p_w, p_w]
end


function predict_loss(model::AbstractQDIM, outcomes1, outcomes2; t = π / 2)
    (;γ, m) = model
    # utility difference between accepting/rejecting second gamble given a (1) win,
    # or (2) loss in the first gamble 
    d_win,d_loss = get_utility_diffs(model, outcomes1, outcomes2)
    # rotates belief in favor of accepting gamble 
    H1 = make_H1(d_win, d_loss)
    # hamiltonian matrix for reducing cognitive dissonance
    # aligns action with belief about winning gamble
	H2 = make_H2(γ)
    # combine both hamiltonian matrices so that time evolution reflects their joint contribution
	H = H1 .+ H2
    # unitary transformation matrix
    U = exp(-im * t * H)

    # cognitive state after observing loss in first gamble 
	ψl = [0, 0, √(.5), √(.5)]
    # cognitive state when outcome of first gamble is unknown
    ψ0 = fill(.5, 4)

    # basic vectors 
    # 1. win first gamble, accept second gamble 
    # 2. win first gamble, decline second gamble 
    # 3. lose first gamble, accept second gamble 
    # 4. lose first gamble, decline second gamble 

    # projection matrix for accepting second gamble  
    Ma = Diagonal([1.0,0.0,1.0,0.0])
    # projection matrix for losing first gamble  
    Ml = Diagonal([0.0,0.0,1.0,1.0])
    # projection matrix for losing first gamble and accepting second gamble  
    Mal = Diagonal([0.0,0.0,1.0,0.0])

    # compute probability of planning to accept second gamble given a loss in the first gamble 
	proj_l = Ma * U * ψl
	p_l = real(proj_l' * proj_l)

    # compute probability of planning to accept second gamble and losing first gamble
    proj = Mal * U * ψ0
	p_p_al = real(proj' * proj)

    # compute probability of losing first gamble when making a plan 
    proj = Ml * U * ψ0
	p_p_l = real(proj' * proj)

    return [p_p_al / p_p_l, p_l]
end

function predict_joint_probs(model::AbstractQDIM, p_plan, p_final)
    (;m) = model
    # probability of accepting first gamble and accepting second gamble
    p_aa = p_plan * (m + (1 - m) * p_final) 
    # probability of accepting first gamble and rejecting second gamble
    p_ar = p_plan * (1 - m) * (1 - p_final)
    # probability of accepting first gamble and rejecting second gamble
    p_ra = (1 - p_plan) * (1 - m) * p_final  
    # probability of rejecting first gamble and rejecting second gamble
    p_rr = (1 - p_plan) * (m  +  (1 - m) * (1 - p_final)) 
    return [p_aa,p_ar,p_ra,p_rr]
end

"""
    predict_sure_thing(model::AbstractQDIM, outcomes1, outcomes2; t = π / 2)

Returns predicted response probabilities for the following conditions 

1. Accept second gamble after winning first gamble 
2. Accept second gamble after losing first gamble
3. Plan to accept second gamble before observing outcome

# Arguments

- `model::AbstractQDIM`: a subtype of `AbstractQDIM``
- `outcomes1::Vector{<:Number}`: outcomes for the first gamble
- `outcomes2::Vector{<:Number}`: outcomes for the second gamble 

# Keywords

- `t = π / 2`: time of decision

# Example 

```julia 
using QuantumDynamicInconsistencyModels
using QuantumDynamicInconsistencyModels: predict_sure_thing
model = QDIM(; α = .9, λ = 2, w₁ = .5, γ = -1.74)
outcomes1 = [2,-1]
outcomes2 = [2,-1]
preds = predict_sure_thing(model, outcomes1, outcomes2)
```
"""
function predict_sure_thing(model::AbstractQDIM, outcomes1, outcomes2; t = π / 2)
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