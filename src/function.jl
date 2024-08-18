"""
    predict(
        model::AbstractQDIM,
        outcomes1::Vector{<:Real},
        outcomes2::Vector{<:Real},
        won_first::Bool;
        t = π / 2
    )

Returns the joint choice distribution for the planned and final decision of the 
second gamble conditioned on outcome of first gamble. 

# Arguments

- `model::AbstractQDIM`: a subtype of `AbstractQDIM``
- `outcomes1::Vector{<:Real}`: outcomes for the first gamble
- `outcomes2::Vector{<:Real}`: outcomes for the second gamble
- `won_first`: set true if first gamble was won, in which case evaluation of final decision
    is conditioned on winning first gamble 

# Keywords

- `t = π / 2`: time of decision

# Returns 

Returns a vector respresenting the joint distribution over planned and final choices, where elements correspond to 

1. probability of planning to accept second gamble and accepting second gamble
2. probability of planning to accept second gamble and rejecting second gamble
3. probability of planning to reject second gamble and accepting second gamble
4. probability of planning to reject second gamble and rejecting second gamble

# Example 

```julia 
using QuantumDynamicInconsistencyModels
model = QDIM(; α = .9, λ = 2, w₁ = .5, γ = -1.74, m = .3)
outcomes1 = [2,-1]
outcomes2 = [2,-1]
won_first = true
preds = predict(model, outcomes1, outcomes2, won_first)
```
"""
function predict(
    model::AbstractQDIM,
    outcomes1::Vector{<:Real},
    outcomes2::Vector{<:Real},
    won_first::Bool;
    t = π / 2
)
    (; γ, m) = model
    p_plan, p_final =
        won_first ? predict_given_win(model, outcomes1, outcomes2; t) :
        predict_given_loss(model, outcomes1, outcomes2; t)
    return predict_joint_probs(model, p_plan, p_final)
end

"""
    predict_given_win(
        model::AbstractQDIM,
        outcomes1::Vector{<:Real},
        outcomes2::Vector{<:Real};
        t = π / 2
    )

Returns the probability of planning to accept the second gamble (not conditioned on an anticipated outcome) and
the probability of accepting the second gamble in the final decision conditioned on winning the first gamble.

# Arguments

- `model::AbstractQDIM`: a subtype of `AbstractQDIM``
- `outcomes1::Vector{<:Real}`: outcomes for the first gamble
- `outcomes2::Vector{<:Real}`: outcomes for the second gamble

# Keywords

- `t = π / 2`: time of decision
"""
function predict_given_win(
    model::AbstractQDIM,
    outcomes1::Vector{<:Real},
    outcomes2::Vector{<:Real};
    t = π / 2
)
    (; γ) = model
    # utility difference between accepting/rejecting second gamble given a (1) win,
    # or (2) loss in the first gamble 
    d_win, d_loss = get_utility_diffs(model, outcomes1, outcomes2)
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
    ψw = [√(0.5), √(0.5), 0, 0]
    # cognitive state when outcome of first gamble is unknown
    ψ0 = fill(0.5, 4)

    # basis vectors 
    # 1. win first gamble, accept second gamble 
    # 2. win first gamble, decline second gamble 
    # 3. lose first gamble, accept second gamble 
    # 4. lose first gamble, decline second gamble 

    # projection matrix for accepting second gamble  
    Pa = Diagonal([1.0, 0.0, 1.0, 0.0])

    # compute probability of planning to accept second gamble
    proj = Pa * U * ψ0
    p_p_a = real(proj' * proj)

    # compute probability of accepting second gamble given an experienced win
    proj = Pa * U * ψw
    p_f_a = real(proj' * proj)

    return [p_p_a, p_f_a]
end

"""
    predict_given_loss(
        model::AbstractQDIM,
        outcomes1::Vector{<:Real},
        outcomes2::Vector{<:Real};
        t = π / 2
    )

Returns the probability of planning to accept the second gamble (not conditioned on an anticipated outcome) and
    the probability of accepting the second gamble in the final decision conditioned on lossing the first gamble.

# Arguments

- `model::AbstractQDIM`: a subtype of `AbstractQDIM``
- `outcomes1::Vector{<:Real}`: outcomes for the first gamble
- `outcomes2::Vector{<:Real}`: outcomes for the second gamble

# Keywords

- `t = π / 2`: time of decision
"""
function predict_given_loss(
    model::AbstractQDIM,
    outcomes1::Vector{<:Real},
    outcomes2::Vector{<:Real};
    t = π / 2
)
    (; γ) = model
    # utility difference between accepting/rejecting second gamble given a (1) win,
    # or (2) loss in the first gamble 
    d_win, d_loss = get_utility_diffs(model, outcomes1, outcomes2)
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
    ψl = [0, 0, √(0.5), √(0.5)]
    # cognitive state when outcome of first gamble is unknown
    ψ0 = fill(0.5, 4)

    # basis vectors 
    # 1. win first gamble, accept second gamble 
    # 2. win first gamble, decline second gamble 
    # 3. lose first gamble, accept second gamble 
    # 4. lose first gamble, decline second gamble 

    # projection matrix for accepting second gamble  
    Pa = Diagonal([1.0, 0.0, 1.0, 0.0])

    # compute probability of planning to accept second gamble
    proj = Pa * U * ψ0
    p_p_a = real(proj' * proj)

    # compute probability of accepting second gamble given an experienced loss
    proj = Pa * U * ψl
    p_f_a = real(proj' * proj)

    return [p_p_a, p_f_a]
end

""" 
    predict_joint_probs(
        model::AbstractQDIM,
        p_plan::Real,
        p_final::Real
    )

Returns the joint choice distribution for the planned and final decision of the 
second gamble conditioned on outcome of first gamble. The joint probability distribution includes 
the probability of repeating of remembering and repeating the choice, denoted  by parameter `m`, which 
allows dependencies in the joint probability distribution 

# Arguments

- `model::AbstractQDIM`: a subtype of `AbstractQDIM``
- `p_plan::Real`: probability of planning to accept gamble 
- `p_final::Real`: probability of accepting gamble in final decision 

# Returns 

Returns a vector respresenting the joint distribution over planned and final choices, where elements correspond to 

1. probability of planning to accept second gamble and accepting second gamble
2. probability of planning to accept second gamble and rejecting second gamble
3. probability of planning to reject second gamble and accepting second gamble
4. probability of planning to reject second gamble and rejecting second gamble
"""
function predict_joint_probs(
    model::AbstractQDIM,
    p_plan::Real,
    p_final::Real
)
    (; m) = model
    # probability of planning to accept second gamble and accepting second gamble
    p_aa = p_plan * (m + (1 - m) * p_final)
    # probability of planning to accept second gamble and rejecting second gamble
    p_ar = p_plan * (1 - m) * (1 - p_final)
    # probability of planning to reject second gamble and accepting second gamble
    p_ra = (1 - p_plan) * (1 - m) * p_final
    # probability of planning to reject second gamble and rejecting second gamble
    p_rr = (1 - p_plan) * (m + (1 - m) * (1 - p_final))
    return [p_aa, p_ar, p_ra, p_rr]
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
    hw = tanh(0.5 * d_win)
    hl = tanh(0.5 * d_loss)

    return [
        hw/√(1 + hw^2) 1/√(1 + hw^2) 0 0
        1/√(1 + hw^2) -hw/√(1 + hw^2) 0 0
        0 0 hl/√(1 + hl^2) 1/√(1 + hl^2)
        0 0 1/√(1 + hl^2) -hl/√(1 + hl^2)
    ]
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
    H = [
        v 0 v 0
        0 -v 0 v
        v 0 -v 0
        0 v 0 v
    ]
    return H
end

"""
    rand(
        model::AbstractQDIM, 
        outcomes1::Vector{<:Real}, 
        outcomes2::Vector{<:Real}, 
        n::Int; 
        t = π / 2
    )

Returns samples from the joint choice distribution for the planned and final decision of the 
second gamble conditioned on outcome of first gamble. 


# Arguments

- `model::AbstractQDIM`: a subtype of `AbstractQDIM``
- `outcomes1::Vector{<:Real}`: outcomes for the first gamble
- `outcomes2::Vector{<:Real}`: outcomes for the second gamble 
- `won_first::Bool`: true if first gamble won
- `n`: the number of trials per condition 

# Keywords

- `t = π / 2`: time of decision

# Returns 

Returns a vector respresenting the samples from the joint distribution over planned and final choices, where elements correspond to 

1. frequency of planning to accept second gamble and accepting second gamble
2. frequency of planning to accept second gamble and rejecting second gamble
3. frequency of planning to reject second gamble and accepting second gamble
4. frequency of planning to reject second gamble and rejecting second gamble

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
    outcomes1::Vector{<:Real},
    outcomes2::Vector{<:Real},
    won_first::Bool,
    n::Int;
    t = π / 2
)
    Θ = predict(model, outcomes1, outcomes2, won_first; t)
    return rand(Multinomial(n, Θ))
end

function rand(
    model::AbstractQDIM,
    outcomes1::Vector{<:Real},
    outcomes2::Vector{<:Real},
    won_first::Bool;
    t = π / 2
)
    return rand(model, outcomes1, outcomes2, won_first, 1; t)
end

"""
    logpdf(
        model::AbstractQDIM, 
        outcomes1::Vector{<:Real}, 
        outcomes2::Vector{<:Real},
        data::Vector{<:Real}, 
        n::Int; 
        t = π / 2
    )

Returns the multinomial log loglikelihood for the joint planned decision and final decision of the 
second gamble conditioned on outcome of first gamble. The joint distribution is as follows:

1. probability of planning to accept second gamble and accepting second gamble
2. probability of planning to accept second gamble and rejecting second gamble
3. probability of planning to reject second gamble and accepting second gamble
4. probability of planning to reject second gamble and rejecting second gamble

# Arguments

- `model::AbstractQDIM`: a subtype of `AbstractQDIM`
- `outcomes1::Vector{<:Real}`: the win and loss outcomes for stage 1 
- `outcomes2::Vector{<:Real}`: the win and loss outcomes for stage 2
- `won_first::Bool`: indicates true if the larger of the two outcomes is won
- `n::Int`: the number of repetitions of the trial 
 - `data::Vector{<:Real}`: a vector of response frequencies joint planned decision and final decision of the 
    second gamble conditioned on outcome of first gamble. The frequencies correspond to the joint distribution above.

# Keywords

- `t = π / 2`: time of decision

# Example 

```julia 
model = QDIM(; α = .9, λ = 1, w₁ = .5, γ = -1.74)
outcomes1 = [2,-1]
outcomes2 = [2,-1]
n_trials = [10,20]
data = rand(model, outcomes1, outcomes2, n_trials)
```
"""
function logpdf(
    model::AbstractQDIM,
    outcomes1::Vector{<:Real},
    outcomes2::Vector{<:Real},
    won_first::Bool,
    n::Int,
    data::Vector{<:Real};
    t = π / 2
)
    Θ = predict(model, outcomes1, outcomes2, won_first; t)
    return logpdf(Multinomial(n, Θ), data)
end

loglikelihood(d::AbstractQDIM, data::Tuple) = sum(logpdf.(d, data...))

logpdf(model::AbstractQDIM, x::Tuple) = logpdf(model, x...)

function get_expected_utility(model::AbstractQDIM, vals::Vector{<:Real})
    (; λ, α, w₁) = model
    utils = get_utility.(vals, α, λ)
    w = [w₁, 1 - w₁]
    return utils' * w
end

get_utility(v, α, λ) = v < 0 ? -λ * abs(v)^α : v^α

"""
    get_utility_diffs(model::AbstractQDIM, outcomes1::Vector{<:Real}, outcomes2::Vector{<:Real})

Computes the utility differences for chosing accepting gamble given a win and given a loss
during the previous stage. 

# Arguments

- `model::AbstractQDIM`: a subtype of `AbstractQDIM`
- `outcomes1::Vector{<:Real}`: the win and loss outcomes for stage 1 
- `outcomes2::Vector{<:Real}`: the win and loss outcomes for stage 2

# Returns 

- `util_diffs`: utility diff between taking second gamble and rejecting second gamble. The first element
is conditioned on winning in the first stage and the second element is conditioned on losing in the first stage.
"""
function get_utility_diffs(
    model::AbstractQDIM,
    outcomes1::Vector{<:Real},
    outcomes2::Vector{<:Real}
)
    (; α, λ) = model
    vals = map(x -> outcomes2 .+ x, outcomes1)
    # utility of gamble 1 outcomes 
    u1 = get_utility.(outcomes1, α, λ)
    # gamble 2 expected utility 
    u2 = [get_expected_utility(model, vals[i]) for i ∈ 1:length(vals)]
    return u2 - u1
end

function predict_temp(model::AbstractQDIM, outcomes1, outcomes2; t = π / 2)
    (; γ) = model
    # utility difference between accepting/rejecting second gamble given a (1) win,
    # or (2) loss in the first gamble 
    d_win, d_loss = get_utility_diffs(model, outcomes1, outcomes2)
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
    ψw = [√(0.5), √(0.5), 0, 0]
    # cognitive state when outcome of first gamble is unknown
    ψ0 = fill(0.5, 4)

    # basis vectors 
    # 1. win first gamble, accept second gamble 
    # 2. win first gamble, decline second gamble 
    # 3. lose first gamble, accept second gamble 
    # 4. lose first gamble, decline second gamble 

    # projection matrix for accepting second gamble  
    Pa = Diagonal([1.0, 0.0, 1.0, 0.0])
    Paw = Diagonal([1.0, 0.0, 0.0, 0.0])
    Pw = Diagonal([1.0, 1.0, 0.0, 0.0])
    Pal = Diagonal([0.0, 0.0, 1.0, 0.0])
    Pl = Diagonal([0.0, 0.0, 1.0, 1.0])

    # compute probability of planning to accept second gamble
    proj = Paw * U * ψ0
    p_p_aw = real(proj' * proj)

    proj = Pw * U * ψ0
    p_p_w = real(proj' * proj)

    # compute probability of planning to accept second gamble
    proj = Pal * U * ψ0
    p_p_al = real(proj' * proj)

    proj = Pl * U * ψ0
    p_p_l = real(proj' * proj)

    # compute probability of accepting second gamble given an experienced win
    proj = Pa * U * ψw
    p_f_a = real(proj' * proj)

    return [p_p_aw / p_p_w, p_p_al / p_p_l, p_f_a]
end
