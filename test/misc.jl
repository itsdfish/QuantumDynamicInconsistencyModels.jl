
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
