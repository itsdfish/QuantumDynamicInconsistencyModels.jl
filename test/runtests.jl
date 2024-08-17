using SafeTestsets

@safetestset "rand" begin
    @safetestset "rand 1" begin
        using QuantumDynamicInconsistencyModels
        using Test
        using Random

        Random.seed!(7878)

        n = 100_000

        parms = (α = 0.9, λ = 1, m = 0.30, w₁ = 0.5, γ = -1.74)

        model = QDIM(; parms...)
        outcomes1 = [2, -1]
        outcomes2 = [2, -1]
        won = true

        data = rand(model, outcomes1, outcomes2, won, n)
        preds = predict(model, outcomes1, outcomes2, won)
        probs = data ./ n
        @test probs ≈ preds atol = 1e-2
    end

    @safetestset "rand 2" begin
        using QuantumDynamicInconsistencyModels
        using Test
        using Random

        Random.seed!(665)

        n = 100_000

        parms = (α = 0.9, λ = 2, w₁ = 0.5, m = 0.30, γ = 2.5)

        model = QDIM(; parms...)
        outcomes1 = [2, -1]
        outcomes2 = [2, -1]
        won = false

        data = rand(model, outcomes1, outcomes2, won, n)
        preds = predict(model, outcomes1, outcomes2, won)
        probs = data ./ n
        @test probs ≈ preds atol = 1e-2
    end
end

@safetestset "H1" begin
    using QuantumDynamicInconsistencyModels
    using QuantumDynamicInconsistencyModels: make_H1
    using Test

    H1 = make_H1(1.5, 2)
    @test H1[1, 1] == -H1[2, 2]
    @test H1[3, 3] == -H1[4, 4]
    @test H1[1, 2] == H1[2, 1]
    @test H1[4, 3] == H1[3, 4]
    @test H1 == H1'
end

@safetestset "H2" begin
    using QuantumDynamicInconsistencyModels
    using QuantumDynamicInconsistencyModels: make_H2
    using Test

    H2 = make_H2(√(2) * 2)
    @test H2[1, 1] == -2
    @test H2[2, 2] == 2
    @test H2[2, 4] == -2
    @test H2[3, 1] == -2
    @test H2[3, 3] == 2
    @test H2[4, 2] == -2
    @test H2[4, 4] == -2
    @test H2 == H2'
end

@safetestset "logpdf" begin
    using QuantumDynamicInconsistencyModels
    using Test
    using Random

    Random.seed!(410)

    parms = (α = 0.9, λ = 2, m = 0.3, w₁ = 0.5, γ = 2.5)

    outcomes1 = [[2, -1], [5, -3], [0.5, -0.25], [2, -2], [5, -5], [0.5, -0.50]]
    outcomes2 = [[2, -1], [5, -3], [0.5, -0.25], [2, -2], [5, -5], [0.5, -0.50]]
    won = [true, false, true, true, false, false]

    ns = fill(20_000, 6)
    model = QDIM(; parms...)
    data = rand.(model, outcomes1, outcomes2, won, ns)

    γs = range(0.8 * parms.γ, 1.2 * parms.γ, length = 100)
    LLs = map(γ -> sum(logpdf.(QDIM(; parms..., γ), outcomes1, outcomes2, won, ns, data)), γs)
    _, mxi = findmax(LLs)
    @test γs[mxi] ≈ parms.γ rtol = 1e-2

    αs = range(0.8 * parms.α, 1.2 * parms.α, length = 100)
    LLs = map(α -> sum(logpdf.(QDIM(; parms..., α), outcomes1, outcomes2, won, ns, data)), αs)
    _, mxi = findmax(LLs)
    @test αs[mxi] ≈ parms.α rtol = 1e-2

    λs = range(0.8 * parms.λ, 1.2 * parms.λ, length = 100)
    LLs = map(λ -> sum(logpdf.(QDIM(; parms..., λ), outcomes1, outcomes2, won, ns, data)), λs)
    _, mxi = findmax(LLs)
    @test λs[mxi] ≈ parms.λ rtol = 1e-1

    w₁s = range(0.8 * parms.w₁, 1.2 * parms.w₁, length = 100)
    LLs = map(w₁ -> sum(logpdf.(QDIM(; parms..., w₁), outcomes1, outcomes2, won, ns, data)), w₁s)
    _, mxi = findmax(LLs)
    @test w₁s[mxi] ≈ parms.w₁ rtol = 1e-2
end

# @safetestset "pdf" begin
#     using QuantumDynamicInconsistencyModels
#     using Distributions
#     using Test
#     using Random 

#     Random.seed!(11214)

#     n = 5

#     model = QDIM(;μd=.5, γ=2)
#     data = rand(model, n)

#     for _ ∈ 1:10
#         μd = rand(Uniform(-1, 1))
#         γ = rand(Uniform(-2, 2))
#         LL1 = logpdf(QDIM(;μd, γ), n, data)
#         LL2 = log(pdf(QDIM(;μd, γ), n, data))
#         @test LL1 ≈ LL2
#     end
# end

@safetestset "predict_given_win" begin
    @safetestset "1" begin
        using QuantumDynamicInconsistencyModels
        using QuantumDynamicInconsistencyModels: predict_given_win
        using Test

        model = QDIM(; α = 0.70, λ = 2.0, w₁ = 0.50, m = 0.50, γ = 2.5)
        outcomes1 = [2, -1]
        outcomes2 = [2, -1]
        preds = predict_given_win(model, outcomes1, outcomes2)
        preds_true = [0.66995, 0.65928]

        @test preds ≈ preds_true atol = 1e-4
    end

    @safetestset "2" begin
        using QuantumDynamicInconsistencyModels
        using QuantumDynamicInconsistencyModels: predict_given_win
        using Test

        model = QDIM(; α = 0.80, λ = 1.5, w₁ = 0.50, m = 0.30, γ = -2.0)
        outcomes1 = [20, -22.0]
        outcomes2 = [20, -22.0]
        preds = predict_given_win(model, outcomes1, outcomes2)
        preds_true = [0.36311, 0.24679]

        @test preds ≈ preds_true atol = 1e-4
    end
end

@safetestset "predict_given_loss" begin
    @safetestset "1" begin
        using QuantumDynamicInconsistencyModels
        using QuantumDynamicInconsistencyModels: predict_given_loss
        using Test

        model = QDIM(; α = 0.70, λ = 2.0, w₁ = 0.50, m = 0.50, γ = 2.5)
        outcomes1 = [2, -1]
        outcomes2 = [2, -1]
        preds = predict_given_loss(model, outcomes1, outcomes2)
        preds_true = [0.66995, 0.69627]

        @test preds ≈ preds_true atol = 1e-4
    end

    @safetestset "2" begin
        using QuantumDynamicInconsistencyModels
        using QuantumDynamicInconsistencyModels: predict_given_loss
        using Test

        model = QDIM(; α = 0.80, λ = 1.5, w₁ = 0.50, m = 0.30, γ = -2.0)
        outcomes1 = [20, -22.0]
        outcomes2 = [20, -22.0]
        preds = predict_given_loss(model, outcomes1, outcomes2)
        preds_true = [0.36311, 0.57019]

        @test preds ≈ preds_true atol = 1e-4
    end
end

@safetestset "predict" begin
    @safetestset "loss1" begin
        using QuantumDynamicInconsistencyModels
        using QuantumDynamicInconsistencyModels: predict
        using Test

        model = QDIM(; α = 0.70, λ = 2.0, w₁ = 0.50, m = 0.50, γ = 2.5)
        outcomes1 = [2, -1]
        outcomes2 = [2, -1]
        preds = predict(model, outcomes1, outcomes2, false)
        preds_true = [0.56821, 0.10174, 0.11490, 0.21515]

        @test preds ≈ preds_true atol = 1e-4
    end

    @safetestset "loss2" begin
        using QuantumDynamicInconsistencyModels
        using QuantumDynamicInconsistencyModels: predict
        using Test

        model = QDIM(; α = 0.80, λ = 1.5, w₁ = 0.50, m = 0.30, γ = -2.0)
        outcomes1 = [20, -22.0]
        outcomes2 = [20, -22.0]
        preds = predict(model, outcomes1, outcomes2, false)
        preds_true = [0.25386, 0.10925, 0.25420, 0.38269]

        @test preds ≈ preds_true atol = 1e-4
    end

    @safetestset "win1" begin
        using QuantumDynamicInconsistencyModels
        using QuantumDynamicInconsistencyModels: predict
        using Test

        model = QDIM(; α = 0.70, λ = 2.0, w₁ = 0.50, m = 0.50, γ = 2.5)
        outcomes1 = [2, -1]
        outcomes2 = [2, -1]
        preds = predict(model, outcomes1, outcomes2, true)
        preds_true = [0.55582, 0.11413, 0.10880, 0.22125]

        @test preds ≈ preds_true atol = 1e-4
    end

    @safetestset "win2" begin
        using QuantumDynamicInconsistencyModels
        using QuantumDynamicInconsistencyModels: predict
        using Test

        model = QDIM(; α = 0.80, λ = 1.5, w₁ = 0.50, m = 0.30, γ = -2.0)
        outcomes1 = [20, -22.0]
        outcomes2 = [20, -22.0]
        preds = predict(model, outcomes1, outcomes2, true)
        preds_true = [0.17166, 0.19145, 0.11002, 0.52687]

        @test preds ≈ preds_true atol = 1e-4
    end
end

@safetestset "get_utility_diffs" begin
    @safetestset "1" begin
        using QuantumDynamicInconsistencyModels
        using QuantumDynamicInconsistencyModels: get_utility_diffs
        using Test

        model = QDIM(; α = 0.65, λ = 1.6, m = 0.50, w₁ = 0.5, γ = 2.2)
        outcomes1 = [2, -1]
        outcomes2 = [2, -1]
        d = get_utility_diffs(model, outcomes1, outcomes2)

        @test d[1] ≈ 0.162 atol = 0.001
        @test d[2] ≈ 0.8447 atol = 0.001
    end

    @safetestset "2" begin
        using QuantumDynamicInconsistencyModels
        using QuantumDynamicInconsistencyModels: get_utility_diffs
        using Test

        model = QDIM(; α = 0.50, λ = 2, m = 0.50, w₁ = 0.5, γ = 2.2)
        outcomes1 = [9, -4]
        outcomes2 = [9, -4]
        d = get_utility_diffs(model, outcomes1, outcomes2)
        # lose 4 in first gamble 
        d2 = (0.5 * (-4 + 9)^0.5 + -0.5 * 2 * abs(-4 + -4)^0.5) - -2 * abs(-4)^0.5

        # win 9 in first gamble 
        d1 = (0.5 * (9 + 9)^0.5 + 0.5 * abs(9 - 4)^0.5) - 9^0.5

        @test d[1] ≈ d1 atol = 1e-5
        @test d[2] ≈ d2 atol = 1e-5
    end
end

@safetestset "get_utility" begin
    using QuantumDynamicInconsistencyModels
    using QuantumDynamicInconsistencyModels: get_utility
    using Test

    utility = get_utility(4, 0.5, 2)
    @test utility ≈ 2

    utility = get_utility(-4, 0.5, 2)
    @test utility ≈ -4
end
