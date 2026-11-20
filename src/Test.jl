using Test
using Random
using LinearAlgebra
using Statistics

if !@isdefined(JackknifePlus)
    include("JackknifePlus.jl")
end
using .JackknifePlus

@testset "Jackknife+ Exactness and Coverage Tests" begin
    Random.seed!(42)

    @testset "Rank-1 Update vs. Naive Refit Equivalence" begin
        n, p, m = 60, 4, 10
        X_train = randn(n, p)
        y_train = X_train * [1.5, -2.0, 0.5, -1.0] .+ randn(n) .* 0.5
        X_test  = randn(m, p)

        # 1. Fast Rank-1 Model
        model_fast = fit_jackknife_plus(X_train, y_train; fit_intercept=true)
        intervals_fast = predict_intervals(model_fast, X_test; alpha=0.10)

        # 2. Naive LOO Model
        intervals_naive = predict_naive_loo(X_train, y_train, X_test; alpha=0.10)

        # Max difference should be at machine precision limits
        max_diff = maximum(abs.(intervals_fast .- intervals_naive))
        @info "Numerical tolerance check" MaxAbsoluteDifference=max_diff
        @test max_diff < 1e-12
    end

    @testset "Monte Carlo Empirical Coverage Verification" begin
        n_sims = 1000
        n_train = 50
        p = 3
        alpha = 0.10 # Target 90% coverage; theoretical bound is >= 80%
        true_beta = [2.0, -1.5, 0.8]

        covered = 0
        widths = Float64[]

        for _ in 1:n_sims
            X_tr = randn(n_train, p)
            y_tr = X_tr * true_beta .+ randn(n_train)

            # Draw a single unseen test point
            X_te = randn(1, p)
            y_te = (X_te * true_beta)[1] + randn()

            model = fit_jackknife_plus(X_tr, y_tr; fit_intercept=true)
            interval = predict_intervals(model, X_te; alpha=alpha)

            low, high = interval[1, 1], interval[1, 2]
            push!(widths, high - low)

            if low <= y_te <= high
                covered += 1
            end
        end

        empirical_coverage = covered / n_sims
        avg_width = mean(widths)

        @info "Simulation Results" Nominal=(1.0 - alpha) Empirical=empirical_coverage TheoreticalBound=(1.0 - 2*alpha) MeanWidth=avg_width

        # Theoretical guarantee: coverage must be >= 1 - 2*alpha (80%)
        @test empirical_coverage >= (1.0 - 2.0 * alpha)
        # Typically empirical coverage is very close to nominal (1 - alpha = 90%)
        @test empirical_coverage >= 0.88
    end
end