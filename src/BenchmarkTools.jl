using BenchmarkTools
using Random

if !@isdefined(JackknifePlus)
    include("JackknifePlus.jl")
end
using .JackknifePlus

Random.seed!(42)

println("="^60)
println("Jackknife+ Scaling Benchmark: Naive Refit vs Fast Rank-1")
println("="^60)

for n in [50, 100, 200, 400]
    p = 5
    m = 20
    X_tr = randn(n, p)
    y_tr = X_tr * ones(p) .+ randn(n)
    X_te = randn(m, p)

    println("\n--> Testing Sample Size n = $n, Features p = $p, Test Points m = $m")
    
    print("  Naive LOO Refit:    ")
    b_naive = @benchmark predict_naive_loo($X_tr, $y_tr, $X_te; alpha=0.10) samples=20
    println("Median time: ", BenchmarkTools.prettytime(median(b_naive).time), 
            " | Allocations: ", median(b_naive).allocs)

    print("  Fast Rank-1 OLS:    ")
    b_fast = @benchmark begin
        model = fit_jackknife_plus($X_tr, $y_tr; fit_intercept=true)
        predict_intervals(model, $X_te; alpha=0.10)
    end samples=50
    println("Median time: ", BenchmarkTools.prettytime(median(b_fast).time), 
            " | Allocations: ", median(b_fast).allocs)

    speedup = median(b_naive).time / median(b_fast).time
    println("  >> Speedup Factor:  ", round(speedup, digits=1), "x faster")
end