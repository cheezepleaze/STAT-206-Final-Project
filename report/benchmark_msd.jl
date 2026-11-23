using Pkg

using HTTP
using ZipFile
using CSV
using DataFrames
using Random
using LinearAlgebra
using Statistics
using BenchmarkTools

if !@isdefined(JackknifePlus)
    include("../src/JackknifePlus.jl")
end
using .JackknifePlus

Random.seed!(42)

# ==============================================================================
# 1. Download & Ingest YearPredictionMSD
# ==============================================================================
println("="^60)
println("YEARPREDICTIONMSD BENCHMARK FOR JACKKNIFE+")
println("="^60)

local_zip_path = "../data/YearPredictionMSD.txt.zip"
local_txt_path = "../data/YearPredictionMSD.txt"

if !isfile(local_txt_path)
    if !isfile(local_zip_path)
        println("--> Downloading YearPredictionMSD from UCI (~200 MB compressed)...")
        uci_url = "https://archive.ics.uci.edu/static/public/203/yearpredictionmsd.zip"
        HTTP.download(uci_url, local_zip_path)
    end

    println("--> Extracting archive...")
    r = ZipFile.Reader(local_zip_path)
    for f in r.files
        if endswith(f.name, ".txt")
            open(local_txt_path, "w") do out
                write(out, read(f))
            end
            break
        end
    end
    close(r)
end

println("--> Ingesting raw data (CSV parse)...")
df = CSV.read(local_txt_path, DataFrame; header=false, types=Float64)
println("Dataset loaded successfully: $(size(df, 1)) rows x $(size(df, 2)) columns.")

# ==============================================================================
# 2. Standard Train/Test Protocol Partitioning
# ==============================================================================
n_official_train = 463_715
total_rows = size(df, 1)

# Target: Column 1 (Release Year)
# Features: Columns 2 through 91 (90 timbre metrics)
y_all = Vector{Float64}(df[:, 1])
X_all = Matrix{Float64}(df[:, 2:end])

# Standardize features based on training statistics
println("--> Normalizing features using training distribution...")
X_train_raw = X_all[1:n_official_train, :]
y_train = y_all[1:n_official_train]

μ_x = mean(X_train_raw, dims=1)
σ_x = std(X_train_raw, dims=1)

# Guard against potential constant features (σ ≈ 0)
σ_x[σ_x .== 0.0] .= 1.0

X_train = (X_train_raw .- μ_x) ./ σ_x

# Prep test set
X_test_raw = X_all[n_official_train+1:end, :]
y_test = y_all[n_official_train+1:end]
X_test = (X_test_raw .- μ_x) ./ σ_x

println("Training dimensions: N = $(size(X_train, 1)), P = $(size(X_train, 2))")
println("Testing dimensions:  M = $(size(X_test, 1))")

# ==============================================================================
# 3. Fit Model Using Fast Rank-1 Updates (N ≈ 463k)
# ==============================================================================
println("\n--> Fitting Jackknife+ via Fast Rank-1 Inversion (N = 463,715)...")
t_fit = @elapsed begin
    model = fit_jackknife_plus(X_train, y_train; fit_intercept=true)
end
println("Model fitted in: $(round(t_fit, digits=3)) seconds")

# Verify the fundamental OLS leverage identity: sum(h_ii) == p
println("Model coefficients computed: $(length(model.beta)) parameters (including intercept).")

# ==============================================================================
# 4. Batched Interval Predictions
# ==============================================================================
function predict_intervals_batched(model, X_eval; alpha=0.10, batch_size=500)
    total_m = size(X_eval, 1)
    results = Matrix{Float64}(undef, total_m, 2)
    
    n_batches = Int(ceil(total_m / batch_size))
    for b in 1:n_batches
        start_idx = (b - 1) * batch_size + 1
        end_idx = min(b * batch_size, total_m)
        
        X_batch = X_eval[start_idx:end_idx, :]
        results[start_idx:end_idx, :] = predict_intervals(model, X_batch; alpha=alpha)
    end
    return results
end

# Evaluate predictive intervals on a subset of test samples
eval_test_size = 2_000
println("\n--> Generating 90% Predictive Intervals for $(eval_test_size) test points...")
alpha = 0.10

X_test_sub = X_test[1:eval_test_size, :]
y_test_sub = y_test[1:eval_test_size]

t_pred = @elapsed begin
    intervals = predict_intervals_batched(model, X_test_sub; alpha=alpha, batch_size=500)
end
println("Inference completed in: $(round(t_pred, digits=3)) seconds")

# ==============================================================================
# 5. Coverage and Efficiency Metrics
# ==============================================================================
lowers = intervals[:, 1]
uppers = intervals[:, 2]

covered_count = count((y_test_sub .>= lowers) .& (y_test_sub .<= uppers))
empirical_coverage = covered_count / eval_test_size
mean_width = mean(uppers .- lowers)

println("\n" * "="^60)
println("YEARPREDICTIONMSD EMPIRICAL VALIDATION")
println("="^60)
println("Nominal Target Coverage (1 - α):       ", round((1.0 - alpha) * 100, digits=1), "%")
println("Barber et al. Theoretical Floor (1 - 2α): ", round((1.0 - 2*alpha) * 100, digits=1), "%")
println("Observed Test Set Coverage:            ", round(empirical_coverage * 100, digits=2), "%")
println("Average Predictive Interval Width:     ", round(mean_width, digits=2), " years")

# ==============================================================================
# 6. Scaling Demonstration: Naive LOO vs. Rank-1 Inversion
# ==============================================================================
println("\n" * "="^60)
println("SCALING TEST ON SUBSET (N = 1,000, P = 90)")
println("="^60)

sub_n = 1_000
X_sub = X_train[1:sub_n, :]
y_sub = y_train[1:sub_n]
X_eval_mini = X_test[1:20, :]

print("  Naive LOO Refit (1,000 inversions): ")
t_naive = @elapsed predict_naive_loo(X_sub, y_sub, X_eval_mini; alpha=alpha)
println("$(round(t_naive, digits=3)) s")

print("  Fast Rank-1 Inversion:              ")
t_fast = @elapsed begin
    m_mini = fit_jackknife_plus(X_sub, y_sub; fit_intercept=true)
    predict_intervals(m_mini, X_eval_mini; alpha=alpha)
end
println("$(round(t_fast, digits=4)) s")

speedup = t_naive / t_fast
println("  >> Speedup on N=1,000: $(round(speedup, digits=1))x faster")
println("  >> Estimated Naive Time for full N=463,715: ~$(round((t_naive * (463_715 / sub_n)^2) / 3600, digits=1)) hours")