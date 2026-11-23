module JackknifePlus

using LinearAlgebra
using Statistics

export JackknifePlusOLS, fit_jackknife_plus, predict_intervals, predict_naive_loo

# Immutable container storing precomputed quantities for fast LOO inference.

struct JackknifePlusOLS{T<:AbstractFloat}
    beta::Vector{T}                 # model coefficients (p,)
    XtX_inv::Matrix{T}              # Precomputed (X'X)^(-1) (p x p)
    X_train::Matrix{T}              # Training the design matrix (n x p)
    residuals_loo::Vector{T}        # Absolute LOO residuals (n,)
    weighted_errors::Vector{T}      # (y - y_hat) ./ (1 - h_ii) (n,)
    n::Int
    p::Int
end

# Following function, fit_jackknife_plus, fits a fast OLS Jackknife+ estimator in O(np^2 + p^3)
# time using closed-form rank-1 Sherman-Morrison updates

function fit_jackknife_plus(X::AbstractMatrix{T}, y::AbstractVector;
                            fit_intercept::Bool = true) where {T<:AbstractFloat}
    n, p_raw = size(X)
    @assert n == length(y)                              # both x and y must match
    @assert n > (p_raw + (fit_intercept ? 1 : 0))       # sample size must exceed number of features

    # Augment design matrix with bias column if requested
    X_mat = fit_intercept ? hcat(ones(T,n), X) : Matrix{T}(X)
    p = size(X_mat, 2)

    # Compute (X'X)^(-1) via Cholesky decomposition for numerical stability
    XtX = Hermitian(X_mat' * X_mat)
    chol = cholesky(XtX)
    XtX_inv = Matrix{T}(chol \ I(p))

    # OLS fit
    beta = XtX_inv * (X_mat' * y)
    y_hat = X_mat * beta

    # Exact leverage scores h_ii = diag(X * (X'X)^(-1) * X')
    # Computed row-wise in O(n*p^2) without allocating an (n x n) matrix

    h = zeros(T, n)
    @inbounds for i in 1:n
        xi = view(X_mat, i, :)
        h[i] = dot(xi, XtX_inv * xi)
        @assert h[i] < 1.0
    end

    # Closed-form LOO residuals and weighted error vectors
    raw_res = y .- y_hat
    denom = 1.0 .- h
    weighted_errors = raw_res ./ denom
    residuals_loo = abs.(weighted_errors)

    return JackknifePlusOLS{T}(beta, XtX_inv, X_mat, residuals_loo, weighted_errors,n , p)
end

# Following function, predict_intervals, calculates (1 - alpha) predicition intervals for X_test
# via exact discrete quantiles of augmented LOO candidates

function predict_intervals(model::JackknifePlusOLS{T}, X_test::AbstractMatrix{T};
                            alpha::T = 0.10) where {T<:AbstractFloat}
    m = size(X_test, 1)

    # Handle intercept mathcing training data
    has_intercept = (model.p == size(X_test, 2) + 1)
    X_eval = has_intercept ? hcat(ones(T, m), X_test) : Matrix{T}(X_test)
    @assert size(X_eval, 2) == model.p          # Feature dimension mismatch

    # Model prediction on test points
    y_full_test = X_eval * model.beta

    # A = X_eval * (X'X)^(-1) * X_train' => shape (m, n)
    A = (X_eval * model.XtX_inv) * model.X_train'

    # Compute LOO test predictions
    # Matrix shape: (n_train, m_test)
    W = model.weighted_errors
    loo_preds = (ones(T, model.n) * y_full_test') .- (W .* A')

    # Candidate intervals: shape (n_train, m_test)
    lower_cands = loo_preds .- model.residuals_loo
    upper_cands = loo_preds .+ model.residuals_loo

    # Discrete quantile index selection (Barber et al. (2021))
    k_lower = max(1, Int(ceil(alpha * (model.n + 1))))
    k_upper = min(model.n, Int(ceil((1.0 - alpha) * (model.n + 1))))

    intervals = Matrix{T}(undef, m, 2)

    @inbounds for j in 1:m
        sort!(view(lower_cands, :, j))
        sort!(view(upper_cands, :, j))

        intervals[j, 1] = lower_cands[k_lower, j]
        intervals[j, 2] = upper_cands[k_upper, j]
    end

    return intervals
end

# Native O(n^2 * p) LOO refitting for testing and verification benchmarks

function predict_naive_loo(X_train::Matrix{Float64}, y_train::Vector{Float64},
                            X_test::Matrix{Float64}; alpha::Float64 = 0.10)
    n = size(X_train, 1)
    m = size(X_test, 1)

    X_tr = hcat(ones(n), X_train)
    X_te = hcat(ones(m), X_test)

    loo_residuals = zeros(n)
    loo_preds = zeros(n, m)

    for i in 1:n
        mask = [k != i for k in 1:n]
        b_i = X_tr[mask, :] \ y_train[mask]

        loo_residuals[i] = abs(y_train[i] - dot(X_tr[i, :], b_i))
        loo_preds[i, :] = X_te * b_i
    end

    lower_cands = loo_preds .- loo_residuals
    upper_cands = loo_preds .+ loo_residuals

    k_lower = max(1, Int(ceil(alpha * (n + 1))))
    k_upper = min(n, Int(ceil((1.0 - alpha) * (n + 1))))

    intervals = zeros(m, 2)
    for j in 1:m
        intervals[j, 1] = sort(lower_cands[:, j])[k_lower]
        intervals[j, 2] = sort(upper_cands[:, j])[k_upper]
    end
    return intervals
end

end