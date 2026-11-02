# Ipopt wrapper for the XPR fast global search.
#
# Ported from the seeding study harness (`julia/bakeoff/solvers.jl` in
# ../xpr-seeding-study). Three deliberate changes from that original:
#
#   * every name is prefixed `xpr_` so nothing in Julia's `Main` is clobbered by
#     an unrelated `include` elsewhere in the process;
#   * `XPR_SOLVERS_V1` is a sentinel so `xpr_backend.include_solvers()` can be
#     idempotent without re-reading the file;
#   * the f/g/h evaluation counters are kept. They are how you diagnose a solve
#     that is burning its iteration budget, and `n_cpu_limited` in the run
#     provenance depends on being able to tell a converged solve from a truncated
#     one.
#
# The problem is ALWAYS posed on the unit cube [0,1]^n: `XPR.load_data` is given
# `param_lower`/`param_upper`, which puts the physical<->normalised map inside
# ForwardDiff's tape, so `XPR_grad`/`XPR_hess` come back already chain-ruled in
# normalised coordinates. Calling these against a PHYSICAL-units load silently
# optimises nonsense, which is why the Python side asserts `is_normalized()`.

using Ipopt
using XPR

const XPR_SOLVERS_V1 = true

const XPR_NF = Ref(0)
const XPR_NG = Ref(0)
const XPR_NH = Ref(0)

xpr_reset_counts() = (XPR_NF[] = 0; XPR_NG[] = 0; XPR_NH[] = 0)
xpr_get_counts() = (nf = XPR_NF[], ng = XPR_NG[], nh = XPR_NH[])

function _xpr_solve(x0; exact_hess::Bool, max_cpu = 60.0, max_iter = 500,
                    print_level = 0, tol = 1e-8)
    n = length(x0)

    # Ipopt minimises, XPR returns a log-likelihood, so every objective and
    # derivative is negated.
    function eval_f(x)
        XPR_NF[] += 1
        val, _ = XPR.XPR_sim_ref(x)
        return -val
    end
    function eval_grad_f(x, grad)
        XPR_NG[] += 1
        grad .= -XPR.XPR_grad(x)
    end
    eval_g(x, g) = nothing
    eval_jac_g(x, rows, cols, values) = nothing

    function eval_h(x, rows, cols, obj_factor, lambda, values)
        if values === nothing
            idx = 1
            for r in 1:n, c in 1:r
                rows[idx] = r
                cols[idx] = c
                idx += 1
            end
        else
            XPR_NH[] += 1
            H = -XPR.XPR_hess(x)
            idx = 1
            for r in 1:n, c in 1:r
                values[idx] = obj_factor * H[r, c]
                idx += 1
            end
        end
        return nothing
    end
    eval_h_none(x, rows, cols, obj_factor, lambda, values) = nothing

    nele_h = exact_hess ? div(n * (n + 1), 2) : 0
    prob = CreateIpoptProblem(n, zeros(n), ones(n), 0, Float64[], Float64[], 0,
                              nele_h, eval_f, eval_g, eval_grad_f, eval_jac_g,
                              exact_hess ? eval_h : eval_h_none)
    AddIpoptIntOption(prob, "print_level", print_level)
    AddIpoptStrOption(prob, "sb", "yes")
    AddIpoptNumOption(prob, "max_cpu_time", max_cpu)
    AddIpoptIntOption(prob, "max_iter", max_iter)
    AddIpoptNumOption(prob, "tol", tol)
    AddIpoptNumOption(prob, "acceptable_tol", exact_hess ? 1e-7 : 1e-6)
    AddIpoptStrOption(prob, "hessian_approximation",
                      exact_hess ? "exact" : "limited-memory")
    AddIpoptStrOption(prob, "mu_strategy", "adaptive")
    prob.x = collect(Float64, x0)
    xpr_reset_counts()
    stat = IpoptSolve(prob)
    return (minimum = prob.x, status = stat, obj = prob.obj_val,
            nf = XPR_NF[], ng = XPR_NG[], nh = XPR_NH[])
end

# L-BFGS (`limited-memory`) is the workhorse: the study measured exact-Hessian
# Newton to be dominated by it at ~4x the cost per solve.
xpr_ipopt_bfgs(x0; kw...) = _xpr_solve(x0; exact_hess = false, kw...)
xpr_ipopt_newton(x0; kw...) = _xpr_solve(x0; exact_hess = true, kw...)
