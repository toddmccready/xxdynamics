use crate::utils::{iterate, resolve_domain, Sequence};
use polars::prelude::*;
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;

/// Finds equilibria of a map `f` over a given domain.
///
/// # Parameters
///
/// - `f`: A map of interest.
/// - `domain`: A tuple containing the lower and upper bounds for the relevant
///     domain.
/// - `n_seeds`: The number of initial function evaluations to perform when
///     searching for equilibria.
/// - `eq_tol`: The tolerance for determining if a point is an equilibrium.
/// - `co_tol`: The tolerance for how distant two points can be before being
///     assigned to different "equilibrium groups".
/// - `n_ref`: The number of points used for refining an equilibrium point.
pub fn find_equilibria(
    f: &(impl Fn(f64) -> f64 + Sync),
    domain: (f64, f64),
    n_seeds: u32,
    eq_tol: f64,
    co_tol: f64,
    n_ref: u32,
    n_threads: usize,
) -> Vec<f64> {
    // Get bounds from domain
    let (lb, ub): (f64, f64) = domain;

    // Split domain into chunks based on number of threads
    let chunk_stepsize: f64 = (ub - lb) / n_threads as f64;
    let chunks: Vec<Sequence> = (0..n_threads)
        .map(|i: usize| {
            Sequence::new(
                lb + i as f64 * chunk_stepsize,
                lb + (i + 1) as f64 * chunk_stepsize,
                n_seeds / n_threads as u32,
            )
        })
        .collect();

    // Create threadpool
    let pool = ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .build()
        .unwrap();

    // Iterate over chunks
    let approx_equilibria: Vec<Vec<f64>> = pool.install(|| {
        chunks
            .par_iter()
            .map(|seq: &Sequence| {
                // Allocate vector for approximate equilibria
                let mut approx_equilibria: Vec<f64> = Vec::new();

                // Iterate over seeds
                for seed in *seq {
                    // copy occurs
                    // Evaluate current seed
                    let seed_val: f64 = f(seed);

                    // Check equilibrium tolerance
                    if (seed_val - seed).abs() < eq_tol {
                        approx_equilibria.push(seed);
                    }
                }

                approx_equilibria
            })
            .collect()
    });

    // Collect parallel evaluations
    let approx_equilibria: Vec<f64> = approx_equilibria.into_iter().flatten().collect();

    // Check if equilibria exists
    if approx_equilibria.is_empty() {
        return approx_equilibria;
    }

    // Strip domain into regions with equilibria
    let intervals: Vec<(f64, f64)> = resolve_domain(&approx_equilibria, co_tol);

    // Initialize vector for refined equilibria
    let mut equilibria: Vec<f64> = Vec::new();

    // Iterate over intervals
    for (lb, ub) in intervals {
        // Identify best equilibrium value
        let mut best: f64 = lb;
        let mut best_diff: f64 = (f(best) - best).abs();

        // Seed interval with points to check
        let seq: Sequence = Sequence::new(lb, ub, n_ref);

        // Iterate over approximate equilibria
        let mut cur_diff: f64;
        for cur_eq in seq {
            cur_diff = (f(cur_eq) - cur_eq).abs();

            if cur_diff < best_diff {
                best = cur_eq;
                best_diff = cur_diff;
            }
        }

        // Include refined equilibrium point
        equilibria.push(best);
    }

    equilibria
}

/// Finds periodic points of a map `f` over a given domain.
///
/// # Parameters
///
/// - `f`: A map of interest.
/// - `n_iter`: `n_iter`-th map to evaluate.
/// - `domain`: A tuple containing the lower and upper bounds for the relevant
///     domain.
/// - `n_seeds`: The number of initial function evaluations to perform when
///     searching for equilibria.
/// - `eq_tol`: The tolerance for determining if a point is an equilibrium.
/// - `co_tol`: The tolerance for how distant two points can be before being
///     assigned to different "equilibrium groups".
/// - `n_ref`: The number of points used for refining an equilibrium point.
/// - `n_threads`: Number of threads to use when evaluating seeds.
pub fn find_periodics(
    f: &(impl Fn(f64) -> f64 + Sync),
    n_iter: i32,
    domain: (f64, f64),
    n_seeds: u32,
    eq_tol: f64,
    co_tol: f64,
    n_ref: u32,
    n_threads: usize,
) -> Vec<f64> {
    let f = iterate(f, n_iter);

    find_equilibria(&f, domain, n_seeds, eq_tol, co_tol, n_ref, n_threads)
}

/// Find local behaviour of a set of equilibria for a map `f`.
///
/// # Parameters
/// - `f`: A map of interest,
/// - `equilibria`: A set of equilibria to evaluate.
/// - `n_steps`: The number of steps to use when evaluating the forward
///     trajectory.
/// - `s_tol`: Tolerance used for assessing stability.
/// - `a_scale`: Scaling of `s_tol` to check for convergence.
pub fn find_behaviour(
    f: &impl Fn(f64) -> f64,
    equilibria: &[f64],
    n_steps: i32,
    s_tol: f64,
    a_scale: f64,
) -> (Vec<bool>, Vec<bool>) {
    // Containers for behaviour of equilibria
    let mut stable: Vec<bool> = Vec::with_capacity(equilibria.len());
    let mut attractive: Vec<bool> = Vec::with_capacity(equilibria.len());

    // Iterate over equilibria
    for e in equilibria {
        // Evaluate local behaviour
        let (cur_s, cur_a) = behaviour(&f, *e, n_steps, s_tol, a_scale);

        stable.push(cur_s);
        attractive.push(cur_a);
    }

    (stable, attractive)
}

/// Find local behaviour of a single equilibrium for a map `f`.
///
/// # Parameters
/// - `f`: A map of interest,
/// - `e`: An equilibrium to evaluate.
/// - `n_steps`: The number of steps to use when evaluating the forward
///     trajectory.
/// - `s_tol`: Tolerance used for assessing stability.
/// - `a_scale`: Scaling of `s_tol` to check for convergence.
fn behaviour(
    f: &impl Fn(f64) -> f64,
    e: f64,
    n_steps: i32,
    s_tol: f64,
    a_scale: f64,
) -> (bool, bool) {
    let mut stable: bool = true;
    let mut attractive: bool = true;

    // Check each side of equilibrium
    for x0 in vec![e - s_tol, e + s_tol] {
        // Initialize dynamical system
        let mut val: f64 = x0;

        // Evaluate forward trajectory
        for _ in 0..n_steps {
            val = f(val);

            // Check whether system has left tolerance neighbourhood
            if stable && (val - e).abs() > s_tol + 1e-15 {
                stable &= false;
            }
        }

        // Check whether system has converged
        if (val - e).abs() > s_tol * a_scale + 1e-15 {
            attractive = false;
        }
    }

    (stable, attractive)
}

/// Get bifurcation data for a parameterized map.
///
/// # Parameters
/// - `par_f`: A parameterized map of interest f(x, p).
/// - `par_space`: Parameter space under consideration(similar to `domain`).
/// - `n_pars`: Number of parameter values to evaluate.
/// - `domain`: A tuple containing the lower and upper bounds for the relevant
///     domain.
/// - `n_seeds`: The number of initial function evaluations to perform when
///     searching for equilibria.
/// - `eq_tol`: The tolerance for determining if a point is an equilibrium.
/// - `co_tol`: The tolerance for how distant two points can be before being
///     assigned to different "equilibrium groups".
/// - `n_ref`: The number of points used for refining an equilibrium point.
/// - `n_steps`: The number of steps to use when evaluating the forward
///     trajectory.
/// - `s_tol`: Tolerance used for assessing stability.
/// - `a_scale`: Scaling of `s_tol` to check for convergence.
/// - `max_period`: Maximum period for a periodic point to look for.
pub fn find_bifurcations(
    par_f: &(impl Fn(f64, f64) -> f64 + Sync),
    par_space: (f64, f64),
    n_pars: u32,
    domain: (f64, f64),
    n_seeds: u32,
    eq_tol: f64,
    co_tol: f64,
    n_ref: u32,
    n_steps: i32,
    s_tol: f64,
    a_scale: f64,
    max_period: i32,
    n_threads: usize,
) -> DataFrame {
    // Containers for recording bifurcation info
    let mut par_info: Vec<f64> = Vec::new();
    let mut state_info: Vec<f64> = Vec::new();
    let mut per_info: Vec<i32> = Vec::new();
    let mut stable_info: Vec<bool> = Vec::new();
    let mut attr_info: Vec<bool> = Vec::new();

    // Get bounds from parameter space
    let (p_lb, p_ub) = par_space;

    // Seed parameter space with parameter values to check
    let pars: Sequence = Sequence::new(p_lb, p_ub, n_pars);

    // Iterate over parameter values
    for p in pars {
        // Create function with fixed parameter
        let f = move |x: f64| par_f(x, p);

        // Store current parameters invariant sets
        let mut fixed_invariants: Vec<f64> = Vec::new();

        // Iterate over periods
        for period in 1..=max_period {
            // Create function iterate
            let fp = iterate(f, period);

            // Find equilibria
            let mut equilibria: Vec<f64> =
                find_equilibria(&fp, domain, n_seeds, eq_tol, co_tol, n_ref, n_threads);
            
            // Iterate over equilibria of previous iterates
            for &e in &fixed_invariants {
                // Keep current equilibria if not found before
                equilibria = equilibria
                    .iter()
                    .filter_map(|new_e| {
                        if (e - new_e).abs() > co_tol {
                            Some(*new_e)
                        } else {
                            None
                        }
                    })
                    .collect();
            }

            // Evaluate local behaviour of equilibria
            let (stable, attractive) = find_behaviour(&fp, &equilibria, n_steps, s_tol, a_scale);
            
            // Record invariant states
            fixed_invariants.extend(&equilibria);

            // Append information
            par_info.extend(&vec![p; equilibria.len()]);
            state_info.extend(&equilibria);
            per_info.extend(&vec![period; equilibria.len()]);
            stable_info.extend(&stable);
            attr_info.extend(&attractive);
        }
    }

    // Collect into dataframe
    let data: DataFrame = df!(
        "parameter" => par_info,
        "state" => state_info,
        "stable" => stable_info,
        "attractive" => attr_info,
        "period" => per_info
    )
    .unwrap();

    // Return information
    data
}

/// Find the attractors of a parameterized function
pub fn find_attractors(
    par_f: &(impl Fn(f64, f64) -> f64 + Sync),
    par_space: (f64, f64),
    n_pars: u32,
    x0: f64,
    n_burnin: u32,
    n_hist: u32
) -> DataFrame {
    // Containers for recording orbit info
    let mut data: Vec<DataFrame> = Vec::new();
    
    // Get bounds from parameter space
    let (p_lb, p_ub) = par_space;
    
    // Seed parameter space with parameter values to check
    let pars: Sequence = Sequence::new(p_lb, p_ub, n_pars);
    
    // Iterate over parameters
    for p in pars {
        // Create function with fixed parameter
        let f = move |x: f64| par_f(x, p);
        
        let mut x_cur: f64 = x0;
        // Run dynamical system for `n_warmup` iterations
        for _ in 0..n_burnin {
            x_cur = f(x_cur);
        }
        
        // Container for forward orbit
        let mut hist: Vec<f64> = Vec::<f64>::with_capacity(n_hist as usize);
        let mut pars: Vec<f64> = Vec::<f64>::with_capacity(n_hist as usize);
        
        // Record iterations
        for _ in 0..n_hist {
            x_cur = f(x_cur);
            
            hist.push(x_cur);
            pars.push(p);
        }
        
        // Record information
        let cur_data: DataFrame = df!(
            "orbit" => hist,
            "parameter" => pars
        ).unwrap();
        data.push(cur_data);
    }
    
    // Collect data into dataframe
    let mut final_data: DataFrame = data.pop().unwrap();
    
    for _ in 0..data.len() {
        final_data.vstack_mut(&data.pop().unwrap()).unwrap();
    }
    
    final_data
}