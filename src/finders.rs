use core::f64;

use crate::utils::{iterate, refiner, Sequence};
use polars::prelude::*;
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;


/// Adaptively find equilibria for a given map within a certain interval.
///
/// # Parameters
///
/// - `f`: A map of interest.
/// - `state_space`: A tuple containing the lower and upper bounds for the relevant
///     state space.
/// - `stepsize_range`: The minimum and maximum stepsize allowed.
/// - `eq_tol`: The tolerance for determining if a point is an equilibrium.
pub fn adaptive_finder(
    f: &(impl Fn(f64) -> f64 + Sync),
    state_space: (f64, f64),
    stepsize_range: (f64, f64),
    eq_tol: f64
) -> Vec<f64> {
    // Container for equilibria
    let mut equilibria: Vec<f64> = vec![];
    
    // Unpack state space and stepsize ranges
    let (lb, ub) = state_space;
    let (min_step, max_step) = stepsize_range;
    
    let mut tracking: bool = false; // Tracker for distinct equilibria regions
    let mut decreasing: bool; // Tracker for whether difference is decreasing
    let mut cur_state: f64 = lb; // State of current iteration
    let mut cur_diff: f64; // |f(state) - state| of current iteration
    let mut min_diff: f64 = f64::INFINITY; // Min. diff for a unique region
    let mut max_diff: f64 = f64::NAN; // Max. diff observed
    let mut cur_step: f64; // Current iteration's stepsize
    
    // Iterate until upper bound is reached
    while cur_state <= ub {
        // Compute difference
        cur_diff = (f(cur_state) - cur_state).abs();
        
        // Check if decreasing
        decreasing = cur_diff <= min_diff;
        
        // Check equilibrium tolerance
        if cur_diff < eq_tol {
            if tracking & decreasing {            
                // Replace last equilibrium
                if let Some(last) = equilibria.last_mut() {
                    *last = cur_state;
                }
                
                // Update minimum difference
                min_diff = cur_diff;
            }
            else if !tracking {
                // Push new equilibrium
                equilibria
                .push(cur_state);
                
                // Reset min_diff and update tracker
                min_diff = cur_diff;
                tracking = true;
            }
        }
        else {
            tracking = false;
        }
        
        // Update maximum difference
        max_diff = f64::max(cur_diff, max_diff);
        
        // Compute current stepsize
        if decreasing {
            cur_step = min_step + (max_step - min_step) * (cur_diff / max_diff);
        } else { // Pull stepsize upwards if not decreasing
            cur_step = min_step + (max_step - min_step) * (cur_diff / max_diff).sqrt();
        }
        
        // Move to next state
        cur_state = cur_state + cur_step;
    }

    equilibria
}


/// Finds equilibria of a map `f` over a given domain.
pub fn find_equilibria(
    f: &(impl Fn(f64) -> f64 + Sync),
    state_space: (f64, f64),
    stepsize_range: (f64, f64),
    n_partitions: u64,
    eq_tol: f64,
    co_tol: f64,
    n_threads: usize,
) -> Vec<f64> {
    // Get bounds from state space
    let (lb, ub): (f64, f64) = state_space;
    
    // Compute length of partitions
    let length: f64 = (ub - lb) / (n_partitions as f64);
    
    // Check if stepsizes are less than partition length
    if (length, length) < stepsize_range {
        panic!("Stepsizes should not be less than partition length.");
    }
    
    // Partition state space
    let partitions: Vec<(f64, f64)> = (0..n_partitions).into_iter()
        .map(|i| {
            (
                lb + (i as f64) * length, 
                lb + ((i + 1) as f64) * length
            )
        })
        .collect();
    
    // Create threadpool
    let pool = ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .build()
        .unwrap();
    
    // Iterate over partition in parallel
    let equilibria: Vec<Vec<f64>> = pool.install(|| {
        partitions
            .into_par_iter()
            .map(|cur_state_space | {
                // Find equilibria in current partition
                adaptive_finder(f, cur_state_space, stepsize_range, eq_tol)
            })
            .collect()
    });
    
    // Collect parallel evaluations
    let equilibria: Vec<f64> = equilibria
        .into_iter()
        .flatten()
        .collect();
    
    // Check if any equilibria exists
    if equilibria.is_empty() {
        return equilibria;
    }
    
    
    let equilibria: Vec<f64> = refiner(&f, &equilibria, co_tol);
    
    equilibria
}

/// Finds periodic points of a map `f` over a given domain.
///
/// # Parameters
///
/// - `f`: A map of interest.
/// - `n_iter`: `n_iter`-th map to evaluate.
/// - `state_space`: A tuple containing the lower and upper bounds for the relevant
///     state space.
/// - `n_seeds`: The number of initial function evaluations to perform when
///     searching for equilibria.
/// - `eq_tol`: The tolerance for determining if a point is an equilibrium.
/// - `co_tol`: The tolerance for how distant two points can be before being
///     assigned to different "equilibrium groups".
/// - `n_threads`: Number of threads to use when evaluating seeds.
pub fn find_periodics(
    f: &(impl Fn(f64) -> f64 + Sync),
    n_iter: u64,
    state_space: (f64, f64),
    stepsize_range: (f64, f64),
    n_partitions: u64,
    eq_tol: f64,
    co_tol: f64,
    n_threads: usize,
) -> Vec<f64> {
    let f = iterate(f, n_iter);
    
    find_equilibria(&f, state_space, stepsize_range, n_partitions, eq_tol, co_tol, n_threads)
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
    n_steps: u64,
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
    n_steps: u64,
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
/// - `par_f`: A parameterized map of interest.
/// - `par_space`: The range of parameters to evaluate.
/// - `n_pars`: The number of parameters to evaluate.
/// - `state_space`: The state space to scan for invariant sets.
/// - `stepsize_range`: The minimum and maximum stepsizes allowed for the 
/// adaptive finder.
/// - `n_partitions`: The number of intervals to partition the state space 
/// into.
/// - `eq_tol`: The equilibrium tolerance.
/// - `co_tol`: The coalesence tolerance.
/// - `n_steps`: The number of steps to evaluate local behaviour.
/// - `s_tol`: The tolerance for determining stability.
/// - `a_scale`: The scaling factor for determining attraction.
/// - `max_period`: The maximum period to look for.
/// - `n_threads`: The number of threads to use.
pub fn find_bifurcations(
    par_f: &(impl Fn(f64, f64) -> f64 + Sync),
    par_space: (f64, f64),
    n_pars: u64,
    state_space: (f64, f64),
    stepsize_range: (f64, f64),
    n_partitions: u64,
    eq_tol: f64,
    co_tol: f64,
    n_steps: u64,
    s_tol: f64,
    a_scale: f64,
    max_period: u64,
    n_threads: usize,
) -> DataFrame {
    // Containers for recording bifurcation info
    let mut par_info: Vec<f64> = Vec::new();
    let mut state_info: Vec<f64> = Vec::new();
    let mut per_info: Vec<u64> = Vec::new();
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
                find_periodics(&f, period, state_space, stepsize_range, n_partitions, eq_tol, co_tol, n_threads);
            
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
    n_pars: u64,
    init_states: &[f64],
    n_burnin: u64,
    n_hist: u64
) -> DataFrame {
    // Containers for recording orbit info
    let mut par_info: Vec<f64> = Vec::<f64>::new();
    let mut init_info: Vec<f64> = Vec::<f64>::new();
    let mut orbit_info: Vec<f64> = Vec::<f64>::new();
    
    // Get bounds
    let (p_lb, p_ub) = par_space;
    
    // Seed parameter space with parameter values to check
    let pars: Sequence = Sequence::new(p_lb, p_ub, n_pars);
    
    // Iterate over parameters
    for p in pars {
        // Create function with fixed parameter
        let f = move |x: f64| par_f(x, p);
        
        // Iterate over initial states
        for x0 in init_states {
            let mut x_cur: f64 = x0.clone();
            
            // Run dynamical system for `n_burnin` iterations
            for _ in 0..n_burnin {
                x_cur = f(x_cur);
            }
            
            // Record orbit iterations
            for _ in 0..n_hist {
                x_cur = f(x_cur);
                
                par_info.push(p);
                init_info.push(*x0);
                orbit_info.push(x_cur);
            }
        }
    }
    
    // Record information
    let data: DataFrame = df!(
        "parameter" => par_info,
        "initial_state" => init_info,
        "orbit" => orbit_info
    ).unwrap();
    
    data
}

pub fn cobweb(
    f: &(impl Fn(f64) -> f64 + Sync),
    init_states: &[f64],
    n_burnin: u64,
    n_iters: u64
) -> DataFrame {
    let mut x_info: Vec<f64> = 
        Vec::with_capacity(1 + 2 * init_states.len() * (n_iters as usize));
    let mut y_info: Vec<f64> = 
        Vec::with_capacity(1 + 2 * init_states.len() * (n_iters as usize));
    let mut init_info: Vec<f64> = 
        Vec::with_capacity(1 + 2 * init_states.len() * (n_iters as usize));
    
    // Iterate over initial states
    for x0 in init_states {
        let mut x_cur: f64 = x0.clone();
        
        // Go through burn-in iterations
        for _ in 0..n_burnin {
            x_cur = f(x_cur);
        }
        
        // Initialize cobweb
        x_info.push(x_cur);
        y_info.push(0.0);
        init_info.push(*x0);
        
        // Start cobwebbing
        for _ in 0..n_iters {
            // Move vertically to map
            x_info.push(x_cur);
            y_info.push(f(x_cur));
            init_info.push(*x0);
            
            // Update state
            x_cur = f(x_cur);
            
            // Move horizontally to y=x
            x_info.push(x_cur);
            y_info.push(x_cur);
            init_info.push(*x0);
        }
    }
    
    let data: DataFrame = df!(
        "initial_state" => init_info,
        "x" => x_info,
        "y" => y_info
    ).unwrap();
    
    return data
}
