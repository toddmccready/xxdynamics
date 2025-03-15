#[allow(dead_code)]
/// A sequence of values over an interval.
pub struct Sequence {
    a: f64,
    b: f64,
    n: u64,
    current: u64,
    step: f64,
}

impl Sequence {
    pub fn new(a: f64, b: f64, n: u64) -> Self {
        let step: f64 = (b - a) / (n as f64 - 1.0);
        Sequence {
            a,
            b,
            n,
            current: 0,
            step,
        }
    }
}

impl Iterator for Sequence {
    type Item = f64;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current < self.n {
            let result: f64 = self.a + self.step * (self.current as f64);
            self.current += 1;

            Some(result)
        } else {
            None
        }
    }
}

/// Refine a set of equilibria
pub fn refiner(f: &(impl Fn(f64) -> f64 + Sync), eq: &[f64], tol: f64) -> Vec<f64> {
    // Check if input collection of equilibria is empty
    if eq.is_empty() {
        panic!("`eq` argument was empty");
    }
    
    // Initialize with the first equilibrium
    let mut ref_equilibria: Vec<f64> = vec![eq[0]];
    
    let mut ref_eq: &mut f64; // Mutable reference to a refined equilibrium
    let mut distance: f64; // Distance between refined and current equilibrium
    let mut diff_r: f64; // f(x) = x approximation error for refined
    let mut diff_c: f64; // f(x) = x approximation error for current

    // Iterate through the remaining equilibria
    for e in eq.iter().skip(1) {
        // Get the current equilibrium being refined
        ref_eq = ref_equilibria.last_mut().unwrap();
        
        // Compute distance
        distance = (*ref_eq - *e).abs();
        
        // If the state is nearby:
        if distance <= tol {
            // Compute how well they approximate f(x) = x
            diff_r = (f(*ref_eq) - *ref_eq).abs();
            diff_c = (f(*e) - *e).abs();
            
            // Replace the last refined point if better fit is found
            if diff_c < diff_r {
                *ref_eq = *e; 
            }
        } 
        // If the state is far:
        else { 
            // Add the current state to the refined list.
            ref_equilibria.push(*e);
        }
    }
    
    ref_equilibria
}


/// Iterate a function multiple times
pub fn iterate(f: impl Fn(f64) -> f64, n: u64) -> impl Fn(f64) -> f64 {
    move |mut x: f64| {
        for _ in 0..n {
            x = f(x);
        }
        x
    }
}
