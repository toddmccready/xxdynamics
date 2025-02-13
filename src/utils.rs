#[allow(dead_code)]
/// A sequence of values over an interval.
#[derive(Copy, Clone)]
pub struct Sequence {
    a: f64,
    b: f64,
    n: u32,
    current: u32,
    step: f64,
}

impl Sequence {
    pub fn new(a: f64, b: f64, n: u32) -> Self {
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

pub fn resolve_domain(x: &[f64], tol: f64) -> Vec<(f64, f64)> {
    if x.is_empty() {
        panic!("x argument was empty");
    }

    // Container for intervals containing groups
    let mut intervals: Vec<(f64, f64)> = Vec::<(f64, f64)>::new();
    
    // Bounds for current interval
    let mut lb: f64 = x[0];
    let mut ub: f64;

    // Iterate over values
    for i in 1..x.len() {
        let diff: f64 = (x[i] - x[i - 1]).abs();

        if diff > tol {
            ub = x[i - 1];
            intervals.push((lb, ub));

            lb = x[i];
        }
    }

    // Include last value
    ub = x[x.len() - 1];
    intervals.push((lb, ub));

    intervals
}

/// Iterate a function multiple times
pub fn iterate(f: impl Fn(f64) -> f64, n: i32) -> impl Fn(f64) -> f64 {
    move |mut x: f64| {
        for _ in 0..n {
            x = f(x);
        }
        x
    }
}
