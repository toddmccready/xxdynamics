pub mod finders;
pub mod utils;

#[cfg(test)]
mod tests {
    use super::*; // Bring functions from the outer scope into the test module

    // Define your tests here
    #[test]
    fn play() {
        fn f(x: f64) -> f64 {
            -3.0*x.cos()
        }
        
        let e: Vec<f64> = finders::find_periodics(
            &f, 
            1,
            (-10.0, 10.0),
            (1e-15, 1e-4),
            100,
            1e-7,
            1e-3,
            8);
        
        for i in e {
            println!("Equilibria: {} | Difference: {}", i, (f(i) - i).abs());
        }
        
    }
}
