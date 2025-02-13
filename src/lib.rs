pub mod finders;
pub mod utils;

#[cfg(test)]
mod tests {
    use super::*; // Bring functions from the outer scope into the test module

    // Define your tests here
    #[test]
    fn play() {
        fn f(x: f64) -> f64 {
            x.powi(2)
        }

        let e = finders::find_equilibria(&f, (-3.0, 3.0), 1e8 as u32, 1e-5, 1e-3, 1e8 as u32, 12);

        println!("hi");
        for i in e {
            println!("{i}");
        }
    }
}
