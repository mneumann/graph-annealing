use ::std::f32::INFINITY;
use ::std::fmt::Debug;
use ::evo::nsga2::MultiObjective3;

#[derive(Debug)]
pub struct Stat<T: Debug> {
    pub min: T,
    pub max: T,
    pub avg: T,
}

impl Stat<f32> {
    pub fn for_objectives(fit: &[MultiObjective3<f32>], i: usize) -> Stat<f32> {
        let min = fit.iter().fold(INFINITY, |acc, f| {
            let x = f.objectives[i];
            if x < acc {
                x
            } else {
                acc
            }
        });
        let max = fit.iter().fold(-INFINITY, |acc, f| {
            let x = f.objectives[i];
            if x > acc {
                x
            } else {
                acc
            }
        });
        let sum = fit.iter()
                     .fold(0.0, |acc, f| acc + f.objectives[i]);
        Stat {
            min: min,
            max: max,
            avg: if fit.is_empty() {
                0.0
            } else {
                sum / fit.len() as f32
            },
        }
    }
}
