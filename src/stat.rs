use std::f32::INFINITY;
use std::fmt::Debug;

#[derive(Debug)]
pub struct Stat<T: Debug> {
    pub min: T,
    pub max: T,
    pub avg: T,
}

impl Stat<f32> {
    pub fn from_iter<I>(iter: I) -> Stat<f32>
        where I: Iterator<Item = f32>
    {
        let mut min = INFINITY;
        let mut max = -INFINITY;
        let mut sum = 0.0;
        let mut cnt = 0usize;

        for x in iter {
            if x < min {
                min = x;
            }
            if x > max {
                max = x;
            }
            sum += x;
            cnt += 1;
        }

        Stat {
            min: min,
            max: max,
            avg: if cnt == 0 {
                0.0
            } else {
                sum / cnt as f32
            },
        }
    }
}
