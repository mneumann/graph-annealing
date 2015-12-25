use std::fmt::Debug;

#[derive(Debug)]
pub struct Stat<T: Debug> {
    pub min: T,
    pub max: T,
    pub avg: T,
}

impl Stat<f32> {
    pub fn from_iter<I>(mut iter: I) -> Option<Stat<f32>>
        where I: Iterator<Item = f32>
    {
        let mut min;
        let mut max;
        let mut sum;
        let mut cnt;

        if let Some(x) = iter.next() {
            min = x;
            max = x;
            sum = x;
            cnt = 1;
        } else {
            return None;
        }

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

        Some(Stat {
            min: min,
            max: max,
            avg: sum / cnt as f32,
        })
    }
}
