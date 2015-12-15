use rand::distributions::{IndependentSample, Range, Sample, Weighted};
use rand::Rng;

/// A distribution that selects from a finite collection of weighted items.
///
/// Each item has an associated weight that influences how likely it
/// is to be chosen: higher weight is more likely.
///
/// The `Clone` restriction is a limitation of the `Sample` and
/// `IndependentSample` traits. Note that `&T` is (cheaply) `Clone` for
/// all `T`, as is `u32`, so one can store references or indices into
/// another vector.
///
/// # Example
///
/// ```rust,ignore
/// use rand::distributions::{Weighted, IndependentSample};
///
/// let items = vec!(Weighted { weight: 2, item: 'a' },
///                      Weighted { weight: 4, item: 'b' },
///                      Weighted { weight: 1, item: 'c' });
/// let owc = OwnedWeightedChoice::new(items);
/// let mut rng = rand::thread_rng();
/// for _ in 0..16 {
///      // on average prints 'a' 4 times, 'b' 8 and 'c' twice.
///      println!("{}", wc.ind_sample(&mut rng));
/// }
/// ```
pub struct OwnedWeightedChoice<T> {
    items: Vec<Weighted<T>>,
    weight_range: Range<u32>,
}

impl<T: Clone> OwnedWeightedChoice<T> {
    /// Create a new `OwnedWeightedChoice`.
    ///
    /// Panics if:
    /// - `v` is empty
    /// - the total weight is 0
    /// - the total weight is larger than a `u32` can contain.
    pub fn new(mut items: Vec<Weighted<T>>) -> OwnedWeightedChoice<T> {
        // strictly speaking, this is subsumed by the total weight == 0 case
        assert!(!items.is_empty(),
                "WeightedChoice::new called with no items");

        let mut running_total: u32 = 0;

        // we convert the list from individual weights to cumulative
        // weights so we can binary search. This *could* drop elements
        // with weight == 0 as an optimisation.
        for item in items.iter_mut() {
            running_total = match running_total.checked_add(item.weight) {
                Some(n) => n,
                None => {
                    panic!("WeightedChoice::new called with a total weight larger than a u32 can \
                            contain")
                }
            };

            item.weight = running_total;
        }
        assert!(running_total != 0,
                "WeightedChoice::new called with a total weight of 0");

        OwnedWeightedChoice {
            items: items,
            // we're likely to be generating numbers in this range
            // relatively often, so might as well cache it
            weight_range: Range::new(0, running_total),
        }
    }
}

impl<T: Clone> Sample<T> for OwnedWeightedChoice<T> {
    fn sample<R: Rng>(&mut self, rng: &mut R) -> T {
        self.ind_sample(rng)
    }
}

impl<T: Clone> IndependentSample<T> for OwnedWeightedChoice<T> {
    fn ind_sample<R: Rng>(&self, rng: &mut R) -> T {
        // we want to find the first element that has cumulative
        // weight > sample_weight, which we do by binary since the
        // cumulative weights of self.items are sorted.

        // choose a weight in [0, total_weight)
        let sample_weight = self.weight_range.ind_sample(rng);

        // short circuit when it's the first item
        if sample_weight < self.items[0].weight {
            return self.items[0].item.clone();
        }

        let mut idx = 0;
        let mut modifier = self.items.len();

        // now we know that every possibility has an element to the
        // left, so we can just search for the last element that has
        // cumulative weight <= sample_weight, then the next one will
        // be "it". (Note that this greatest element will never be the
        // last element of the vector, since sample_weight is chosen
        // in [0, total_weight) and the cumulative weight of the last
        // one is exactly the total weight.)
        while modifier > 1 {
            let i = idx + modifier / 2;
            if self.items[i].weight <= sample_weight {
                // we're small, so look to the right, but allow this
                // exact element still.
                idx = i;
                // we need the `/ 2` to round up otherwise we'll drop
                // the trailing elements when `modifier` is odd.
                modifier += 1;
            } else {
                // otherwise we're too big, so go left. (i.e. do
                // nothing)
            }
            modifier /= 2;
        }
        return self.items[idx + 1].item.clone();
    }
}
