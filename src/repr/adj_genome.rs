// Adjacency Matrix Genome

use evo::bit_string::BitString;
use evo::{Individual, Probability};
use rand::Rng;
use petgraph::{Directed, Graph};
use petgraph::graph::NodeIndex;

#[derive(Clone, Debug)]
pub struct AdjGenome {
    matrix_n: usize,
    pub bits: BitString,
}

impl AdjGenome {
    pub fn new(bs: BitString, matrix_n: usize) -> AdjGenome {
        assert!(bs.len() == matrix_n * matrix_n);
        AdjGenome {
            bits: bs,
            matrix_n: matrix_n,
        }
    }

    pub fn random<R: Rng>(rng: &mut R, matrix_n: usize) -> AdjGenome {
        let iter = rng.gen_iter::<bool>().take(matrix_n * matrix_n);
        AdjGenome {
            bits: BitString::from_iter(iter),
            matrix_n: matrix_n,
        }
    }

    pub fn matrix_n(&self) -> usize {
        self.matrix_n
    }

    pub fn mutate<R: Rng>(&mut self, rng: &mut R, prop: Probability) {
        self.bits.flip_bits_randomly(rng, prop);
    }

    pub fn to_graph(&self) -> Graph<(), (), Directed, u32> {
        let b = &self.bits;
        let mut g: Graph<(), (), Directed, u32> = Graph::new();

        let n = self.matrix_n;
        assert!(b.len() == n * n);

        for _ in 0..n {
            let _ = g.add_node(());
        }

        let mut c = 0;
        for j in 0..n {
            for k in 0..n {
                if b.get(c) {
                    g.add_edge(NodeIndex::new(j), NodeIndex::new(k), ());
                }
                c += 1;
            }
        }

        return g;
    }
}

impl Individual for AdjGenome {}
