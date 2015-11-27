// Edge List Genome

use evo::prob::{Probability, ProbabilityValue};
use evo::crossover::linear_2point_crossover_random;
use rand::Rng;
use rand::distributions::{IndependentSample, Range, Weighted, WeightedChoice};
use petgraph::{Directed, Graph};
use petgraph::graph::NodeIndex;

#[derive(Clone, Debug)]
pub struct EdgeListGenome {
    max_nodes: usize,
    edges: Vec<(usize, usize)>,
}

#[derive(Clone)]
enum Mutation {
    Reverse,
    ModifySource,
    ModifyDestination,
}

impl EdgeListGenome {
    pub fn random<R: Rng>(rng: &mut R, max_nodes: usize) -> EdgeListGenome {
        let max_edges = 2 * max_nodes; // number of edges can be higher than max_nodes.
        let edge_range = Range::new(0, max_nodes);
        let num_edges = Range::new(1, max_edges).ind_sample(rng);
        let edges: Vec<_> = (0..num_edges)
                                .map(|_| (edge_range.ind_sample(rng), edge_range.ind_sample(rng)))
                                .collect();
        EdgeListGenome {
            max_nodes: max_nodes,
            edges: edges,
        }
    }

    #[inline]
    pub fn max_nodes(&self) -> usize {
        self.max_nodes
    }

    pub fn num_edges(&self) -> usize {
        self.edges.len()
    }

    pub fn mate<R: Rng>(rng: &mut R, p1: &EdgeListGenome, p2: &EdgeListGenome) -> EdgeListGenome {
        assert!(p1.max_nodes() == p2.max_nodes());
        EdgeListGenome {
            max_nodes: p1.max_nodes(),
            edges: linear_2point_crossover_random(rng, &p1.edges[..], &p2.edges[..]),
        }
    }

    pub fn mutate<R: Rng>(&mut self, rng: &mut R, mutate_elem_prob: Probability) {
        let edge_range = Range::new(0, self.max_nodes());
        let mut items = [Weighted {
                             weight: 2,
                             item: Mutation::Reverse,
                         },
                         Weighted {
                             weight: 1,
                             item: Mutation::ModifySource,
                         },
                         Weighted {
                             weight: 1,
                             item: Mutation::ModifyDestination,
                         }];
        let wc = WeightedChoice::new(&mut items);

        for edge in self.edges.iter_mut() {
            if rng.gen::<ProbabilityValue>().is_probable_with(mutate_elem_prob) {
                let new_edge = match wc.ind_sample(rng) {
                    Mutation::Reverse => {
                        (edge.1, edge.0)
                    }
                    Mutation::ModifySource => {
                        // XXX: Add an offset?
                        (edge_range.ind_sample(rng), edge.1)
                    }
                    Mutation::ModifyDestination => {
                        // XXX: Add an offset?
                        (edge.0, edge_range.ind_sample(rng))
                    }
                };
                *edge = new_edge;
            }
        }
    }

    pub fn to_graph(&self) -> Graph<(), (), Directed, u32> {
        let mut g: Graph<(), (), Directed, u32> = Graph::new();

        let n = self.max_nodes();

        for _ in 0..n {
            let _ = g.add_node(());
        }

        for &(i, j) in self.edges.iter() {
            g.add_edge(NodeIndex::new(i), NodeIndex::new(j), ());
        }

        return g;
    }
}
