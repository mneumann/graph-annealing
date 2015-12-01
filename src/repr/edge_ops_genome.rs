// Edge Operation Genome

use evo::prob::{Probability, ProbabilityValue};
use evo::crossover::linear_2point_crossover_random;
use evo::nsga2::Mate;
use rand::Rng;
use rand::distributions::{IndependentSample, Weighted};
use petgraph::{Directed, Graph};
use petgraph::graph::NodeIndex;
use graph_edge_evolution::{EdgeOperation, GraphBuilder};
use owned_weighted_choice::OwnedWeightedChoice;

#[derive(Clone, Debug)]
pub struct EdgeOpsGenome {
    edge_ops: Vec<EdgeOperation<f32, ()>>,
}

#[derive(Copy, Clone)]
pub enum Op {
    Dup,
    Split,
    Loop,
    Merge,
    Next,
    Parent,
    Reverse,
}

pub struct Toolbox {
    weighted_op_choice: OwnedWeightedChoice<Op>,
    prob_mutate_elem: Probability,
}

impl Mate<EdgeOpsGenome> for Toolbox {
    fn mate<R: Rng>(&mut self,
                    rng: &mut R,
                    p1: &EdgeOpsGenome,
                    p2: &EdgeOpsGenome)
                    -> EdgeOpsGenome {

        let mut child = EdgeOpsGenome {
            edge_ops: linear_2point_crossover_random(rng, &p1.edge_ops[..], &p2.edge_ops[..]),
        };
        self.mutate(rng, &mut child);
        child
    }
}

impl Toolbox {
    pub fn mutate<R: Rng>(&self, rng: &mut R, ind: &mut EdgeOpsGenome) {
        for edge_op in ind.edge_ops.iter_mut() {
            if rng.gen::<ProbabilityValue>().is_probable_with(self.prob_mutate_elem) {
                let new_edge_op = match edge_op {
                    &mut EdgeOperation::Merge{..} => {
                        EdgeOperation::Merge { n: rng.gen::<u32>() }
                    }
                    &mut EdgeOperation::Next{..} => {
                        EdgeOperation::Next { n: rng.gen::<u32>() }
                    }
                    &mut EdgeOperation::Parent{..} => {
                        EdgeOperation::Parent { n: rng.gen::<u32>() }
                    }
                    ref op => {
                        (*op).clone()
                    }
                };

                *edge_op = new_edge_op;
            }
        }
    }

    pub fn new(prob_mutate_elem: Probability, weighted_op_choices: &[(Op, u32)]) -> Toolbox {
        let mut w = Vec::new();
        for &(op, weight) in weighted_op_choices {
            if weight > 0 {
                // an operation with weight=0 cannot be selected
                w.push(Weighted {
                    weight: weight,
                    item: op,
                });
            }
        }
        Toolbox {
            prob_mutate_elem: prob_mutate_elem,
            weighted_op_choice: OwnedWeightedChoice::new(w),
        }
    }

    fn generate_random_edge_operation<R: Rng>(&self, rng: &mut R) -> EdgeOperation<f32, ()> {
        match self.weighted_op_choice.ind_sample(rng) {
            Op::Dup => {
                EdgeOperation::Duplicate { weight: 0.0 }
            }
            Op::Split => {
                EdgeOperation::Split { weight: 0.0 }
            }
            Op::Loop => {
                EdgeOperation::Loop { weight: 0.0 }
            }
            Op::Merge => {
                EdgeOperation::Merge { n: rng.gen::<u32>() }
            }
            Op::Next => {
                EdgeOperation::Next { n: rng.gen::<u32>() }
            }
            Op::Parent => {
                EdgeOperation::Parent { n: rng.gen::<u32>() }
            }
            Op::Reverse => {
                EdgeOperation::Reverse
            }
        }
    }

    pub fn random_genome<R: Rng>(&self, rng: &mut R, len: usize) -> EdgeOpsGenome {
        let edge_ops: Vec<_> = (0..len)
                                   .map(|_| self.generate_random_edge_operation(rng))
                                   .collect();

        EdgeOpsGenome { edge_ops: edge_ops }
    }
}


impl EdgeOpsGenome {
    pub fn len(&self) -> usize {
        self.edge_ops.len()
    }

    pub fn to_graph(&self) -> Graph<(), (), Directed, u32> {
        let mut builder = GraphBuilder::new();
        for op in self.edge_ops.iter() {
            builder.apply_operation(op.clone());
        }
        let edge_list = builder.to_edge_list();

        let mut g: Graph<(), (), Directed, u32> = Graph::new();
        let n = edge_list.len();
        for _ in 0..n {
            let _ = g.add_node(());
        }

        for (i, node) in edge_list.iter().enumerate() {
            for &(j, _weight) in node.iter() {
                g.add_edge(NodeIndex::new(i), NodeIndex::new(j), ());
            }
        }

        return g;
    }
}
