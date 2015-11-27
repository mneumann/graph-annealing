// Edge Operation Genome

use evo::prob::{Probability, ProbabilityValue};
use evo::crossover::linear_2point_crossover_random;
use rand::Rng;
use rand::distributions::{IndependentSample, Range, Weighted, WeightedChoice};
use petgraph::{Directed, Graph};
use petgraph::graph::NodeIndex;
use graph_edge_evolution::{EdgeOperation, GraphBuilder}; 

#[derive(Clone, Debug)]
pub struct EdgeOpsGenome {
    edge_ops: Vec<EdgeOperation<f32, ()>>,
}

#[derive(Clone)]
enum Op {
   Dup, Split, Loop, Merge, Next, Parent, Reverse
}

fn generate_random_edge_operation<R:Rng>(rng: &mut R) -> EdgeOperation<f32, ()> {
    let mut items = [Weighted {
                         weight: 1,
                         item: Op::Dup,
                     },
                     Weighted {
                         weight: 3,
                         item: Op::Split
                     },
                     Weighted {
                         weight: 1,
                         item: Op::Loop
                     },
                     Weighted {
                         weight: 1,
                         item: Op::Merge
                     },
                     Weighted {
                         weight: 2,
                         item: Op::Next
                     },
                     Weighted {
                         weight: 2,
                         item: Op::Parent
                     },
                     Weighted {
                         weight: 2,
                         item: Op::Reverse
                     },
                     ];
    let wc = WeightedChoice::new(&mut items);
    match wc.ind_sample(rng) {
        Op::Dup => {
            EdgeOperation::Duplicate{weight: 0.0}
        }
        Op::Split => {
            EdgeOperation::Split{weight: 0.0}
        }
        Op::Loop => {
            EdgeOperation::Split{weight: 0.0}
        }
        Op::Merge => {
            EdgeOperation::Merge{n: rng.gen::<u32>()} 
        }
        Op::Next => {
            EdgeOperation::Next{n: rng.gen::<u32>()} 
        }
        Op::Parent => {
            EdgeOperation::Parent{n: rng.gen::<u32>()} 
        }
        Op::Reverse => {
            EdgeOperation::Reverse
        }
    }
}

impl EdgeOpsGenome {
    pub fn random<R: Rng>(rng: &mut R, upper_num_edge_ops: usize) -> EdgeOpsGenome {
        let num_edge_ops = Range::new(1, upper_num_edge_ops).ind_sample(rng);  
        let edge_ops: Vec<_> = (0..num_edge_ops)
                                .map(|_| generate_random_edge_operation(rng)).collect();

        EdgeOpsGenome {
            edge_ops: edge_ops,
        }
    }

    pub fn len(&self) -> usize { self.edge_ops.len() }

    pub fn mate<R: Rng>(rng: &mut R, p1: &EdgeOpsGenome, p2: &EdgeOpsGenome) -> EdgeOpsGenome {
        EdgeOpsGenome {
            edge_ops: linear_2point_crossover_random(rng, &p1.edge_ops[..], &p2.edge_ops[..])
        }
    }

    pub fn mutate<R: Rng>(&mut self, rng: &mut R, mutate_elem_prob: Probability) {
        for edge_op in self.edge_ops.iter_mut() {
            if rng.gen::<ProbabilityValue>().is_probable_with(mutate_elem_prob) {
                let new_edge_op = match edge_op {
                    &mut EdgeOperation::Merge{..} => {
                        EdgeOperation::Merge{n: rng.gen::<u32>()} 
                    }
                    &mut EdgeOperation::Next{..} => {
                        EdgeOperation::Next{n: rng.gen::<u32>()} 
                    }
                    &mut EdgeOperation::Parent{..} => {
                        EdgeOperation::Parent{n: rng.gen::<u32>()} 
                    }
                    ref op => {
                        (*op).clone()
                    }
                };

                *edge_op = new_edge_op;
            }
        }
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
