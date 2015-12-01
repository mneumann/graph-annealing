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
use std::str::FromStr;

#[derive(Clone, Debug)]
pub struct EdgeOpsGenome {
    edge_ops: Vec<EdgeOperation<f32, ()>>,
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Op {
    Dup,
    Split,
    Loop,
    Merge,
    Next,
    Parent,
    Reverse, // XXX Save: Restore
}

impl FromStr for Op {
   type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "Dup" => Ok(Op::Dup),
            "Split" => Ok(Op::Split),
            "Loop" => Ok(Op::Loop),
            "Merge" => Ok(Op::Merge),
            "Next" => Ok(Op::Next),
            "Parent" => Ok(Op::Parent),
            "Reverse" => Ok(Op::Reverse),
            _ => Err(format!("Invalid opcode: {}", s)),
        }
    }
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

#[test]
fn test_parse_weighted_op_choice_list() {
    assert_eq!(Ok(vec![]), Toolbox::parse_weighted_op_choice_list(""));
    assert_eq!(Ok(vec![(Op::Dup, 1)]),
               Toolbox::parse_weighted_op_choice_list("Dup"));
    assert_eq!(Ok(vec![(Op::Dup, 1)]),
               Toolbox::parse_weighted_op_choice_list("Dup:1"));
    assert_eq!(Ok(vec![(Op::Dup, 2)]),
               Toolbox::parse_weighted_op_choice_list("Dup:2"));
    assert_eq!(Ok(vec![(Op::Dup, 2), (Op::Split, 1)]),
               Toolbox::parse_weighted_op_choice_list("Dup:2,Split"));
    assert_eq!(Err("invalid weight: ".to_string()),
               Toolbox::parse_weighted_op_choice_list("Dup:2,Split:"));
    assert_eq!(Err("invalid weight: a".to_string()),
               Toolbox::parse_weighted_op_choice_list("Dup:2,Split:a"));
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

    pub fn parse_weighted_op_choice_list(s: &str) -> Result<Vec<(Op, u32)>, String> {
        let mut v = Vec::new();
        for opstr in s.split(",") {
            if opstr.is_empty() {
                continue;
            }
            let mut i = opstr.splitn(2, ":");
            if let Some(ops) = i.next() {
                match Op::from_str(ops) {
                    Ok(op) => {
                        let ws = i.next().unwrap_or("1");
                        if let Ok(weight) = u32::from_str(ws) {
                            v.push((op, weight));
                        } else {
                            return Err(format!("invalid weight: {}", ws));
                        }
                    }
                    Err(s) => {
                        return Err(s);
                    }
                }
            } else {
                return Err("missing op".to_string());
            }
        }
        return Ok(v);
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
