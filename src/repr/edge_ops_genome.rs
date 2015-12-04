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

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Op {
    Dup,
    Split,
    Loop,
    Merge,
    Next,
    Parent,
    Reverse,
    Save,
    Restore,
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
            "Save" => Ok(Op::Save),
            "Restore" => Ok(Op::Restore),
            _ => Err(format!("Invalid opcode: {}", s)),
        }
    }
}

#[derive(Clone, Debug)]
pub struct EdgeOpsGenome {
    edge_ops: Vec<(Op, f32)>,
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
                edge_op.1 = rng.gen::<f32>();
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
                            return Err(format!("Invalid weight: {}", ws));
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

    fn generate_random_edge_operation<R: Rng>(&self, rng: &mut R) -> (Op, f32) /*EdgeOperation<f32, ()>*/ {
        (self.weighted_op_choice.ind_sample(rng), rng.gen::<f32>())
    }

    pub fn random_genome<R: Rng>(&self, rng: &mut R, len: usize) -> EdgeOpsGenome {
        EdgeOpsGenome {
            edge_ops: (0..len)
                          .map(|_| self.generate_random_edge_operation(rng))
                          .collect(),
        }
    }
}


impl EdgeOpsGenome {
    pub fn len(&self) -> usize {
        self.edge_ops.len()
    }

    pub fn to_graph(&self, max_node: u32) -> Graph<(), (), Directed, u32> {
        let mut builder: GraphBuilder<f32, ()> = GraphBuilder::new();
        for &(op, f) in &self.edge_ops[..] {
            let n = (max_node as f32 * f) as u32;
            let graph_op = match op {
                Op::Dup => {
                    EdgeOperation::Duplicate { weight: f }
                }
                Op::Split => {
                    EdgeOperation::Split { weight: f }
                }
                Op::Loop => {
                    EdgeOperation::Loop { weight: f }
                }
                Op::Merge => {
                    EdgeOperation::Merge { n: n }
                }
                Op::Next => {
                    EdgeOperation::Next { n: n }
                }
                Op::Parent => {
                    EdgeOperation::Parent { n: n }
                }
                Op::Reverse => {
                    EdgeOperation::Reverse
                }
                Op::Save => {
                    EdgeOperation::Save
                }
                Op::Restore => {
                    EdgeOperation::Restore
                }
            };
            builder.apply_operation(graph_op);
        }
        let edge_list = builder.to_edge_list();

        let mut g: Graph<(), (), Directed, u32> = Graph::new();
        let n = edge_list.len();
        for _ in 0..n {
            let _ = g.add_node(());
        }

        for (i, (_node_function, node_edges)) in edge_list.into_iter().enumerate() {
            for &(j, _weight) in node_edges.iter() {
                g.add_edge(NodeIndex::new(i), NodeIndex::new(j), ());
            }
        }

        return g;
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
    assert_eq!(Err("Invalid weight: ".to_string()),
               Toolbox::parse_weighted_op_choice_list("Dup:2,Split:"));
    assert_eq!(Err("Invalid weight: a".to_string()),
               Toolbox::parse_weighted_op_choice_list("Dup:2,Split:a"));
    assert_eq!(Err("Invalid opcode: dup".to_string()),
               Toolbox::parse_weighted_op_choice_list("dup:2,Split:a"));
}
