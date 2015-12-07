// Edge Operation Genome

use evo::prob::{Probability, ProbabilityValue};
use evo::crossover::linear_2point_crossover_random;
use evo::nsga2::Mate;
use rand::Rng;
use rand::distributions::{IndependentSample, Weighted};
use graph_edge_evolution::{EdgeOperation, GraphBuilder, NthEdgeF};
use owned_weighted_choice::OwnedWeightedChoice;
use std::str::FromStr;
use triadic_census::OptDenseDigraph;
use std::collections::BTreeMap;

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


/// Element-wise Mutation operation.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum MutOp {
    /// No mutation (copy element)
    Copy,
    /// Insert new operation
    Insert,
    /// Remove an operation
    Remove,
    /// Modify the operation
    ModifyOp,
    /// Modify a parameter value
    ModifyParam,
    /// Modify both operation and parameter
    Replace,
}

impl FromStr for MutOp {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "Copy" => Ok(MutOp::Copy),
            "Insert" => Ok(MutOp::Insert),
            "Remove" => Ok(MutOp::Remove),
            "ModifyOp" => Ok(MutOp::ModifyOp),
            "ModifyParam" => Ok(MutOp::ModifyParam),
            "Replace" => Ok(MutOp::Replace),
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
    weighted_mut_op: OwnedWeightedChoice<MutOp>,
    prob_mutate_elem: Probability,
}

impl Mate<EdgeOpsGenome> for Toolbox {
    fn mate<R: Rng>(&mut self,
                    rng: &mut R,
                    p1: &EdgeOpsGenome,
                    p2: &EdgeOpsGenome)
                    -> EdgeOpsGenome {

        // XXX: either mutate or crossover, or both?

        let child = EdgeOpsGenome {
            edge_ops: linear_2point_crossover_random(rng, &p1.edge_ops[..], &p2.edge_ops[..]),
        };
        let mutated_child = self.mutate(rng, child);
        mutated_child
    }
}

pub fn parse_weighted_op_list<T>(s: &str) -> Result<Vec<(T, u32)>, String> where T:FromStr<Err=String> {
    let mut v = Vec::new();
    for opstr in s.split(",") {
        if opstr.is_empty() {
            continue;
        }
        let mut i = opstr.splitn(2, ":");
        if let Some(ops) = i.next() {
            match T::from_str(ops) {
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

impl Toolbox {
    pub fn mutate<R: Rng>(&self, rng: &mut R, ind: EdgeOpsGenome) -> EdgeOpsGenome {
        let mut mut_ind = Vec::with_capacity(ind.len() + 1);

        for edge_op in ind.edge_ops.iter() {
            let new_op = if rng.gen::<ProbabilityValue>().is_probable_with(self.prob_mutate_elem) {
                match self.weighted_mut_op.ind_sample(rng) {
                    MutOp::Copy => {
                        edge_op.clone()
                    }
                    MutOp::Insert => {
                        // Insert a new op before the current operation
                        mut_ind.push(self.generate_random_edge_operation(rng));
                        edge_op.clone()
                    }
                    MutOp::Remove => {
                        // remove current operation
                        continue;
                    }
                    MutOp::ModifyOp => {
                        (self.weighted_op_choice.ind_sample(rng), edge_op.1)
                    }
                    MutOp::ModifyParam => {
                        (edge_op.0, rng.gen::<f32>())
                    }
                    MutOp::Replace => {
                        self.generate_random_edge_operation(rng)
                    }
                }
            } else {
                edge_op.clone()
            };
            mut_ind.push(new_op);
        }

        EdgeOpsGenome { edge_ops: mut_ind }
    }

    pub fn new(weighted_op_choices: &[(Op, u32)],
               prob_mutate_elem: Probability,
               weighed_mut_op: &[(MutOp, u32)])
               -> Toolbox {
        let mut w_ops = Vec::new();
        for &(op, weight) in weighted_op_choices {
            if weight > 0 {
                // an operation with weight=0 cannot be selected
                w_ops.push(Weighted {
                    weight: weight,
                    item: op,
                });
            }
        }

        let mut w_mut_ops = Vec::new();
        for &(op, weight) in weighed_mut_op {
            if weight > 0 {
                // an operation with weight=0 cannot be selected
                w_mut_ops.push(Weighted {
                    weight: weight,
                    item: op,
                });
            }
        }

        assert!(w_ops.len() > 0);
        assert!(w_mut_ops.len() > 0);

        Toolbox {
            prob_mutate_elem: prob_mutate_elem,
            weighted_op_choice: OwnedWeightedChoice::new(w_ops),
            weighted_mut_op: OwnedWeightedChoice::new(w_mut_ops),
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

    pub fn to_graph(&self) -> OptDenseDigraph<(), ()> {
        let mut builder: GraphBuilder<f32, ()> = GraphBuilder::new();
        for &(op, f) in &self.edge_ops[..] {
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
                    EdgeOperation::Merge { n: NthEdgeF(f) }
                }
                Op::Next => {
                    EdgeOperation::Next { n: NthEdgeF(f) }
                }
                Op::Parent => {
                    EdgeOperation::Parent { n: NthEdgeF(f) }
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

        let mut g: OptDenseDigraph<(), ()> = OptDenseDigraph::new(builder.total_number_of_nodes()); // XXX: rename to real_number

        // maps node_idx to index used within the graph.
        let mut node_map: BTreeMap<usize, usize> = BTreeMap::new(); // XXX: with_capacity

        builder.visit_nodes(|node_idx, _| {
            let graph_idx = g.add_node();
            node_map.insert(node_idx, graph_idx);
        });

        builder.visit_edges(|(a, b), _| {
            g.add_edge(node_map[&a], node_map[&b]);
        });

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
