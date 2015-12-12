// Edge Operation Genome

mod edgeop;

use evo::prob::{Probability, ProbabilityValue};
use evo::crossover::linear_2point_crossover_random;
use evo::nsga2::Mate;
use rand::Rng;
use rand::distributions::{IndependentSample, Weighted};
use graph_annealing::owned_weighted_choice::OwnedWeightedChoice;
use std::str::FromStr;
use std::string::ToString;
use std::collections::BTreeMap;
use triadic_census::OptDenseDigraph;
use serde_json::Value;
use sexp::{Atom, Sexp};
use self::edgeop::{edgeops_to_graph, EdgeOp};

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


/// Variation operators
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum VarOp {
    /// No variation. Reproduce exactly
    Copy,

    /// Mutate
    Mutate,

    /// 2-point Linear crossover
    LinearCrossover2,

    /// Uniform crossover
    UniformCrossover,
}

impl FromStr for VarOp {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "Copy" => Ok(VarOp::Copy),
            "Mutate" => Ok(VarOp::Mutate),
            "LinearCrossover2" => Ok(VarOp::LinearCrossover2),
            "UniformCrossover" => Ok(VarOp::UniformCrossover),
            _ => Err(format!("Invalid opcode: {}", s)),
        }
    }
}

#[derive(Clone, Debug)]
pub struct EdgeOpsGenome {
    edge_ops: Vec<(EdgeOp, f32)>,
}

pub struct Toolbox {
    weighted_op: OwnedWeightedChoice<EdgeOp>,
    weighted_var_op: OwnedWeightedChoice<VarOp>,
    weighted_mut_op: OwnedWeightedChoice<MutOp>,
    prob_mutate_elem: Probability,
}

impl Mate<EdgeOpsGenome> for Toolbox {
    // p1 is potentially "better" than p2
    fn mate<R: Rng>(&mut self,
                    rng: &mut R,
                    p1: &EdgeOpsGenome,
                    p2: &EdgeOpsGenome)
                    -> EdgeOpsGenome {

        match self.weighted_var_op.ind_sample(rng) {
            VarOp::Copy => {
                p1.clone()
            }
            VarOp::Mutate => {
                self.mutate(rng, p1)
            }
            VarOp::LinearCrossover2 => {
                EdgeOpsGenome {
                    edge_ops: linear_2point_crossover_random(rng,
                                                             &p1.edge_ops[..],
                                                             &p2.edge_ops[..]),
                }
            }
            VarOp::UniformCrossover => {
                panic!("TODO");
            }
        }
    }
}

impl EdgeOpsGenome {
    pub fn len(&self) -> usize {
        self.edge_ops.len()
    }

    pub fn to_json(&self) -> Value {
        let edge_ops: Vec<Value> = self.edge_ops
                                       .iter()
                                       .map(|&(ref op, param)| {
                                           let mut gene = BTreeMap::new();
                                           gene.insert("op".to_string(),
                                                       Value::String(op.to_string()));
                                           gene.insert("param".to_string(),
                                                       Value::F64(param as f64));
                                           Value::Object(gene)
                                           // Value::Array(vec![Value::String(op.to_string()),
                                           // Value::F64(param as f64)])
                                       })
                                       .collect();
        let mut map = BTreeMap::new();
        map.insert("edge_ops".to_string(), Value::Array(edge_ops));
        Value::Object(map)
    }

    pub fn to_sexp(&self) -> Sexp {
        let edge_ops: Vec<Sexp> = self.edge_ops
                                      .iter()
                                      .map(|&(ref op, param)| {
                                          Sexp::List(vec![Sexp::Atom(Atom::S(op.to_string())),
                                                          Sexp::Atom(Atom::F(param as f64))])
                                      })
                                      .collect();

        Sexp::List(vec![Sexp::Atom(Atom::S("EdgeOpsGenome".to_string())), Sexp::List(edge_ops)])
    }

    pub fn from_json(val: &Value) -> EdgeOpsGenome {
        let edge_ops = val.find("edge_ops").unwrap();
        let edge_ops = edge_ops.as_array().unwrap();
        let edge_ops = edge_ops.iter()
                               .map(|gene| {
                                   let op = EdgeOp::from_str(gene.find("op")
                                                             .unwrap()
                                                             .as_string()
                                                             .unwrap())
                                                .unwrap();
                                   let param = gene.find("param").unwrap().as_f64().unwrap();
                                   (op, param as f32)
                               })
                               .collect();
        EdgeOpsGenome { edge_ops: edge_ops }
    }

    pub fn to_graph(&self) -> OptDenseDigraph<(), ()> {
        edgeops_to_graph(&self.edge_ops)
    }
}

#[inline]
pub fn random_genome<R>(rng: &mut R,
                        len: usize,
                        weighted_op: &OwnedWeightedChoice<EdgeOp>)
                        -> EdgeOpsGenome
    where R: Rng
{
    EdgeOpsGenome {
        edge_ops: (0..len)
                      .map(|_| generate_random_edge_operation(weighted_op, rng))
                      .collect(),
    }
}

#[inline]
pub fn generate_random_edge_operation<R: Rng>(weighted_op: &OwnedWeightedChoice<EdgeOp>,
                                              rng: &mut R)
                                              -> (EdgeOp, f32) {
    (weighted_op.ind_sample(rng), rng.gen::<f32>())
}

impl Toolbox {
    pub fn mutate<R: Rng>(&self, rng: &mut R, ind: &EdgeOpsGenome) -> EdgeOpsGenome {
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
                        (self.weighted_op.ind_sample(rng), edge_op.1)
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

    pub fn new(weighted_op: Vec<Weighted<EdgeOp>>,
               weighted_var_op: Vec<Weighted<VarOp>>,
               weighted_mut_op: Vec<Weighted<MutOp>>,
               prob_mutate_elem: Probability)
               -> Toolbox {

        Toolbox {
            prob_mutate_elem: prob_mutate_elem,
            weighted_op: OwnedWeightedChoice::new(weighted_op),
            weighted_var_op: OwnedWeightedChoice::new(weighted_var_op),
            weighted_mut_op: OwnedWeightedChoice::new(weighted_mut_op),
        }
    }

    fn generate_random_edge_operation<R: Rng>(&self, rng: &mut R) -> (EdgeOp, f32) {
        generate_random_edge_operation(&self.weighted_op, rng)
    }

    pub fn random_genome<R:Rng>(&self, rng: &mut R, len: usize) -> EdgeOpsGenome {
        random_genome(rng, len, &self.weighted_op)
    }
}
