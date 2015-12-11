// Edge Operation L-System Genome

mod edgeop;

use evo::prob::{Probability, ProbabilityValue};
use evo::crossover::linear_2point_crossover_random;
use evo::nsga2::Mate;
use rand::Rng;
use rand::distributions::{IndependentSample, Weighted};
use graph_edge_evolution::{EdgeOperation, GraphBuilder, NthEdgeF};
use graph_annealing::owned_weighted_choice::OwnedWeightedChoice;
use std::str::FromStr;
use triadic_census::OptDenseDigraph;
use std::collections::BTreeMap;
use lindenmayer_system::{System, Symbol};
use lindenmayer_system::symbol::Sym2;
use self::edgeop::EdgeOp;

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

// We use 2-ary symbols, i.e. symbols with two parameters.
type Sym = Sym2<u32, f32>;

#[derive(Clone, Debug)]
pub struct Genome {
    edge_ops: Vec<(EdgeOp, f32)>,
    axiom: Sym,
}

pub struct Toolbox {
    weighted_op: OwnedWeightedChoice<EdgeOp>,
    weighted_var_op: OwnedWeightedChoice<VarOp>,
    weighted_mut_op: OwnedWeightedChoice<MutOp>,
    prob_mutate_elem: Probability,
}

impl Mate<Genome> for Toolbox {
    // p1 is potentially "better" than p2
    fn mate<R: Rng>(&mut self,
                    rng: &mut R,
                    p1: &Genome,
                    p2: &Genome)
                    -> Genome {

        match self.weighted_var_op.ind_sample(rng) {
            VarOp::Copy => {
                p1.clone()
            }
            VarOp::Mutate => {
                self.mutate(rng, p1)
            }
            VarOp::LinearCrossover2 => {
                Genome {
                    edge_ops: linear_2point_crossover_random(rng,
                                                             &p1.edge_ops[..],
                                                             &p2.edge_ops[..]),
                    axiom: Symbol::new(0), // XXX
                }
            }
            VarOp::UniformCrossover => {
                panic!("TODO");
            }
        }
    }
}

impl Genome {
    pub fn len(&self) -> usize {
        self.edge_ops.len()
    }

    pub fn to_graph(&self) -> OptDenseDigraph<(), ()> {
        let mut builder: GraphBuilder<f32, ()> = GraphBuilder::new();
        for &(op, f) in &self.edge_ops[..] {
            let graph_op = match op {
                EdgeOp::Dup => {
                    EdgeOperation::Duplicate { weight: f }
                }
                EdgeOp::Split => {
                    EdgeOperation::Split { weight: f }
                }
                EdgeOp::Loop => {
                    EdgeOperation::Loop { weight: f }
                }
                EdgeOp::Merge => {
                    EdgeOperation::Merge { n: NthEdgeF(f) }
                }
                EdgeOp::Next => {
                    EdgeOperation::Next { n: NthEdgeF(f) }
                }
                EdgeOp::Parent => {
                    EdgeOperation::Parent { n: NthEdgeF(f) }
                }
                EdgeOp::Reverse => {
                    EdgeOperation::Reverse
                }
                EdgeOp::Save => {
                    EdgeOperation::Save
                }
                EdgeOp::Restore => {
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

#[inline]
pub fn random_genome<R>(rng: &mut R,
                        len: usize,
                        weighted_op: &OwnedWeightedChoice<EdgeOp>)
                        -> Genome
    where R: Rng
{
    Genome {
        edge_ops: (0..len)
                      .map(|_| generate_random_edge_operation(weighted_op, rng))
                      .collect(),
        axiom: Symbol::new(0),
    }
}

#[inline]
pub fn generate_random_edge_operation<R: Rng>(weighted_op: &OwnedWeightedChoice<EdgeOp>,
                                              rng: &mut R)
                                              -> (EdgeOp, f32) {
    (weighted_op.ind_sample(rng), rng.gen::<f32>())
}

impl Toolbox {
    pub fn mutate<R: Rng>(&self, rng: &mut R, ind: &Genome) -> Genome {
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

        Genome { edge_ops: mut_ind, axiom: Symbol::new(0) }
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

    pub fn random_genome<R:Rng>(&self, rng: &mut R, len: usize) -> Genome {
        random_genome(rng, len, &self.weighted_op)
    }
}
