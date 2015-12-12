// Edge Operation L-System Genome

mod edgeop;
mod expr_op;
mod cond_op;

use evo::prob::{Probability, ProbabilityValue};
use evo::crossover::linear_2point_crossover_random;
use evo::nsga2::Mate;
use rand::Rng;
use rand::distributions::{IndependentSample, Weighted};
use rand::distributions::range::Range;
use graph_annealing::owned_weighted_choice::OwnedWeightedChoice;
use std::str::FromStr;
use triadic_census::OptDenseDigraph;
use lindenmayer_system::{Alphabet, Symbol, SymbolString, System};
use lindenmayer_system::symbol::Sym2;
use lindenmayer_system::expr::Expr;
use self::edgeop::{EdgeOp, edgeops_to_graph};
use self::expr_op::{ExprOp, random_expr};
use self::cond_op::{CondOp, random_cond};

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

// The alphabet of terminal and non-terminals we use.
// Non-terminals are our rules, while terminals are the EdgeOps.  
#[derive(Debug, PartialEq, Eq, Clone)]
pub enum EdgeAlphabet {
    Terminal(EdgeOp),
    NonTerminal(u32),
}

impl Alphabet for EdgeAlphabet {}

// We use 2-ary symbols, i.e. symbols with two parameters.
type Sym = Sym2<EdgeAlphabet, f32>;

pub struct SymbolGenerator {
    pub max_expr_depth: usize,
    pub expr_weighted_op: OwnedWeightedChoice<ExprOp>,
    pub expr_weighted_op_max_depth: OwnedWeightedChoice<ExprOp>,

    // terminal symbols
    pub edge_weighted_op: OwnedWeightedChoice<EdgeOp>,

    pub nonterminal_symbols: Range<u32>,

    /// The probability with which a terminal value is choosen.
    pub prob_terminal: Probability,
}

impl SymbolGenerator {
    /// Generates a random production (right hand-side of a rule).
    pub fn gen_symbolstring<R, S>(&self,
                                  rng: &mut R,
                                  len: usize,
                                  arity: usize,
                                  num_params: usize)
                                  -> SymbolString<S>
        where R: Rng,
              S: Symbol<A = EdgeAlphabet, T = f32>
    {
        SymbolString((0..len)
                         .into_iter()
                         .map(|_| self.gen_symbol(rng, arity, num_params))
                         .collect())
    }

    pub fn gen_symbol<R, S>(&self, rng: &mut R, arity: usize, num_params: usize) -> S
        where R: Rng,
              S: Symbol<A = EdgeAlphabet, T = f32>
    {
        S::from_iter(self.gen_symbol_value(rng),
                     (0..arity).into_iter().map(|_| self.gen_expr(rng, num_params)))

    }

    fn gen_expr<R: Rng>(&self, rng: &mut R, num_params: usize) -> Expr<f32> {
        random_expr(rng,
                    num_params,
                    self.max_expr_depth,
                    &self.expr_weighted_op,
                    &self.expr_weighted_op_max_depth)
    }

    fn gen_terminal<R: Rng>(&self, rng: &mut R) -> EdgeAlphabet {
        EdgeAlphabet::Terminal(self.edge_weighted_op.ind_sample(rng))
    }

    pub fn gen_nonterminal<R: Rng>(&self, rng: &mut R) -> EdgeAlphabet {
        EdgeAlphabet::NonTerminal(self.nonterminal_symbols.ind_sample(rng))
    }

    fn gen_symbol_value<R: Rng>(&self, rng: &mut R) -> EdgeAlphabet {
        if rng.gen::<ProbabilityValue>().is_probable_with(self.prob_terminal) {
            self.gen_terminal(rng)
        } else {
            self.gen_nonterminal(rng)
        }
    }
}



#[derive(Clone, Debug)]
pub struct Genome {
    edge_ops: Vec<(EdgeOp, f32)>,
    axiom: SymbolString<Sym>,
    system: System<Sym>,
}

pub struct Toolbox {
    weighted_op: OwnedWeightedChoice<EdgeOp>,
    weighted_var_op: OwnedWeightedChoice<VarOp>,
    weighted_mut_op: OwnedWeightedChoice<MutOp>,
    prob_mutate_elem: Probability,
}

impl Mate<Genome> for Toolbox {
    // p1 is potentially "better" than p2
    fn mate<R: Rng>(&mut self, rng: &mut R, p1: &Genome, p2: &Genome) -> Genome {

        match self.weighted_var_op.ind_sample(rng) {
            VarOp::Copy => p1.clone(),
            VarOp::Mutate => self.mutate(rng, p1),
            VarOp::LinearCrossover2 => {
                Genome {
                    edge_ops: linear_2point_crossover_random(rng,
                                                             &p1.edge_ops[..],
                                                             &p2.edge_ops[..]),
                    axiom: SymbolString(vec![]),
                    system: System::new(),
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
        edgeops_to_graph(&self.edge_ops)
    }
}


// When we need a value within (0, 1], we simply cut off the integer part.
// Max depth.

// There are many parameters that influence the creation of a random
// genome:
//
//     - Number of rules
//     - Arity of symbols
//     - Axiom and Length
//     - Length of a production rule
//     - Number of (condition, successor) pairs per rule.
//     - Complexity of expression in Symbol
//
//     - Number of Iterations.
//
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
        axiom: SymbolString(vec![]),
        system: System::new(),
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
                    MutOp::Copy => edge_op.clone(),
                    MutOp::Insert => {
                        // Insert a new op before the current operation
                        mut_ind.push(self.generate_random_edge_operation(rng));
                        edge_op.clone()
                    }
                    MutOp::Remove => {
                        // remove current operation
                        continue;
                    }
                    MutOp::ModifyOp => (self.weighted_op.ind_sample(rng), edge_op.1),
                    MutOp::ModifyParam => (edge_op.0, rng.gen::<f32>()),
                    MutOp::Replace => self.generate_random_edge_operation(rng),
                }
            } else {
                edge_op.clone()
            };
            mut_ind.push(new_op);
        }

        Genome {
            edge_ops: mut_ind,
            axiom: SymbolString(vec![]),
            system: System::new(),
        }
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

    pub fn random_genome<R: Rng>(&self, rng: &mut R, len: usize) -> Genome {
        random_genome(rng, len, &self.weighted_op)
    }
}
