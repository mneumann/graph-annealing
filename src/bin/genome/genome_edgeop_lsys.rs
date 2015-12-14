// Edge Operation L-System Genome

pub mod edgeop;
mod expr_op;
mod cond_op;

use evo::prob::{Probability, ProbabilityValue};
use evo::crossover::linear_2point_crossover_random;
use evo::nsga2::{self, FitnessEval, Mate, MultiObjective3};
use rand::Rng;
use rand::distributions::{IndependentSample, Weighted};
use rand::distributions::range::Range;
use graph_annealing::owned_weighted_choice::OwnedWeightedChoice;
use graph_annealing::fitness_function::FitnessFunction;
use graph_annealing::goal::Goal;
use std::str::FromStr;
use triadic_census::OptDenseDigraph;
use lindenmayer_system::{Alphabet, Condition, Symbol, SymbolString, System};
use lindenmayer_system::symbol::Sym2;
use lindenmayer_system::expr::Expr;
use self::edgeop::{EdgeOp, edgeops_to_graph};
use self::expr_op::{ConstExprOp, ExprOp, random_const_expr, random_expr};
use self::cond_op::{CondOp, random_cond};
use simple_parallel::Pool;
use crossbeam;

/// Element-wise Mutation operation.
defops!{MutOp;
    // No mutation (copy element)
    Copy,
    // Insert new operation
    Insert,
    // Remove an operation
    Remove,
    // Modify the operation
    ModifyOp,
    // Modify a parameter value
    ModifyParam,
    // Modify both operation and parameter
    Replace
}

/// Variation operators
defops!{VarOp;
    // No variation. Reproduce exactly
    Copy,

    // Mutate
    Mutate,

    // 2-point Linear crossover
    LinearCrossover2,

    // Uniform crossover
    UniformCrossover
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

    pub const_expr_weighted_op: OwnedWeightedChoice<ConstExprOp>,
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

    /// Generate a simple condition like:
    ///     Arg(n) or 0.0 [>=] or [<=] constant expr
    fn gen_simple_rule_condition<R: Rng>(&self, rng: &mut R, arity: usize) -> Condition<f32> {
        let lhs = if arity > 0 {
            Expr::Arg(rng.gen_range(0, arity))
        } else {
            Expr::Const(0.0)
        };

        let rhs = random_const_expr(rng, &self.const_expr_weighted_op);

        if rng.gen::<bool>() {
            Condition::GreaterEqual(Box::new(lhs), Box::new(rhs))
        } else {
            Condition::LessEqual(Box::new(lhs), Box::new(rhs))
        }
    }

    // Generate a random rule with `symbol` and `arity` parameters.
    //
    // fn gen_rule<R: Rng>(&self, symbol: EdgeAlphabet, arity: usize) {
    // }
}


// When we need a value within (0, 1], we simply cut off the integer part.
// Max depth.


// # Genome interpretation:
//
//     * We use rule with number 0 as axiom.
//       The arguments for the axiom are fixed and passed in from the command line.
//       For simplicity, these arguments are duplicated in each Genome in `axiom_params`.
//       One could also apply genetic operations onto these.
//
//
#[derive(Clone, Debug)]
pub struct Genome {
    system: System<Sym>,
}

pub struct Toolbox<N, E> {
    goal: Goal<N, E>,
    pool: Pool,
    fitness_functions: (FitnessFunction, FitnessFunction, FitnessFunction),

    weighted_op: OwnedWeightedChoice<EdgeOp>,
    weighted_var_op: OwnedWeightedChoice<VarOp>,
    weighted_mut_op: OwnedWeightedChoice<MutOp>,
    prob_mutate_elem: Probability,

    /// Arguments to the axiom rule.
    pub axiom_args: Vec<Expr<f32>>,

    /// Maximum number of iterations of the L-system  (XXX: Limit also based on generated length)
    pub iterations: usize,
}

impl<N: Clone + Default, E: Clone + Default> Toolbox<N, E> {
    pub fn new(goal: Goal<N, E>,
               pool: Pool,
               fitness_functions: (FitnessFunction, FitnessFunction, FitnessFunction),
               weighted_op: Vec<Weighted<EdgeOp>>,
               weighted_var_op: Vec<Weighted<VarOp>>,
               weighted_mut_op: Vec<Weighted<MutOp>>,
               prob_mutate_elem: Probability)
               -> Toolbox<N, E> {

        Toolbox {
            goal: goal,
            pool: pool,
            fitness_functions: fitness_functions,
            prob_mutate_elem: prob_mutate_elem,
            weighted_op: OwnedWeightedChoice::new(weighted_op),
            weighted_var_op: OwnedWeightedChoice::new(weighted_var_op),
            weighted_mut_op: OwnedWeightedChoice::new(weighted_mut_op),
            axiom_args: vec![],
            iterations: 10,
        }
    }

    pub fn mutate<R: Rng>(&self, rng: &mut R, ind: &Genome) -> Genome {
        // let mut mut_ind = Vec::with_capacity(ind.len() + 1);
        //
        // for edge_op in ind.edge_ops.iter() {
        // let new_op = if rng.gen::<ProbabilityValue>().is_probable_with(self.prob_mutate_elem) {
        // match self.weighted_mut_op.ind_sample(rng) {
        // MutOp::Copy => edge_op.clone(),
        // MutOp::Insert => {
        // Insert a new op before the current operation
        // mut_ind.push(self.generate_random_edge_operation(rng));
        // edge_op.clone()
        // }
        // MutOp::Remove => {
        // remove current operation
        // continue;
        // }
        // MutOp::ModifyOp => (self.weighted_op.ind_sample(rng), edge_op.1),
        // MutOp::ModifyParam => (edge_op.0, rng.gen::<f32>()),
        // MutOp::Replace => self.generate_random_edge_operation(rng),
        // }
        // } else {
        // edge_op.clone()
        // };
        // mut_ind.push(new_op);
        // }
        //

        Genome {
            system: System::new(),
        }
    }

    fn generate_random_edge_operation<R: Rng>(&self, rng: &mut R) -> (EdgeOp, f32) {
        generate_random_edge_operation(&self.weighted_op, rng)
    }


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
    //     There are many
    //
    pub fn random_genome<R: Rng>(&self, rng: &mut R) -> Genome {
        Genome {
            system: System::new(),
        }

    }
}

impl<N:Clone+Sync+Default,E:Clone+Sync+Default> FitnessEval<Genome, MultiObjective3<f32>> for Toolbox<N,E> {
    /// Evaluates the fitness of a Genome population.
    fn fitness(&mut self, pop: &[Genome]) -> Vec<MultiObjective3<f32>> {
        let pool = &mut self.pool;
        let goal = &self.goal;
        let axiom_args = &self.axiom_args[..];
        let iterations = self.iterations;

        let fitness_functions = self.fitness_functions;

        crossbeam::scope(|scope| {
            pool.map(scope, pop, |ind| {
                let edge_ops = ind.to_edge_ops(axiom_args, iterations);
                let g = edgeops_to_graph(&edge_ops);

                //let g = ind.to_graph();
                MultiObjective3::from((goal.apply_fitness_function(fitness_functions.0, &g),
                                       goal.apply_fitness_function(fitness_functions.1, &g),
                                       goal.apply_fitness_function(fitness_functions.2, &g)))

            })
                .collect()
        })
    }
}


impl<N: Clone + Default, E: Clone + Default> Mate<Genome> for Toolbox<N, E> {
    // p1 is potentially "better" than p2
    fn mate<R: Rng>(&mut self, rng: &mut R, p1: &Genome, p2: &Genome) -> Genome {

        match self.weighted_var_op.ind_sample(rng) {
            VarOp::Copy => p1.clone(),
            VarOp::Mutate => self.mutate(rng, p1),
            VarOp::LinearCrossover2 => {
                Genome {
                    // edge_ops: linear_2point_crossover_random(rng,
                    // &p1.edge_ops[..],
                    // &p2.edge_ops[..]),
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
    /// Develops
    pub fn to_edge_ops(&self, _axiom_args: &[Expr<f32>], _iterations: usize) -> Vec<(EdgeOp, f32)> {
        vec![]
    }
}


#[inline]
fn generate_random_edge_operation<R: Rng>(weighted_op: &OwnedWeightedChoice<EdgeOp>,
                                          rng: &mut R)
                                          -> (EdgeOp, f32) {
    (weighted_op.ind_sample(rng), rng.gen::<f32>())
}
