// Edge Operation L-System Genome


// NOTES:
//
// * When we need a value within (0, 1], we simply cut off the integral part und use fractional part
//   only.
//
// * We start out with a simple Genome, which gets more and more complex through mating.

pub mod edgeop;
pub mod expr_op;
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
use lindenmayer_system::{Alphabet, Condition, LSystem, Rule, Symbol, SymbolString,
                         apply_first_rule};
use lindenmayer_system::symbol::Sym2;
use lindenmayer_system::expr::Expr;
use self::edgeop::{EdgeOp, edgeops_to_graph};
use self::expr_op::{ConstExprOp, ExprOp, FlatExprOp, random_const_expr, random_expr};
use self::cond_op::{CondOp, random_cond};
use simple_parallel::Pool;
use crossbeam;
use std::collections::BTreeMap;

/// Element-wise Mutation operation. XXX
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

/// Rule mutation operations.
defops!{RuleMutOp;
    // No mutation
    Copy,

    // Modify Condition
    ModifyCondition,

    // Modify Production
    ModifyProduction
}

/// Rule production mutation operations.
defops!{RuleProductionMutOp;
    // No mutation
    Copy,

    // Replace a symbol (keeps the parameters)
    ReplaceSymbol,

    // Make a modification to a symbol parameter
    ModifyParameter,

    // Insert a sequence of symbols
    InsertSequence, 

    // Delete a sequence of symbols
    DeleteSequence
}


/// Variation operators
defops!{VarOp;
    // No variation. Reproduce exactly
    Copy,

    // Mutate
    Mutate,

    // Crossover
    Crossover
}

type RuleId = u32;

// The alphabet of terminal and non-terminals we use.
// Non-terminals are our rules, while terminals are the EdgeOps.
#[derive(Debug, PartialEq, Eq, Clone)]
enum EdgeAlphabet {
    Terminal(EdgeOp),
    NonTerminal(RuleId),
}

impl Alphabet for EdgeAlphabet {}

// We use 2-ary symbols, i.e. symbols with two parameters.
type Sym = Sym2<EdgeAlphabet, f32>;

/// Rules can only be stored for NonTerminals
#[derive(Clone, Debug)]
struct System {
    rules: BTreeMap<RuleId, Vec<Rule<Sym>>>,
}

impl System {
    fn new() -> System {
        System { rules: BTreeMap::new() }
    }

    fn add_rule(&mut self,
                rule_id: RuleId,
                production: SymbolString<Sym>,
                condition: Condition<f32>) {
        self.rules
            .entry(rule_id)
            .or_insert(vec![])
            .push(Rule::new_with_condition(EdgeAlphabet::NonTerminal(rule_id),
                                           production,
                                           condition));
    }

    fn random_rule_id<R:Rng>(&self, rng: &mut R) -> RuleId {
        let len = self.rules.len();
        assert!(len > 0);
        let nth = rng.gen_range(0, len);
        self.rules.iter().map(|(&k, _)| k).nth(nth).unwrap()
    }
}

impl LSystem<Sym> for System {
    fn apply_first_rule(&self, sym: &Sym) -> Option<SymbolString<Sym>> {
        match sym.symbol() {
            // We don't store rules for terminal symbols.
            &EdgeAlphabet::Terminal(_) => None,

            // Only apply rules for non-terminals
            &EdgeAlphabet::NonTerminal(id) => {
                self.rules.get(&id).and_then(|rules| apply_first_rule(&rules[..], sym))
            }
        }
    }
}

pub struct SymbolGenerator {
    pub max_expr_depth: usize,

    // terminal symbols
    pub terminal_symbols: OwnedWeightedChoice<EdgeOp>,
    pub nonterminal_symbols: Range<u32>,

    /// The probability with which a terminal value is choosen.
    pub prob_terminal: Probability,

    pub expr_weighted_op: OwnedWeightedChoice<ExprOp>,
    pub flat_expr_weighted_op: OwnedWeightedChoice<FlatExprOp>,
    pub const_expr_weighted_op: OwnedWeightedChoice<ConstExprOp>,
}

impl SymbolGenerator {
    /// Generates a random production (right hand-side of a rule).
    pub fn gen_symbolstring<R, S>(&self,
                                  rng: &mut R,
                                  len: usize,
                                  symbol_arity: usize,
                                  num_params: usize,
                                  expr_depth: usize)
                                  -> SymbolString<S>
        where R: Rng,
              S: Symbol<A = EdgeAlphabet, T = f32>
    {
        SymbolString((0..len)
                         .into_iter()
                         .map(|_| self.gen_symbol(rng, symbol_arity, num_params, expr_depth))
                         .collect())
    }

    fn gen_symbol<R, S>(&self,
                        rng: &mut R,
                        symbol_arity: usize,
                        num_params: usize,
                        expr_depth: usize)
                        -> S
        where R: Rng,
              S: Symbol<A = EdgeAlphabet, T = f32>
    {
        S::from_iter(self.gen_symbol_value(rng),
                     (0..symbol_arity)
                         .into_iter()
                         .map(|_| self.gen_expr(rng, num_params, expr_depth)))

    }

    fn gen_symbol_value<R: Rng>(&self, rng: &mut R) -> EdgeAlphabet {
        if rng.gen::<ProbabilityValue>().is_probable_with(self.prob_terminal) {
            self.gen_terminal(rng)
        } else {
            self.gen_nonterminal(rng)
        }
    }

    fn gen_terminal<R: Rng>(&self, rng: &mut R) -> EdgeAlphabet {
        EdgeAlphabet::Terminal(self.terminal_symbols.ind_sample(rng))
    }

    fn gen_nonterminal<R: Rng>(&self, rng: &mut R) -> EdgeAlphabet {
        EdgeAlphabet::NonTerminal(self.nonterminal_symbols.ind_sample(rng))
    }

    fn gen_expr<R: Rng>(&self, rng: &mut R, num_params: usize, expr_depth: usize) -> Expr<f32> {
        random_expr(rng,
                    num_params,
                    expr_depth,
                    &self.expr_weighted_op,
                    &self.flat_expr_weighted_op)
    }


    /// Generate a simple condition like:
    ///     Arg(n) or 0.0 [>=] or [<=] constant expr
    fn gen_simple_rule_condition<R: Rng>(&self, rng: &mut R, num_params: usize) -> Condition<f32> {
        let lhs = if num_params > 0 {
            Expr::Arg(rng.gen_range(0, num_params))
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
}



#[derive(Clone, Debug)]
pub struct Genome {
    system: System,
}

pub struct Toolbox<N, E> {
    goal: Goal<N, E>,
    pool: Pool,
    fitness_functions: (FitnessFunction, FitnessFunction, FitnessFunction),

    // Variation parameters
    weighted_var_op: OwnedWeightedChoice<VarOp>,
    weighted_mut_op: OwnedWeightedChoice<MutOp>,
    prob_mutate_elem: Probability,

    /// Arguments to the axiom rule.
    pub axiom_args: Vec<Expr<f32>>,

    /// Maximum number of iterations of the L-system  (XXX: Limit also based on generated length)
    pub iterations: usize,

    /// Number of rules per genome.
    num_rules: usize,

    /// Length of an initial random rule production
    initial_rule_length: usize,

    /// Symbol arity
    symbol_arity: usize,

    /// Used symbol generator
    symbol_generator: SymbolGenerator,
}

impl<N: Clone + Default, E: Clone + Default> Toolbox<N, E> {
    pub fn new(goal: Goal<N, E>,
               pool: Pool,
               fitness_functions: (FitnessFunction, FitnessFunction, FitnessFunction),

               iterations: usize,
               num_rules: usize,
               initial_rule_length: usize,
               symbol_arity: usize,
               prob_terminal: Probability,
               max_expr_depth: usize,

               terminal_symbols: Vec<Weighted<EdgeOp>>,
               expr_weighted_op: Vec<Weighted<ExprOp>>,
               flat_expr_weighted_op: Vec<Weighted<FlatExprOp>>,
               const_expr_weighted_op: Vec<Weighted<ConstExprOp>>,

               weighted_var_op: Vec<Weighted<VarOp>>,
               weighted_mut_op: Vec<Weighted<MutOp>>,
               prob_mutate_elem: Probability)
               -> Toolbox<N, E> {

                   assert!(num_rules > 0);

                   // this is fixed for now, because we use fixed 2-ary symbols.
                   assert!(symbol_arity == 2);

        Toolbox {
            goal: goal,
            pool: pool,
            fitness_functions: fitness_functions,

            // XXX:
            prob_mutate_elem: prob_mutate_elem,
            weighted_var_op: OwnedWeightedChoice::new(weighted_var_op),
            weighted_mut_op: OwnedWeightedChoice::new(weighted_mut_op),

            // we use n-ary symbols, so we need n parameters. (XXX)
            axiom_args: (0..symbol_arity).map(|_| Expr::Const(0.0)).collect(),

            // maximum 3 iterations of the L-system.
            iterations: iterations,

            // We start with 20 rules per genome.
            num_rules: num_rules,

            initial_rule_length: initial_rule_length,

            symbol_arity: symbol_arity,

            symbol_generator: SymbolGenerator {
                max_expr_depth: max_expr_depth,

                terminal_symbols: OwnedWeightedChoice::new(terminal_symbols),

                nonterminal_symbols: Range::new(0, num_rules as u32),

                // The probability with which a terminal value is choosen.
                // we favor terminals over non-terminals
                prob_terminal: prob_terminal,

                expr_weighted_op: OwnedWeightedChoice::new(expr_weighted_op),
                flat_expr_weighted_op: OwnedWeightedChoice::new(flat_expr_weighted_op),
                const_expr_weighted_op: OwnedWeightedChoice::new(const_expr_weighted_op),
            },
        }
    }

    // XXX
    fn mutate_rule<R: Rng>(&self, rng: &mut R, rule: &mut Rule<Sym>) { 
        println!("Mutate rule before: {:?}", rule);

        println!("Mutate rule after: {:?}", rule);
    }

    // XXX
    // Mutate the genome, i.e. make a small change to it.
    // 
    // Modify a single symbol-rule:
    //
    //     * Change condition
    //     * Replace one symbol
    //     * Make change to symbol parameter expression (add a constant, lift to more complex expression)
    //     * insert / remove sequence of symbols.
    //
    pub fn mutate<R: Rng>(&self, rng: &mut R, ind: &Genome) -> Genome {
        let mut new_ind = ind.clone();

        let rule_to_modify = new_ind.system.random_rule_id(rng);

        if let Some(local_rules) = new_ind.system.rules.get_mut(&rule_to_modify) {
            // modify one of the rule -> successor pairs
            let len = local_rules.len();
            if len > 0 {
                let rule = &mut local_rules[rng.gen_range(0, len)]; 
                self.mutate_rule(rng, rule);
            }
            else {
                println!("no modification");
            }
        } else {
            println!("no modification");
        }

        new_ind
    }

    // XXX
    pub fn crossover<R: Rng>(&self, rng: &mut R, p1: &Genome, p2: &Genome) -> Genome {
        panic!();
        Genome { system: System::new() }
    }


    // Generate a random genome. This is used in creating a random population.
    //
    // There are many parameters that influence the creation of a random genome:
    //
    //     - Number of rules
    //     - Arity of symbols
    //     - Axiom and Length
    //     - Length of a production rule
    //     - Number of (condition, successor) pairs per rule.
    //     - Complexity of expression in Symbol
    //     - Number of Iterations.
    //
    //
    pub fn random_genome<R: Rng>(&self, rng: &mut R) -> Genome {
        let mut system = System::new();

        let arity = self.symbol_arity;

        // we start out flat.
        let expr_depth = 0;

        for rule_id in 0..self.num_rules as RuleId {
            let production = self.symbol_generator
                                 .gen_symbolstring(rng, self.initial_rule_length, arity, arity, expr_depth);
            let condition = if rule_id == 0 {
                // The axiomatic rule (rule number 0) has Condition::True.
                Condition::True
            } else {
                self.symbol_generator.gen_simple_rule_condition(rng, arity)
            };
            system.add_rule(rule_id, production, condition);
        }

        Genome { system: system }
    }
}

impl<N: Clone + Default, E: Clone + Default> Mate<Genome> for Toolbox<N, E> {
    // p1 is potentially "better" than p2
    fn mate<R: Rng>(&mut self, rng: &mut R, p1: &Genome, p2: &Genome) -> Genome {

        match self.weighted_var_op.ind_sample(rng) {
            VarOp::Copy => p1.clone(),
            VarOp::Mutate => self.mutate(rng, p1),
            VarOp::Crossover => self.crossover(rng, p1, p2)
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

                MultiObjective3::from((goal.apply_fitness_function(fitness_functions.0, &g),
                                       goal.apply_fitness_function(fitness_functions.1, &g),
                                       goal.apply_fitness_function(fitness_functions.2, &g)))

            })
                .collect()
        })
    }
}

impl Genome {
    /// Develops the L-system into a vector of edge operations
    pub fn to_edge_ops(&self, axiom_args: &[Expr<f32>], iterations: usize) -> Vec<(EdgeOp, f32)> {
        let axiom = SymbolString(vec![Sym2::new_parametric(EdgeAlphabet::NonTerminal(0),
                                                           (axiom_args[0].clone(),
                                                            axiom_args[1].clone()))]);
        println!("axiom: {:?}", axiom);

        // XXX: limit #iterations based on produced length
        let (s, iter) = self.system.develop(axiom, iterations);
        println!("produced string: {:?}", s);
        println!("stopped after iterations: {:?}", iter);

        let edge_ops: Vec<_> = s.0.into_iter().filter_map(|op| {
            match op.symbol() {
                &EdgeAlphabet::Terminal(ref edge_op) => {
                    if let Some(&Expr::Const(param)) = op.args().get(0) {
                        Some((edge_op.clone(), param.fract())) // NOTE: we only use the fractional part of the float
                    } else {
                        println!("Invalid parameter");
                        None
                    }
                }
                _ => None
            }
        }).collect();

        return edge_ops;
    }
}
