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
use graph_annealing::helper::{insert_vec_at, remove_at};
use std::str::FromStr;
use triadic_census::OptDenseDigraph;
use lindenmayer_system::{Alphabet, Condition, LSystem, Rule, Symbol, SymbolString,
                         apply_first_rule};
use lindenmayer_system::symbol::Sym2;
use lindenmayer_system::expr::Expr;
use self::edgeop::{EdgeOp, edgeops_to_graph};
use self::expr_op::{ConstExprOp, ExprOp, FlatExprOp, RecursiveExprOp, random_const_expr,
                    random_expr};
use self::cond_op::{CondOp, random_cond};
use simple_parallel::Pool;
use crossbeam;
use std::collections::BTreeMap;
use std::cmp;
use std::fmt::Debug;
use asexp::{Atom, Sexp};

use rayon::par_iter::*;

/// Rule mutation operations.
defops!{RuleMutOp;
    // Modify Condition
    ModifyCondition,

    // Modify Production
    ModifyProduction
}

/// Rule production mutation operations.
defops!{RuleProductionMutOp;
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

impl Into<Sexp> for EdgeAlphabet {
    fn into(self) -> Sexp {
        match self {
            EdgeAlphabet::Terminal(op) => op.into(),
            EdgeAlphabet::NonTerminal(rule_id) => Sexp::from(format!("@{}", rule_id)),
        }
    }
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

    fn random_rule_id<R: Rng>(&self, rng: &mut R) -> RuleId {
        let len = self.rules.len();
        assert!(len > 0);
        let nth = rng.gen_range(0, len);
        self.rules.iter().map(|(&k, _)| k).nth(nth).unwrap()
    }

    fn with_random_rule<R: Rng, F: FnMut(&mut R, &Rule<Sym>)>(&self,
                                                              rng: &mut R,
                                                              mut callback: F) {
        let rule_id = self.random_rule_id(rng);

        if let Some(local_rules) = self.rules.get(&rule_id) {
            let len = local_rules.len();
            if len > 0 {
                let idx = rng.gen_range(0, len);
                callback(rng, &local_rules[idx]);
            }
        }
    }

    fn replace_random_rule<R: Rng, F: FnMut(&mut R, &Rule<Sym>) -> Rule<Sym>>(&mut self,
                                                                              rng: &mut R,
                                                                              mut update: F) {
        let rule_to_modify = self.random_rule_id(rng);

        if let Some(local_rules) = self.rules.get_mut(&rule_to_modify) {
            // modify one of the rule -> successor pairs
            let len = local_rules.len();
            if len > 0 {
                let idx = rng.gen_range(0, len);
                let new_rule = {
                    let rule = &local_rules[idx];
                    // println!("Mutate rule before: {:?}", rule);
                    update(rng, rule)
                };
                // println!("Mutate rule after: {:?}", new_rule);
                local_rules[idx] = new_rule;
            } else {
                println!("no modification");
            }
        } else {
            println!("no modification");
        }
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



pub struct Toolbox<N: Debug, E: Debug> {
    goal: Goal<N, E>,
    pool: Pool,
    fitness_functions: (FitnessFunction, FitnessFunction, FitnessFunction),

    // Variation parameters
    var_op: OwnedWeightedChoice<VarOp>,
    rule_mut_op: OwnedWeightedChoice<RuleMutOp>,
    rule_prod_mut_op: OwnedWeightedChoice<RuleProductionMutOp>,
    recursive_expr_op: OwnedWeightedChoice<RecursiveExprOp>,

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

impl<N: Clone + Default + Debug, E: Clone + Default + Debug> Toolbox<N, E> {
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

               var_op: Vec<Weighted<VarOp>>)
               -> Toolbox<N, E> {

        assert!(num_rules > 0);

        // this is fixed for now, because we use fixed 2-ary symbols.
        assert!(symbol_arity == 2);

        Toolbox {
            goal: goal,
            pool: pool,
            fitness_functions: fitness_functions,

            // XXX:
            var_op: OwnedWeightedChoice::new(var_op),

            // XXX
            rule_mut_op: OwnedWeightedChoice::new(RuleMutOp::uniform_distribution()),
            // XXX
            rule_prod_mut_op: OwnedWeightedChoice::new(RuleProductionMutOp::uniform_distribution()),

            recursive_expr_op: OwnedWeightedChoice::new(RecursiveExprOp::uniform_distribution()),

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
    // The mutation can operate on the logic level (AND, OR, NOT) or on the expression level
    // (>=, <=, >, <, ==).
    //
    // * Specialize condition (AND)
    // * Generalized condition (OR)
    // * Flip condition (NOT)
    fn mutate_rule_condition<R: Rng>(&self, rng: &mut R, cond: Condition<f32>) -> Condition<f32> {
        // XXX: Simply negate it for now.
        Condition::Not(Box::new(cond))
    }

    // XXX
    //     * Replace one symbol
    //     * Make change to symbol parameter expression (add a constant, lift to more complex expression)
    //     * insert / remove sequence of symbols.
    fn mutate_rule_production<R: Rng>(&self,
                                      rng: &mut R,
                                      mut prod: SymbolString<Sym>)
                                      -> SymbolString<Sym> {
        let rule_prod_mut_op = self.rule_prod_mut_op.ind_sample(rng);
        match rule_prod_mut_op {
            RuleProductionMutOp::ReplaceSymbol => {
                // choose a random element, and replace it with a terminal or non-terminal
                let len = prod.0.len();
                if len > 0 {
                    let idx = rng.gen_range(0, len);
                    let sym_value = self.symbol_generator.gen_symbol_value(rng);
                    prod.0[idx].set_symbol(sym_value); // replace symbol value
                } else {
                    println!("unmodified rule production symbol");
                }
            }
            RuleProductionMutOp::ModifyParameter => {
                // choose a random element, and modify one of it's parameters
                let len = prod.0.len();
                if len > 0 {
                    let idx = rng.gen_range(0, len);

                    let mut args = prod.0[idx].args().to_vec();
                    let arity = args.len();
                    if arity > 0 {
                        let argsidx = rng.gen_range(0, arity);
                        // change the expression of argument `argsidx`
                        let expr = args[argsidx].clone();

                        let rec_expr_op = self.recursive_expr_op.ind_sample(rng);
                        let new_expr = match rec_expr_op {
                            RecursiveExprOp::Reciprocz => Expr::Recipz(Box::new(expr)),
                            RecursiveExprOp::Add => {
                                let op2 = self.symbol_generator.gen_expr(rng, self.symbol_arity /* XXX number of parameters */, 0);
                                Expr::Add(Box::new(expr), Box::new(op2))
                            }
                            RecursiveExprOp::Sub => {
                                let op2 = self.symbol_generator.gen_expr(rng, self.symbol_arity /* XXX number of parameters */, 0);
                                Expr::Sub(Box::new(expr), Box::new(op2))
                            }
                            RecursiveExprOp::Mul => {
                                let op2 = self.symbol_generator.gen_expr(rng, self.symbol_arity /* XXX number of parameters */, 0);
                                Expr::Mul(Box::new(expr), Box::new(op2))
                            }
                            RecursiveExprOp::Divz => {
                                let op2 = self.symbol_generator.gen_expr(rng, self.symbol_arity /* XXX number of parameters */, 0);
                                Expr::Divz(Box::new(expr), Box::new(op2))
                            }
                        };

                        args[argsidx] = new_expr;
                    } else {
                        println!("unmodified rule production parameter 1");
                    }

                    let sym_value = prod.0[idx].symbol().clone();
                    // replace symbol with modified parameters
                    prod.0[idx] = Sym::from_iter(sym_value, args.into_iter());
                } else {
                    println!("unmodified rule production parameter");
                }
            }
            RuleProductionMutOp::InsertSequence => {
                // Insert a sequence of random symbols (as we generate
                // during initialization of the genome).
                // Parameters:
                //     * Number of symbols to insert
                //     * Insert Position
                // Insert at most 4 symbols. XXX
                let max_number_of_symbols = cmp::min(4, cmp::max(1, prod.0.len() / 2));
                assert!(max_number_of_symbols >= 1 && max_number_of_symbols <= 4);

                let number_of_symbols = rng.gen_range(0, max_number_of_symbols) + 1;
                assert!(number_of_symbols > 0 && number_of_symbols <= max_number_of_symbols);
                let insert_position = rng.gen_range(0, prod.0.len() + 1);

                let arity = self.symbol_arity;
                let expr_depth = 0;
                let new_symbols = self.symbol_generator.gen_symbolstring(rng,
                                                                         number_of_symbols,
                                                                         arity,
                                                                         arity,
                                                                         expr_depth);

                let new_production = insert_vec_at(prod.0, new_symbols.0, insert_position);

                return SymbolString(new_production);
            }
            RuleProductionMutOp::DeleteSequence => {
                // * Number of symbols to delete
                // * At position
                let max_number_of_symbols = cmp::min(4, cmp::max(1, prod.0.len() / 2));
                assert!(max_number_of_symbols >= 1 && max_number_of_symbols <= 4);

                let number_of_symbols = rng.gen_range(0, max_number_of_symbols) + 1;
                assert!(number_of_symbols > 0 && number_of_symbols <= max_number_of_symbols);
                let remove_position = rng.gen_range(0, prod.0.len() + 1);

                let new_production = remove_at(prod.0, remove_position, number_of_symbols);

                return SymbolString(new_production);

            }
        }
        prod
    }

    fn mutate_rule<R: Rng>(&self, rng: &mut R, rule: Rule<Sym>) -> Rule<Sym> {
        let Rule {symbol, condition, successor} = rule;
        let rule_mut_op = self.rule_mut_op.ind_sample(rng);
        match rule_mut_op {
            RuleMutOp::ModifyCondition => {
                Rule {
                    symbol: symbol,
                    condition: self.mutate_rule_condition(rng, condition),
                    successor: successor,
                }
            }
            RuleMutOp::ModifyProduction => {
                Rule {
                    symbol: symbol,
                    condition: condition,
                    successor: self.mutate_rule_production(rng, successor),
                }
            }
        }
    }

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

        new_ind.system.replace_random_rule(rng, |rng, rule| self.mutate_rule(rng, rule.clone()));

        new_ind
    }

    // At first, a random rule in p2 is determined, then
    // it is copied / inserted / replaced with a
    // rule in p1.
    //
    // * Replacing one rule of p1 with a rule of p2.
    // * Insert a subsequence of a rule of p2 into p1.
    // * Replace a subsequence
    pub fn crossover<R: Rng>(&self, rng: &mut R, p1: &Genome, p2: &Genome) -> Genome {
        let mut new_ind = p1.clone();

        p2.system.with_random_rule(rng, |rng, rule_p2| {
            println!("p2 called with random rule: {:?}", rule_p2);

            new_ind.system.replace_random_rule(rng, |rng, rule_p1| {
                let new_production = linear_2point_crossover_random(rng,
                                                                    &rule_p1.successor.0,
                                                                    &rule_p2.successor.0);

                Rule {
                    successor: SymbolString(new_production),
                    condition: rule_p1.condition.clone(),
                    symbol: rule_p1.symbol.clone(),
                }
            });

        });

        new_ind
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
                                 .gen_symbolstring(rng,
                                                   self.initial_rule_length,
                                                   arity,
                                                   arity,
                                                   expr_depth);
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

impl<N: Clone + Default + Debug, E: Clone + Default + Debug> Mate<Genome> for Toolbox<N, E> {
    // p1 is potentially "better" than p2
    fn mate<R: Rng>(&mut self, rng: &mut R, p1: &Genome, p2: &Genome) -> Genome {

        match self.var_op.ind_sample(rng) {
            VarOp::Copy => p1.clone(),
            VarOp::Mutate => self.mutate(rng, p1),
            VarOp::Crossover => self.crossover(rng, p1, p2),
        }
    }
}

impl<N:Clone+Sync+Default + Debug,E:Clone+Sync+Default + Debug> FitnessEval<Genome, MultiObjective3<f32>> for Toolbox<N,E> {
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

// fn fitness(&mut self, pop: &[Genome]) -> Vec<MultiObjective3<f32>> {
// let goal = &self.goal;
// let axiom_args = &self.axiom_args[..];
// let iterations = self.iterations;
// let fitness_functions = self.fitness_functions;
//
// let mut result = Vec::new();
// pop.into_par_iter().map(|ind| {
// let edge_ops = ind.to_edge_ops(axiom_args, iterations);
// let g = edgeops_to_graph(&edge_ops);
//
// MultiObjective3::from((goal.apply_fitness_function(fitness_functions.0, &g),
// goal.apply_fitness_function(fitness_functions.1, &g),
// goal.apply_fitness_function(fitness_functions.2, &g)))
//
// }).collect_into(&mut result);
// result
// }
//

}

#[derive(Clone, Debug)]
pub struct Genome {
    system: System,
}

impl<'a> Into<Sexp> for &'a Genome {
    fn into(self) -> Sexp {
        let mut rules = Vec::<Sexp>::new();
        for (k, vec_rules) in self.system.rules.iter() {
            for rule in vec_rules.iter() {
                let sym = Into::<Sexp>::into(rule.symbol.clone());
                let cond = Into::<Sexp>::into(&rule.condition);
                let succ: Vec<Sexp> = rule.successor
                                          .0
                                          .iter()
                                          .map(|s| {
                                              let args: Vec<Sexp> = s.args()
                                                                     .iter()
                                                                     .map(|a| a.into())
                                                                     .collect();
                                              Sexp::from((Into::<Sexp>::into((*s.symbol()).clone()),
                                                          Sexp::Array(args)))
                                          })
                                          .collect();
                rules.push(Sexp::from((sym, cond, succ)));
            }
        }

        Sexp::from(("genome", rules))
    }
}

impl Genome {
    /// Develops the L-system into a vector of edge operations
    pub fn to_edge_ops(&self, axiom_args: &[Expr<f32>], iterations: usize) -> Vec<(EdgeOp, f32)> {
        let axiom = SymbolString(vec![Sym2::new_parametric(EdgeAlphabet::NonTerminal(0),
                                                           (axiom_args[0].clone(),
                                                            axiom_args[1].clone()))]);
        // XXX: limit #iterations based on produced length
        let (s, iter) = self.system.develop(axiom, iterations);
        // println!("produced string: {:?}", s);
        // println!("stopped after iterations: {:?}", iter);

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
