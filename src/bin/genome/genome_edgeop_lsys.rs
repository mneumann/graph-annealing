// Edge Operation L-System Genome


// NOTES:
//
// * When we need a value within [0, 1), we simply cut off the integral part und use fractional part
//   only.
//
// * We start out with a simple Genome, which gets more and more complex through mating.

pub mod edgeop;
pub mod expr_op;
mod cond_op;

use evo::prob::{Probability, ProbabilityValue};
use evo::crossover::linear_2point_crossover_random;
use evo::nsga2::{FitnessEval, Mate};
use evo::mo::MultiObjective3;
use rand::Rng;
use rand::distributions::{IndependentSample, Weighted};
use rand::distributions::range::Range;
use graph_annealing::owned_weighted_choice::OwnedWeightedChoice;
use graph_annealing::goal::{Cache, FitnessFunction, Goal};
use graph_annealing::helper::{insert_vec_at, remove_at};
use lindenmayer_system::{Alphabet, DualAlphabet};
use lindenmayer_system::parametric::{PDualMapSystem, PRule, PSym2, ParametricRule,
                                     ParametricSymbol, ParametricSystem};
use expression_num::NumExpr as Expr;
use expression::cond::Cond;
use self::edgeop::{EdgeOp, edgeops_to_graph};
use self::expr_op::{FlatExprOp, RecursiveExprOp, random_flat_expr};
use simple_parallel::Pool;
use crossbeam;
use std::cmp;
use std::fmt::Debug;
use asexp::Sexp;

// How many symbols to insert at most using InsertSequence
const INS_MAX_NUMBER_OF_SYMBOLS: usize = 1;
// How many symbols to delete at most using DeleteSequence
const DEL_MAX_NUMBER_OF_SYMBOLS: usize = 1;

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

impl DualAlphabet for EdgeAlphabet {
    type Terminal = EdgeOp;
    type NonTerminal = RuleId;

    fn nonterminal(&self) -> Option<&Self::NonTerminal> {
        match self {
            &EdgeAlphabet::Terminal(..) => None,
            &EdgeAlphabet::NonTerminal(ref nt) => Some(nt),
        }
    }

    fn terminal(&self) -> Option<&Self::Terminal> {
        match self {
            &EdgeAlphabet::Terminal(ref t) => Some(t),
            &EdgeAlphabet::NonTerminal(..) => None,
        }
    }
}

// We use 2-ary symbols, i.e. symbols with two parameters.
const SYM_ARITY: usize = 2;
type Sym = PSym2<EdgeAlphabet, Expr<f32>>;
type SymParam = PSym2<EdgeAlphabet, f32>;
type Rule = PRule<EdgeAlphabet, Sym, PSym2<EdgeAlphabet, f32>, Cond<Expr<f32>>>;
type System = PDualMapSystem<EdgeAlphabet, Rule>;

pub struct SymbolGenerator {
    // terminal symbols
    pub terminal_symbols: OwnedWeightedChoice<EdgeOp>,
    pub nonterminal_symbols: Range<u32>,

    /// The probability with which a terminal value (i.e. an EdgeOp) is choosen.
    pub prob_terminal: Probability,

    pub flat_expr_weighted_op: OwnedWeightedChoice<FlatExprOp>,
}

impl SymbolGenerator {
    /// Generates a random production (right hand-side of a rule).
    pub fn gen_symbolstring<R, S>(&self,
                                  rng: &mut R,
                                  len: usize,
                                  symbol_arity: usize,
                                  num_params: usize)
                                  -> Vec<S>
        where R: Rng,
              S: ParametricSymbol<Sym = EdgeAlphabet, Param = Expr<f32>>
    {
        (0..len)
            .into_iter()
            .map(|_| self.gen_symbol(rng, symbol_arity, num_params))
            .collect()
    }

    fn gen_symbol<R, S>(&self, rng: &mut R, symbol_arity: usize, num_params: usize) -> S
        where R: Rng,
              S: ParametricSymbol<Sym = EdgeAlphabet, Param = Expr<f32>>
    {
        S::new_from_iter(self.gen_symbol_value(rng),
                         (0..symbol_arity)
                             .into_iter()
                             .map(|_| self.gen_expr(rng, num_params)))
            .unwrap()

    }

    fn gen_symbol_value<R: Rng>(&self, rng: &mut R) -> EdgeAlphabet {
        if rng.gen::<ProbabilityValue>().is_probable_with(self.prob_terminal) {
            EdgeAlphabet::Terminal(self.terminal_symbols.ind_sample(rng))
        } else {
            EdgeAlphabet::NonTerminal(self.nonterminal_symbols.ind_sample(rng))
        }
    }

    // move into crate expression-num
    fn gen_expr<R: Rng>(&self, rng: &mut R, num_params: usize) -> Expr<f32> {
        random_flat_expr(rng, &self.flat_expr_weighted_op, num_params)
    }


    /// Generate a simple condition like:
    ///     Arg(n) or 0.0 [>=] or [<=] flat expr
    fn gen_simple_rule_condition<R: Rng>(&self, rng: &mut R, num_params: usize) -> Cond<Expr<f32>> {
        let lhs = if num_params > 0 {
            Expr::Var(rng.gen_range(0, num_params))
        } else {
            Expr::Const(0.0)
        };

        let rhs = random_flat_expr(rng, &self.flat_expr_weighted_op, num_params);

        if rng.gen::<bool>() {
            Cond::GreaterEqual(Box::new(lhs), Box::new(rhs))
        } else {
            Cond::LessEqual(Box::new(lhs), Box::new(rhs))
        }
    }
}

pub struct Toolbox<N: Debug, E: Debug> {
    goal: Goal<N, E>,
    pool: Pool,
    fitness_functions: Vec<FitnessFunction>,

    // Variation parameters
    var_op: OwnedWeightedChoice<VarOp>,
    rule_mut_op: OwnedWeightedChoice<RuleMutOp>,
    rule_prod_mut_op: OwnedWeightedChoice<RuleProductionMutOp>,
    recursive_expr_op: OwnedWeightedChoice<RecursiveExprOp>,

    /// Arguments to the axiom rule.
    pub axiom_args: Vec<f32>,

    /// Maximum number of iterations of the L-system  (XXX: Limit also based on generated length)
    pub iterations: usize,

    /// Number of rules per genome.
    num_rules: usize,

    /// Length of an initial random rule production
    initial_rule_length: usize,

    /// Symbol arity
    symbol_arity: usize,

    /// Number of available parameters (<= symbol_arity).
    num_params: usize,

    /// Used symbol generator
    symbol_generator: SymbolGenerator,
}

impl<N: Clone + Default + Debug, E: Clone + Default + Debug> Toolbox<N, E> {
    pub fn new(goal: Goal<N, E>,
               pool: Pool,
               fitness_functions: Vec<FitnessFunction>,

               iterations: usize,
               num_rules: usize,
               initial_rule_length: usize,
               symbol_arity: usize,
               num_params: usize,
               prob_terminal: Probability,

               terminal_symbols: Vec<Weighted<EdgeOp>>,
               flat_expr_weighted_op: Vec<Weighted<FlatExprOp>>,
               recursive_expr_op: Vec<Weighted<RecursiveExprOp>>,

               var_op: Vec<Weighted<VarOp>>,
               rule_mut_op: Vec<Weighted<RuleMutOp>>,
               rule_prod_mut_op: Vec<Weighted<RuleProductionMutOp>>,
               )
               -> Toolbox<N, E> {

        assert!(terminal_symbols.len() > 0);
        assert!(flat_expr_weighted_op.len() > 0);
        assert!(var_op.len() > 0);
        assert!(rule_mut_op.len() > 0);
        assert!(rule_prod_mut_op.len() > 0);
        assert!(recursive_expr_op.len() > 0);

        assert!(num_rules > 0);

        // this is fixed for now, because we use fixed 2-ary symbols.
        assert!(symbol_arity == 2);

        assert!(num_params <= symbol_arity);

        Toolbox {
            goal: goal,
            pool: pool,
            fitness_functions: fitness_functions,

            var_op: OwnedWeightedChoice::new(var_op),
            rule_mut_op: OwnedWeightedChoice::new(rule_mut_op),
            rule_prod_mut_op: OwnedWeightedChoice::new(rule_prod_mut_op),

            recursive_expr_op: OwnedWeightedChoice::new(recursive_expr_op),

            // we use n-ary symbols, so we need n parameters. (XXX)
            axiom_args: (0..symbol_arity).map(|_| 0.0).collect(),

            // maximum 3 iterations of the L-system.
            iterations: iterations,

            // We start with 20 rules per genome.
            num_rules: num_rules,

            initial_rule_length: initial_rule_length,

            symbol_arity: symbol_arity,
            num_params: num_params,

            symbol_generator: SymbolGenerator {
                terminal_symbols: OwnedWeightedChoice::new(terminal_symbols),

                nonterminal_symbols: Range::new(0, num_rules as u32),

                // The probability with which a terminal value is choosen.
                // we favor terminals over non-terminals
                prob_terminal: prob_terminal,

                flat_expr_weighted_op: OwnedWeightedChoice::new(flat_expr_weighted_op),
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
    fn mutate_rule_condition<R: Rng>(&self,
                                     _rng: &mut R,
                                     cond: Cond<Expr<f32>>)
                                     -> Cond<Expr<f32>> {
        // XXX: Simply negate it for now.
        Cond::Not(Box::new(cond))
    }

    // XXX
    //     * Replace one symbol
    //     * Make change to symbol parameter expression (add a constant, lift to more complex expression)
    //     * insert / remove sequence of symbols.
    fn mutate_rule_production<R: Rng>(&self, rng: &mut R, mut prod: Vec<Sym>) -> Vec<Sym> {
        let rule_prod_mut_op = self.rule_prod_mut_op.ind_sample(rng);
        match rule_prod_mut_op {
            RuleProductionMutOp::ReplaceSymbol => {
                // choose a random element, and replace it with a terminal or non-terminal
                let len = prod.len();
                if len > 0 {
                    let idx = rng.gen_range(0, len);
                    let sym_value = self.symbol_generator.gen_symbol_value(rng);
                    *prod[idx].symbol_mut() = sym_value; // replace symbol value
                } else {
                    //println!("unmodified rule production symbol");
                }
            }
            RuleProductionMutOp::ModifyParameter => {
                // choose a random element, and modify one of it's parameters
                let len = prod.len();
                if len > 0 {
                    let idx = rng.gen_range(0, len);

                    let mut args = prod[idx].params().to_vec();
                    let arity = args.len();
                    if arity > 0 {
                        let argsidx = rng.gen_range(0, arity);
                        // change the expression of argument `argsidx`
                        let expr = args[argsidx].clone();

                        let rec_expr_op = self.recursive_expr_op.ind_sample(rng);
                        let new_expr = match rec_expr_op {
                            RecursiveExprOp::Reciprocz => Expr::Recipz(Box::new(expr)),
                            RecursiveExprOp::Add => {
                                let op2 = self.symbol_generator.gen_expr(rng, self.num_params);
                                Expr::Add(Box::new(expr), Box::new(op2))
                            }
                            RecursiveExprOp::Sub => {
                                let op2 = self.symbol_generator.gen_expr(rng, self.num_params);
                                Expr::Sub(Box::new(expr), Box::new(op2))
                            }
                            RecursiveExprOp::Mul => {
                                let op2 = self.symbol_generator.gen_expr(rng, self.num_params);
                                Expr::Mul(Box::new(expr), Box::new(op2))
                            }
                            RecursiveExprOp::Divz => {
                                let op2 = self.symbol_generator.gen_expr(rng, self.num_params);
                                Expr::Divz(Box::new(expr), Box::new(op2))
                            }
                        };

                        args[argsidx] = new_expr;
                    } else {
                        //println!("unmodified rule production parameter 1");
                    }

                    let sym_value = prod[idx].symbol().clone();
                    // replace symbol with modified parameters
                    prod[idx] = Sym::new_from_iter(sym_value, args.into_iter()).unwrap();
                } else {
                    //println!("unmodified rule production parameter");
                }
            }
            RuleProductionMutOp::InsertSequence => {
                // Insert a sequence of random symbols (as we generate
                // during initialization of the genome).
                // Parameters:
                //     * Number of symbols to insert
                //     * Insert Position
                // Insert at most INS_MAX_NUMBER_OF_SYMBOLS symbols.
                let max_number_of_symbols = cmp::min(INS_MAX_NUMBER_OF_SYMBOLS, cmp::max(1, prod.len() / 2));
                assert!(max_number_of_symbols >= 1 && max_number_of_symbols <= INS_MAX_NUMBER_OF_SYMBOLS);

                let number_of_symbols = rng.gen_range(0, max_number_of_symbols) + 1;
                assert!(number_of_symbols > 0 && number_of_symbols <= max_number_of_symbols);
                let insert_position = rng.gen_range(0, prod.len() + 1);

                let new_symbols = self.symbol_generator
                                      .gen_symbolstring(rng, number_of_symbols, self.symbol_arity, self.num_params);

                let new_production = insert_vec_at(prod, new_symbols, insert_position);

                return new_production;
            }
            RuleProductionMutOp::DeleteSequence => {
                // * Number of symbols to delete
                // * At position
                // Delete at most DEL_MAX_NUMBER_OF_SYMBOLS symbols.
                let max_number_of_symbols = cmp::min(DEL_MAX_NUMBER_OF_SYMBOLS, cmp::max(1, prod.len() / 2));
                assert!(max_number_of_symbols >= 1 && max_number_of_symbols <= DEL_MAX_NUMBER_OF_SYMBOLS);

                let number_of_symbols = rng.gen_range(0, max_number_of_symbols) + 1;
                assert!(number_of_symbols > 0 && number_of_symbols <= max_number_of_symbols);
                let remove_position = rng.gen_range(0, prod.len() + 1);

                let new_production = remove_at(prod, remove_position, number_of_symbols);

                return new_production;
            }
        }
        prod
    }

    fn mutate_rule<R: Rng>(&self, rng: &mut R, rule: Rule) -> Rule {
        let Rule {symbol, condition, production, arity, ..} = rule;
        let rule_mut_op = self.rule_mut_op.ind_sample(rng);
        match rule_mut_op {
            RuleMutOp::ModifyCondition => {
                Rule::new(symbol,
                          self.mutate_rule_condition(rng, condition),
                          production,
                          arity)
            }
            RuleMutOp::ModifyProduction => {
                Rule::new(symbol,
                          condition,
                          self.mutate_rule_production(rng, production),
                          arity)
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

        new_ind.system.with_random_rule_mut(rng, |rng, opt_rule| {
            if let Some(rule) = opt_rule {
                *rule = self.mutate_rule(rng, rule.clone());
            }
        });

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

        p2.system.with_random_rule(rng, |rng, opt_rule_p2| {
            // println!("p2 called with random rule: {:?}", rule_p2);
            if let Some(rule_p2) = opt_rule_p2 {

                new_ind.system.with_random_rule_mut(rng, |rng, opt_rule_p1| {
                    if let Some(rule_p1) = opt_rule_p1 {
                        let new_production = linear_2point_crossover_random(rng,
                                                                            &rule_p1.production,
                                                                            &rule_p2.production);
                        rule_p1.production = new_production;
                    } else {
                        println!("no modification");
                    }
                });
            }
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

        for rule_id in 0..self.num_rules as RuleId {
            let production = self.symbol_generator
                                 .gen_symbolstring(rng, self.initial_rule_length, self.symbol_arity, self.num_params);
            let condition = if rule_id == 0 {
                // The axiomatic rule (rule number 0) has Cond::True.
                Cond::True
            } else {
                self.symbol_generator.gen_simple_rule_condition(rng, self.num_params)
            };
            system.add_rule(Rule::new(EdgeAlphabet::NonTerminal(rule_id),
                                      condition,
                                      production,
                                      SYM_ARITY));
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

impl FitnessEval<Genome, MultiObjective3<f32>> for Toolbox<f32, f32> {
    /// Evaluates the fitness of a Genome population.
    fn fitness(&mut self, pop: &[Genome]) -> Vec<MultiObjective3<f32>> {

        let pool = &mut self.pool;
        let goal = &self.goal;
        let axiom_args = &self.axiom_args[..];
        let iterations = self.iterations;

        let fitness_functions = &self.fitness_functions[..];

        crossbeam::scope(|scope| {
            pool.map(scope, pop, |ind| {
                    let edge_ops = ind.to_edge_ops(axiom_args, iterations);
                    let g = edgeops_to_graph(&edge_ops);
                    let mut cache = Cache::new();

                    MultiObjective3::from(fitness_functions.iter().map(|&f| {
                        goal.apply_fitness_function(f, &g, &mut cache)
                    }))
                })
                .collect()
        })
    }
}

#[derive(Clone, Debug)]
pub struct Genome {
    system: System,
}

impl<'a> Into<Sexp> for &'a Genome {
    fn into(self) -> Sexp {
        let mut rules = Vec::<Sexp>::new();
        self.system.each_rule(|rule| {
            let sym = Into::<Sexp>::into(rule.symbol.clone());
            let cond = Into::<Sexp>::into(&rule.condition);
            let succ: Vec<Sexp> = rule.production
                                      .iter()
                                      .map(|s| {
                                          let args: Vec<Sexp> = s.params()
                                                                 .iter()
                                                                 .map(|a| a.into())
                                                                 .collect();
                                          Sexp::from((Into::<Sexp>::into((*s.symbol()).clone()),
                                                      Sexp::Array(args)))
                                      })
                                      .collect();
            rules.push(Sexp::from((sym, cond, succ)));
        });

        Sexp::from(("genome", rules))
    }
}

impl Genome {
    /// Develops the L-system into a vector of edge operations
    pub fn to_edge_ops(&self, axiom_args: &[f32], iterations: usize) -> Vec<(EdgeOp, f32)> {
        let axiom = vec![SymParam::new_from_iter(EdgeAlphabet::NonTerminal(0),
                                                 axiom_args.iter().take(SYM_ARITY).cloned())
                             .unwrap()];
        // XXX: limit #iterations based on produced length
        let (s, _iter) = self.system.develop(axiom, iterations);
        // println!("produced string: {:?}", s);
        // println!("stopped after iterations: {:?}", iter);

        let edge_ops: Vec<_> = s.into_iter().filter_map(|op| {
            match op.symbol() {
                &EdgeAlphabet::Terminal(ref edge_op) => {
                    if let Some(&param) = op.params().get(0) {
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
