#![feature(zero_one)]

extern crate rand;
extern crate evo;
extern crate petgraph;
#[macro_use]
extern crate graph_annealing;
extern crate crossbeam;
extern crate simple_parallel;
extern crate num_cpus;
extern crate pcg;
extern crate graph_sgf;
extern crate triadic_census;
extern crate time;
extern crate lindenmayer_system;
extern crate graph_edge_evolution;
extern crate asexp;

extern crate rayon;

#[path="genome/genome_edgeop_lsys.rs"]
pub mod genome;

use std::str::FromStr;
use rand::{Rng, SeedableRng};
use rand::distributions::{IndependentSample, Range};
use rand::os::OsRng;
use pcg::PcgRng;
use evo::Probability;
use evo::nsga2::{self, FitnessEval};
use genome::{Genome, Toolbox};
use graph_annealing::helper::{draw_graph, parse_weighted_op_list, to_weighted_vec};
use graph_annealing::goal::Goal;
use graph_annealing::stat::Stat;
use graph_annealing::fitness_function::FitnessFunction;
use simple_parallel::Pool;
use petgraph::{Directed, Graph};
use graph_sgf::{PetgraphReader, Unweighted};
use triadic_census::OptDenseDigraph;
use std::io::BufReader;
use std::fs::File;
use genome::VarOp;
use genome::edgeop::{EdgeOp, edgeops_to_graph};
use genome::expr_op::{ConstExprOp, ExprOp, FlatExprOp};
use std::io::Read;
use asexp::Sexp;
use asexp::sexp::prettyprint;
use std::env;

#[derive(Debug)]
struct Config {
    ngen: usize,
    mu: usize,
    lambda: usize,
    k: usize,
    seed: Vec<u64>,
    objectives: Vec<FitnessFunction>,
    thresholds: Vec<f32>,
    graph: Graph<Unweighted, Unweighted, Directed>,
    edge_ops: Vec<(EdgeOp, u32)>,
    var_ops: Vec<(VarOp, u32)>,
}

fn parse_config(sexp: Sexp) -> Config {
    let map = sexp.into_map().unwrap();

    // number of generations
    let ngen: usize = map.get("ngen").and_then(|v| v.get_uint()).unwrap() as usize;

    // size of population
    let mu: usize = map.get("mu").and_then(|v| v.get_uint()).unwrap() as usize;

    // size of offspring population
    let lambda: usize = map.get("lambda").and_then(|v| v.get_uint()).unwrap() as usize;

    // tournament selection
    let k: usize = map.get("k").and_then(|v| v.get_uint()).unwrap_or(2) as usize;
    assert!(k > 0);

    let seed: Vec<u64>;
    if let Some(seed_expr) = map.get("seed") {
        seed = seed_expr.get_uint_vec().unwrap();
    } else {
        println!("Use OsRng to generate seed..");
        let mut rng = OsRng::new().unwrap();
        seed = (0..2).map(|_| rng.next_u64()).collect();
    }

    // read objective functions
    let mut objectives_arr: Vec<FitnessFunction> = map.get("objectives")
                                                      .unwrap()
                                                      .get_vec(|elm| {
                                                          FitnessFunction::from_str(elm.get_str()
                                                                                       .unwrap())
                                                              .ok()
                                                      })
                                                      .unwrap();

    while objectives_arr.len() < 3 {
        objectives_arr.push(FitnessFunction::Null);
    }

    if objectives_arr.len() > 3 {
        panic!("Max 3 objectives allowed");
    }

    // read objective functions
    let mut threshold_arr: Vec<f32> = map.get("thresholds")
                                         .unwrap()
                                         .get_vec(|elm| elm.get_float())
                                         .unwrap()
                                         .iter()
                                         .map(|&i| i as f32)
                                         .collect();

    while threshold_arr.len() < 3 {
        threshold_arr.push(0.0);
    }

    if threshold_arr.len() > 3 {
        panic!("Max 3 threshold values allowed");
    }

    // read graph
    let graph_file = map.get("graph").unwrap().get_str().unwrap();
    println!("Using graph file: {}", graph_file);
    let graph = {
        let mut f = BufReader::new(File::open(graph_file).unwrap());
        let graph: Graph<Unweighted, Unweighted, Directed> = PetgraphReader::from_sgf(&mut f);
        graph
    };

    // Parse weighted operation choice from command line
    let mut edge_ops: Vec<(EdgeOp, u32)> = Vec::new();
    if let Some(&Sexp::Map(ref list)) = map.get("edgeops") {
        for &(ref k, ref v) in list.iter() {
            edge_ops.push((EdgeOp::from_str(k.get_str().unwrap()).unwrap(),
                           v.get_uint().unwrap() as u32));
        }
    } else {
        panic!();
    }

    // Parse weighted variation operators from command line
    let mut var_ops: Vec<(VarOp, u32)> = Vec::new();
    if let Some(&Sexp::Map(ref list)) = map.get("varops") {
        for &(ref k, ref v) in list.iter() {
            var_ops.push((VarOp::from_str(k.get_str().unwrap()).unwrap(),
                          v.get_uint().unwrap() as u32));
        }
    } else {
        panic!();
    }

    Config {
        ngen: ngen,
        mu: mu,
        lambda: lambda,
        k: k,
        seed: seed,
        objectives: objectives_arr,
        thresholds: threshold_arr,
        graph: graph,
        edge_ops: edge_ops,
        var_ops: var_ops,
    }
}

fn pp_sexp(s: &Sexp) {
    let mut st = String::new();
    let _ = prettyprint(&s, &mut st, 0, false).unwrap();
    println!("{}", st);
}

fn main() {
    let ncpus = num_cpus::get();
    println!("Using {} CPUs", ncpus);

    let mut s = String::new();
    let configfile = env::args().nth(1).unwrap();
    let _ = File::open(configfile).unwrap().read_to_string(&mut s).unwrap();
    let expr = asexp::Sexp::parse_toplevel(&s).unwrap();
    let config = parse_config(expr);

    println!("{:#?}", config);

    let w_ops = to_weighted_vec(&config.edge_ops);
    assert!(w_ops.len() > 0);

    let w_var_ops = to_weighted_vec(&config.var_ops);
    assert!(w_var_ops.len() > 0);

    let mut toolbox = Toolbox::new(Goal::new(OptDenseDigraph::from(config.graph.clone())),
                                   Pool::new(ncpus),
                                   (config.objectives[0],
                                    config.objectives[1],
                                    config.objectives[2]),

                                   3, // iterations
                                   20, // num_rules
                                   4, // initial rule length
                                   2, // we use 2-ary symbols
                                   Probability::new(0.7), // prob_terminal
                                   2, // max_expr_depth

                                   w_ops,
                                   ExprOp::uniform_distribution(),
                                   FlatExprOp::uniform_distribution(),
                                   ConstExprOp::uniform_distribution(),
                                   w_var_ops);

    assert!(config.seed.len() == 2);
    let mut rng: PcgRng = SeedableRng::from_seed([config.seed[0], config.seed[1]]);

    // create initial random population
    let initial_population: Vec<Genome> = (0..config.mu)
                                              .map(|_| toolbox.random_genome(&mut rng))
                                              .collect();

    // output initial population to stdout.
    for ind in initial_population.iter() {
        pp_sexp(&ind.into());
    }

    // evaluate fitness
    let fitness: Vec<_> = toolbox.fitness(&initial_population[..]);
    assert!(fitness.len() == initial_population.len());

    let mut pop = initial_population;
    let mut fit = fitness;

    for iteration in 0..config.ngen {
        print!("# {:>6}", iteration);
        let before = time::precise_time_ns();
        let (new_pop, new_fit) = nsga2::iterate(&mut rng,
                                                pop,
                                                fit,
                                                config.mu,
                                                config.lambda,
                                                config.k,
                                                &mut toolbox);
        let duration = time::precise_time_ns() - before;
        pop = new_pop;
        fit = new_fit;

        // calculate a min/max/avg value for each objective.
        let stat0: Stat<f32> = Stat::for_objectives(&fit[..], 0);
        let stat1: Stat<f32> = Stat::for_objectives(&fit[..], 1);
        let stat2: Stat<f32> = Stat::for_objectives(&fit[..], 2);

        let duration_ms = (duration as f32) / 1_000_000.0;

        let mut num_optima = 0;
        for f in fit.iter() {
            if f.objectives[0] <= config.thresholds[0] && f.objectives[1] <= config.thresholds[1] &&
               f.objectives[2] <= config.thresholds[2] {
                num_optima += 1;
            }
        }

        print!(" | ");
        print!("{:>8.1}", stat0.min);
        print!("{:>9.1}", stat0.avg);
        print!("{:>10.1}", stat0.max);
        print!(" | ");
        print!("{:>8.1}", stat1.min);
        print!("{:>9.1}", stat1.avg);
        print!("{:>10.1}", stat1.max);
        print!(" | ");
        print!("{:>8.1}", stat2.min);
        print!("{:>9.1}", stat2.avg);
        print!("{:>10.1}", stat2.max);

        print!(" | {:>5} | {:>8.0} ms", num_optima, duration_ms);

        println!("");

        if num_optima > 0 {
            println!("Found premature optimum in Iteration {}", iteration);
            break;
        }
    }
    println!("===========================================================");

    // finally evaluate rank and crowding distance (using select()).
    let rank_dist = nsga2::select(&fit[..], config.mu);
    assert!(rank_dist.len() == config.mu);

    let mut j = 0;

    for rd in rank_dist.iter() {
        if rd.rank == 0 {
            println!("-------------------------------------------");
            println!("rd: {:?}", rd);
            println!("fitness: {:?}", fit[rd.idx]);
            // println!("genome: {:?}", pop[rd.idx]);
            pp_sexp(&(&pop[rd.idx]).into());

            // if fit[rd.idx].objectives[0] < 1.0 {
            let ind = &pop[rd.idx];
            let edge_ops = ind.to_edge_ops(&toolbox.axiom_args, toolbox.iterations);
            let g = edgeops_to_graph(&edge_ops);

            draw_graph(g.ref_graph(),
                       // XXX: name
                       &format!("edgeop_lsys_g{}_f{}_i{}.svg",
                                config.ngen,
                                fit[rd.idx].objectives[1] as usize,
                                j));
            j += 1;
            // }
        }
    }
}
