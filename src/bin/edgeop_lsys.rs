#![feature(zero_one)]

extern crate rand;
extern crate evo;
extern crate petgraph;
#[macro_use]
extern crate graph_annealing;
extern crate clap;
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
use clap::{App, Arg};
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

fn read_config() -> Config {
    let matches = App::new("nsga2_edge")
                      .arg(Arg::with_name("NGEN")
                               .long("ngen")
                               .help("Number of generations to run")
                               .takes_value(true)
                               .required(true))
                      .arg(Arg::with_name("OPS")
                               .long("ops")
                               .help("Edge operation and weight specification, e.g. \
                                      Dup:1,Split:3,Parent:2")
                               .takes_value(true)
                               .required(true))
                      .arg(Arg::with_name("GRAPH")
                               .long("graph")
                               .help("SGF graph file")
                               .takes_value(true)
                               .required(true))
                      .arg(Arg::with_name("MU")
                               .long("mu")
                               .help("Population size")
                               .takes_value(true)
                               .required(true))
                      .arg(Arg::with_name("LAMBDA")
                               .long("lambda")
                               .help("Size of offspring population")
                               .takes_value(true)
                               .required(true))
                      .arg(Arg::with_name("K")
                               .long("k")
                               .help("Tournament selection (default: 2)")
                               .takes_value(true))
                      .arg(Arg::with_name("SEED")
                               .long("seed")
                               .help("Seed value for Rng")
                               .takes_value(true))
                      .arg(Arg::with_name("VAROPS")
                               .long("varops")
                               .help("Variation operators and weight specification, e.g. \
                                      Mutate:1,LinearCrossover2:1,UniformCrossover:2,Copy:0")
                               .takes_value(true)
                               .required(true))
                      .arg(Arg::with_name("OBJECTIVES")
                               .long("objectives")
                               .help("Specify 3 objective functions comma separated (null, cc, \
                                      scc, nm, td), e.g. cc,nm,td")
                               .takes_value(true)
                               .required(true))
                      .arg(Arg::with_name("THRESHOLD")
                               .long("threshold")
                               .help("Abort if for all fitness[i] <= value[i] (default: 0,0,0)")
                               .takes_value(true)
                               .required(false))
                      .get_matches();

    // number of generations
    let ngen: usize = FromStr::from_str(matches.value_of("NGEN").unwrap()).unwrap();

    // size of population
    let mu: usize = FromStr::from_str(matches.value_of("MU").unwrap()).unwrap();

    // size of offspring population
    let lambda: usize = FromStr::from_str(matches.value_of("LAMBDA").unwrap()).unwrap();

    // tournament selection
    let k: usize = FromStr::from_str(matches.value_of("K").unwrap_or("2")).unwrap();
    assert!(k > 0);

    let seed: Vec<u64>;
    if let Some(seed_str) = matches.value_of("SEED") {
        seed = seed_str.split(",").map(|s| FromStr::from_str(s).unwrap()).collect();
    } else {
        println!("Use OsRng to generate seed..");
        let mut rng = OsRng::new().unwrap();
        seed = (0..2).map(|_| rng.next_u64()).collect();
    }

    // read objective functions
    let mut objectives_arr: Vec<FitnessFunction> = matches.value_of("OBJECTIVES")
                                                          .unwrap()
                                                          .split(",")
                                                          .map(|s| {
                                                              FitnessFunction::from_str(s).unwrap()
                                                          })
                                                          .collect();

    while objectives_arr.len() < 3 {
        objectives_arr.push(FitnessFunction::Null);
    }

    if objectives_arr.len() > 3 {
        panic!("Max 3 objectives allowed");
    }

    // read objective functions
    let mut threshold_arr: Vec<f32> = Vec::new();
    for s in matches.value_of("THRESHOLD").unwrap_or("").split(",") {
        let value: f32 = FromStr::from_str(s).unwrap();
        threshold_arr.push(value);
    }

    while threshold_arr.len() < 3 {
        threshold_arr.push(0.0);
    }

    if threshold_arr.len() > 3 {
        panic!("Max 3 threshold values allowed");
    }

    // read graph
    let graph_file = matches.value_of("GRAPH").unwrap();
    println!("Using graph file: {}", graph_file);
    let graph = {
        let mut f = BufReader::new(File::open(graph_file).unwrap());
        let graph: Graph<Unweighted, Unweighted, Directed> = PetgraphReader::from_sgf(&mut f);
        graph
    };

    // Parse weighted operation choice from command line
    let edge_ops = parse_weighted_op_list(matches.value_of("OPS").unwrap()).unwrap();

    // Parse weighted variation operators from command line
    let var_ops = parse_weighted_op_list(matches.value_of("VAROPS").unwrap()).unwrap();

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

#[allow(non_snake_case)]
fn main() {
    // rayon::initialize();

    let ncpus = num_cpus::get();
    println!("Using {} CPUs", ncpus);

    /*
    let mut s = String::new();
    let _ = File::open("runit.config").unwrap().read_to_string(&mut s).unwrap();
    println!("{}", s);
    let expr = asexp::Expr::parse_toplevel(&s).unwrap();
    println!("{}", expr);
    return;
    */

    let config = read_config();
    println!("{:#?}", config);

    let w_ops = to_weighted_vec(&config.edge_ops);
    assert!(w_ops.len() > 0);

    let w_var_ops = to_weighted_vec(&config.var_ops);
    assert!(w_var_ops.len() > 0);


    let mut toolbox = Toolbox::new(Goal::new(OptDenseDigraph::from(config.graph.clone())),
                                   Pool::new(ncpus),
                                   (config.objectives[0], config.objectives[1], config.objectives[2]),

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
                                              .map(|_| {
                                                  toolbox.random_genome(&mut rng)
                                              })
                                              .collect();


    // output initial population to stdout.
    println!("Population: {:#?}", initial_population);

    // evaluate fitness
    let fitness: Vec<_> = toolbox.fitness(&initial_population[..]);
    assert!(fitness.len() == initial_population.len());

    let mut pop = initial_population;
    let mut fit = fitness;

    for iteration in 0..config.ngen {
        print!("# {:>6}", iteration);
        let before = time::precise_time_ns();
        let (new_pop, new_fit) = nsga2::iterate(&mut rng, pop, fit, config.mu, config.lambda, config.k, &mut toolbox);
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
            println!("genome: {:?}", pop[rd.idx]);

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
