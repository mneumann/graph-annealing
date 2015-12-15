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
extern crate serde_json;
extern crate sexp;
extern crate graph_edge_evolution;

#[path="genome/genome_edgeop.rs"]
mod genome;

use sexp::{Sexp, atom_s};

use std::str::FromStr;
use clap::{App, Arg};
use rand::{Rng, SeedableRng};
use rand::distributions::{IndependentSample, Range};
use rand::os::OsRng;
use pcg::PcgRng;

use evo::Probability;
use evo::nsga2::{self, FitnessEval, MultiObjective3};
use genome::{EdgeOpsGenome, Toolbox};
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

struct MyEval<N, E> {
    goal: Goal<N, E>,
    pool: Pool,
    fitness_functions: (FitnessFunction, FitnessFunction, FitnessFunction),
}

#[inline]
fn fitness<N: Clone + Default, E: Clone + Default>(fitness_functions: (FitnessFunction,
                                                                       FitnessFunction,
                                                                       FitnessFunction),
                                                   goal: &Goal<N, E>,
                                                   ind: &EdgeOpsGenome)
                                                   -> MultiObjective3<f32> {
    let g = ind.to_graph();
    MultiObjective3::from((goal.apply_fitness_function(fitness_functions.0, &g),
                           goal.apply_fitness_function(fitness_functions.1, &g),
                           goal.apply_fitness_function(fitness_functions.2, &g)))

}

impl<N:Clone+Sync+Default,E:Clone+Sync+Default> FitnessEval<EdgeOpsGenome, MultiObjective3<f32>> for MyEval<N,E> {
    fn fitness(&mut self, pop: &[EdgeOpsGenome]) -> Vec<MultiObjective3<f32>> {
        let pool = &mut self.pool;
        let goal = &self.goal;

        let fitness_functions = self.fitness_functions;

        crossbeam::scope(|scope| {
            pool.map(scope, pop, |ind| fitness(fitness_functions, goal, ind))
                .collect()
        })
    }
}

#[allow(non_snake_case)]
fn main() {
    let ncpus = num_cpus::get();
    println!("Using {} CPUs", ncpus);
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
                      .arg(Arg::with_name("MUTOPS")
                               .long("mutops")
                               .help("Mutation operation and weight specification, e.g. \
                                      Insert:1,Remove:1,Replace:2,ModifyOp:0,ModifyParam:2,Copy:\
                                      0")
                               .takes_value(true)
                               .required(true))
                      .arg(Arg::with_name("MUTP")
                               .long("mutp")
                               .help("Probability for element mutation")
                               .takes_value(true)
                               .required(true))
                      .arg(Arg::with_name("ILEN")
                               .long("ilen")
                               .help("Initial genome length (random range)")
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
    let NGEN: usize = FromStr::from_str(matches.value_of("NGEN").unwrap()).unwrap();
    println!("NGEN: {}", NGEN);

    // size of population
    let MU: usize = FromStr::from_str(matches.value_of("MU").unwrap()).unwrap();
    println!("MU: {}", MU);

    // size of offspring population
    let LAMBDA: usize = FromStr::from_str(matches.value_of("LAMBDA").unwrap()).unwrap();
    println!("LAMBDA: {}", LAMBDA);

    // tournament selection
    let K: usize = FromStr::from_str(matches.value_of("K").unwrap_or("2")).unwrap();
    assert!(K > 0);
    println!("K: {}", K);

    let seed: Vec<u64>;
    if let Some(seed_str) = matches.value_of("SEED") {
        seed = seed_str.split(",").map(|s| FromStr::from_str(s).unwrap()).collect();
    } else {
        println!("Use OsRng to generate seed..");
        let mut rng = OsRng::new().unwrap();
        seed = (0..2).map(|_| rng.next_u64()).collect();
    }
    println!("SEED: {:?}", seed);

    let MUTP: f32 = FromStr::from_str(matches.value_of("MUTP").unwrap()).unwrap();
    println!("MUTP: {:?}", MUTP);

    // Parse weighted operation choice from command line
    let mutops = parse_weighted_op_list(matches.value_of("MUTOPS").unwrap()).unwrap();
    println!("mut ops: {:?}", mutops);

    let ilen_str = matches.value_of("ILEN").unwrap();
    let ilen: Vec<usize> = ilen_str.split(",").map(|s| FromStr::from_str(s).unwrap()).collect();
    assert!(ilen.len() == 1 || ilen.len() == 2);

    let ilen_from = ilen[0];
    let ilen_to = if ilen.len() == 1 {
        ilen[0]
    } else {
        ilen[1]
    };
    assert!(ilen_from <= ilen_to);

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

    println!("objectives={:?}", objectives_arr);

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
    let ops = parse_weighted_op_list(matches.value_of("OPS").unwrap()).unwrap();
    println!("edge ops: {:?}", ops);

    // Parse weighted variation operators from command line
    let varops = parse_weighted_op_list(matches.value_of("VAROPS").unwrap()).unwrap();
    println!("var ops: {:?}", varops);

    let w_ops = to_weighted_vec(&ops);
    assert!(w_ops.len() > 0);

    let w_var_ops = to_weighted_vec(&varops);
    assert!(w_var_ops.len() > 0);

    let w_mut_ops = to_weighted_vec(&mutops);
    assert!(w_mut_ops.len() > 0);

    let mut toolbox = Toolbox::new(w_ops, w_var_ops, w_mut_ops, Probability::new(MUTP));

    let mut evaluator = MyEval {
        goal: Goal::new(OptDenseDigraph::from(graph)),
        pool: Pool::new(ncpus),
        fitness_functions: (objectives_arr[0], objectives_arr[1], objectives_arr[2]),
    };

    assert!(seed.len() == 2);
    let mut rng: PcgRng = SeedableRng::from_seed([seed[0], seed[1]]);

    // create initial random population
    let initial_population: Vec<EdgeOpsGenome> = (0..MU)
                                                     .map(|_| {
                                                         let len = if ilen_from == ilen_to {
                                                             ilen_from
                                                         } else {
                                                             Range::new(ilen_from, ilen_to)
                                                                 .ind_sample(&mut rng)
                                                         };
                                                         toolbox.random_genome(&mut rng, len)
                                                     })
                                                     .collect();


    // output initial population to stdout.
    let sexp_pop = Sexp::List(vec![atom_s("Population"), 
        Sexp::List(initial_population.iter().map(|ind| ind.to_sexp()).collect()),
    ]);
    println!("{}", sexp_pop);

    // evaluate fitness
    let fitness: Vec<_> = evaluator.fitness(&initial_population[..]);
    assert!(fitness.len() == initial_population.len());

    let mut pop = initial_population;
    let mut fit = fitness;

    for iteration in 0..NGEN {
        print!("# {:>6}", iteration);
        let before = time::precise_time_ns();
        let (new_pop, new_fit) = nsga2::iterate(&mut rng,
                                                pop,
                                                fit,
                                                &mut evaluator,
                                                MU,
                                                LAMBDA,
                                                K,
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
            if f.objectives[0] <= threshold_arr[0] && f.objectives[1] <= threshold_arr[1] &&
               f.objectives[2] <= threshold_arr[2] {
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
    let rank_dist = nsga2::select(&fit[..], MU);
    assert!(rank_dist.len() == MU);

    let mut j = 0;

    for rd in rank_dist.iter() {
        if rd.rank == 0 {
            println!("-------------------------------------------");
            println!("rd: {:?}", rd);
            println!("fitness: {:?}", fit[rd.idx]);
            println!("genome: {:?}", pop[rd.idx]);

            // if fit[rd.idx].objectives[0] < 1.0 {
            draw_graph((&pop[rd.idx].to_graph()).ref_graph(),
                       // XXX: name
                       &format!("nsga2edgeops_g{}_f{}_i{}.svg",
                                NGEN,
                                fit[rd.idx].objectives[1] as usize,
                                j));
            j += 1;
            // }
        }
    }
}
