extern crate rand;
extern crate evo;
extern crate petgraph;
extern crate graph_annealing;
extern crate clap;
extern crate crossbeam;
extern crate simple_parallel;
extern crate num_cpus;
extern crate pcg;
extern crate graph_sgf;

use std::f32;
use std::fmt::Debug;
use std::str::FromStr;
use clap::{App, Arg};
use rand::{Rng, SeedableRng};
use rand::distributions::{IndependentSample, Range};
use rand::os::OsRng;
use pcg::PcgRng;

use evo::Probability;
use evo::nsga2::{self, FitnessEval, Mate, MultiObjective3};
use graph_annealing::repr::edge_ops_genome::{EdgeOpsGenome, Op, Toolbox};
use graph_annealing::helper::draw_graph;
use graph_annealing::goal::Goal;
use simple_parallel::Pool;

use petgraph::{Directed, Graph};
use graph_sgf::{PetgraphReader, Unweighted};

use std::io::BufReader;
use std::fs::File;

fn fitness<N: Clone, E: Clone>(goal: &Goal<N, E>, ind: &EdgeOpsGenome) -> MultiObjective3<f32> {
    let g = ind.to_graph();
    let cc_dist = goal.strongly_connected_components_distance(&g);
    let nmc = goal.neighbor_matching_score(&g);
    let triadic_dist = goal.triadic_distance(g);
    MultiObjective3::from((cc_dist as f32,
                           triadic_dist as f32,
                           nmc /* ind.num_edges() as f32 */))
}

struct MyEval<N, E> {
    goal: Goal<N, E>,
    pool: Pool,
}

impl<N:Clone+Sync,E:Clone+Sync> FitnessEval<EdgeOpsGenome, MultiObjective3<f32>> for MyEval<N,E> {
    fn fitness(&mut self, pop: &[EdgeOpsGenome]) -> Vec<MultiObjective3<f32>> {
        let pool = &mut self.pool;
        let goal = &self.goal;

        crossbeam::scope(|scope| pool.map(scope, pop, |ind| fitness(goal, ind)).collect())
    }
}

#[derive(Debug)]
struct Stat<T: Debug> {
    min: T,
    max: T,
    avg: T,
}

fn main() {
    let ncpus = num_cpus::get();
    println!("Using {} CPUs", ncpus);
    let matches = App::new("nsga2_edge")
                      .arg(Arg::with_name("NGEN")
                               .long("ngen")
                               .help("Number of generations to run")
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
                      .arg(Arg::with_name("PMUT")
                               .long("pmut")
                               .help("Probability for element mutation")
                               .takes_value(true)
                               .required(true))
                      .arg(Arg::with_name("ILEN")
                               .long("ilen")
                               .help("Initial genome length (random range)")
                               .takes_value(true)
                               .required(true))
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

    let PMUT: f32 = FromStr::from_str(matches.value_of("PMUT").unwrap()).unwrap();
    println!("PMUT: {:?}", PMUT);

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

    // read graph
    let graph_file = matches.value_of("GRAPH").unwrap();
    println!("Using graph file: {}", graph_file);
    let graph = {
        let mut f = BufReader::new(File::open(graph_file).unwrap());
        let graph: Graph<Unweighted, Unweighted, Directed> = PetgraphReader::from_sgf(&mut f);
        graph
    };

    // XXX: Parse weighted operation choice from command line
    let mut toolbox = Toolbox::new(Probability::new(PMUT),
                                   &[(Op::Dup, 1),
                                     (Op::Split, 1),
                                     (Op::Loop, 1),
                                     (Op::Merge, 1),
                                     (Op::Next, 1),
                                     (Op::Parent, 1),
                                     (Op::Reverse, 1)]);

    let mut evaluator = MyEval {
        goal: Goal::new(graph),
        pool: Pool::new(ncpus),
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

    // evaluate fitness
    let fitness: Vec<_> = evaluator.fitness(&initial_population[..]);
    assert!(fitness.len() == initial_population.len());

    let mut pop = initial_population;
    let mut fit = fitness;

    for i in 0..NGEN {
        println!("Iteration: {}", i);
        let (new_pop, new_fit) = nsga2::iterate(&mut rng,
                                                pop,
                                                fit,
                                                &mut evaluator,
                                                MU,
                                                LAMBDA,
                                                K,
                                                &mut toolbox);
        pop = new_pop;
        fit = new_fit;

        let stats: Vec<Stat<f32>> = [0, 1, 2]
                                        .iter()
                                        .map(|&i| {
                                            let min = fit.iter().fold(f32::INFINITY, |acc, f| {
                                                let x = f.objectives[i];
                                                if x < acc {
                                                    x
                                                } else {
                                                    acc
                                                }
                                            });
                                            let max = fit.iter().fold(-f32::INFINITY, |acc, f| {
                                                let x = f.objectives[i];
                                                if x > acc {
                                                    x
                                                } else {
                                                    acc
                                                }
                                            });
                                            let sum = fit.iter()
                                                         .fold(0.0, |acc, f| acc + f.objectives[i]);
                                            Stat {
                                                min: min,
                                                max: max,
                                                avg: sum / fit.len() as f32,
                                            }
                                        })
                                        .collect();
        println!("stats: {:?}", stats);
        let mut found_optimum = false;
        for f in fit.iter() {
            if f.objectives[0] < 1.0 && f.objectives[1] < 3.0 && f.objectives[2] < 0.01 {
                found_optimum = true;
                break;
            }
        }
        if found_optimum {
            println!("Found premature optimum in Iteration {}", i);
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

            if fit[rd.idx].objectives[0] < 1.0 {
                draw_graph(&pop[rd.idx].to_graph(),
                           // XXX: name
                           &format!("nsga2edgeops_g{}_f{}_i{}.svg",
                                    NGEN,
                                    fit[rd.idx].objectives[1] as usize,
                                    j));
                j += 1;
            }
        }
    }
}
