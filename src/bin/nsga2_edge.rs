extern crate rand;
extern crate evo;
extern crate petgraph;
extern crate graph_annealing;
extern crate clap;
extern crate crossbeam;
extern crate simple_parallel;
extern crate num_cpus;

use std::str::FromStr;
use clap::{App, Arg};
use rand::{Rng, SeedableRng};
use rand::isaac::Isaac64Rng;

use evo::Probability;
use evo::nsga2::{self, FitnessEval, Mate, MultiObjective3};
use graph_annealing::repr::edge_list_genome::EdgeListGenome;
use graph_annealing::helper::{draw_graph, line_graph};
use graph_annealing::goal::Goal;
use simple_parallel::Pool;

struct Mating {
    prob_mutate_elem: Probability,
}

impl Mate<EdgeListGenome> for Mating {
    fn mate<R:Rng>(&mut self, rng: &mut R, p1: &EdgeListGenome, p2: &EdgeListGenome) -> EdgeListGenome {
        let mut child = EdgeListGenome::mate(rng, p1, p2);
        child.mutate(rng, self.prob_mutate_elem);
        child
    }
}

fn fitness(goal: &Goal, ind: &EdgeListGenome) -> MultiObjective3<f32> {
    let g = ind.to_graph();
    let cc_dist = goal.strongly_connected_components_distance(&g);
    let nmc = goal.neighbor_matching_score(&g);
    let triadic_dist = goal.triadic_distance(g);
    MultiObjective3::from((cc_dist as f32,
                           triadic_dist as f32,
                           nmc /* ind.num_edges() as f32 */))
}

struct MyEval {
    goal: Goal,
    pool: Pool,
}

impl FitnessEval<EdgeListGenome, MultiObjective3<f32>> for MyEval {
    fn fitness(&mut self, pop: &[EdgeListGenome]) -> Vec<MultiObjective3<f32>> {
        let pool = &mut self.pool;
        let goal = &self.goal;

        crossbeam::scope(|scope| pool.map(scope, pop, |ind| fitness(goal, ind)).collect())
    }
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
                      .get_matches();

    const N: usize = 40;    // number of nodes
    const MU: usize = 1000; // size of population
    const LAMBDA: usize = 500; // size of offspring population

    let NGEN: usize = FromStr::from_str(matches.value_of("NGEN").unwrap()).unwrap();
    println!("NGEN: {}", NGEN);

    let mut evaluator = MyEval {
        goal: Goal::new(line_graph(N as u32)),
        pool: Pool::new(ncpus),
    };

    let seed: &[_] = &[45234341, 12343423, 123239];
    let mut rng: Isaac64Rng = SeedableRng::from_seed(seed);

    // create initial random population
    let initial_population: Vec<EdgeListGenome> = (0..MU)
                                                      .map(|_| EdgeListGenome::random(&mut rng, N))
                                                      .collect();

    // evaluate fitness
    let fitness: Vec<_> = evaluator.fitness(&initial_population[..]);
    assert!(fitness.len() == initial_population.len());

    let mut pop = initial_population;
    let mut fit = fitness;

    let mut mating = Mating {
        prob_mutate_elem: Probability::new(1.0 / N as f32),
    };

    for _ in 0..NGEN {
        let (new_pop, new_fit) = nsga2::iterate(&mut rng,
                                                pop,
                                                fit,
                                                &mut evaluator,
                                                MU,
                                                LAMBDA,
                                                &mut mating);
        pop = new_pop;
        fit = new_fit;
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
                           &format!("line_{}_nsga2edge_n{}_f{}_i{}.svg",
                                    N,
                                    NGEN,
                                    fit[rd.idx].objectives[1] as usize,
                                    j));
                j += 1;
            }
        }
    }
}
