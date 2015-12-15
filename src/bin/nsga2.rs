extern crate rand;
extern crate evo;
extern crate petgraph;
extern crate graph_annealing;

use rand::{Rng, SeedableRng};
use rand::isaac::Isaac64Rng;

use evo::Probability;
use evo::bit_string::crossover_one_point;
use evo::nsga2::{self, FitnessEval, Mate, MultiObjective2};
use graph_annealing::repr::adj_genome::AdjGenome;
use graph_annealing::helper::{draw_graph, line_graph};
use graph_annealing::goal::Goal;

struct Mating {
    prob_bitflip: Probability,
}

impl Mate<AdjGenome> for Mating {
    fn mate<R: Rng>(&mut self, rng: &mut R, p1: &AdjGenome, p2: &AdjGenome) -> AdjGenome {
        let male = p1;
        let female = p2;

        debug_assert!(male.matrix_n() == female.matrix_n());
        let rows = male.matrix_n();
        let total = rows * rows;
        debug_assert!(male.bits.len() == total);
        debug_assert!(female.bits.len() == total);
        let point = rng.gen_range(1, rows - 1); // point == 0 makes no sense, as
        // well as rows-1.

        // horizontal crossover is easy.

        let (c1, _c2) = crossover_one_point(point * rows, (&male.bits, &female.bits));
        let mut child = AdjGenome::new(c1, rows);
        // if self.rng.gen::<f32>() < 0.5 {
        child.mutate(rng, self.prob_bitflip);
        // }
        child
    }
}

fn fitness<N: Clone, E: Clone>(goal: &Goal<N, E>, ind: &AdjGenome) -> MultiObjective2<f32> {
    let g = ind.to_graph();
    let cc_dist = goal.connected_components_distance(&g);
    let triadic_dist = goal.triadic_distance(g);
    MultiObjective2::from((cc_dist as f32, triadic_dist as f32))
}

struct MyEval<N, E> {
    goal: Goal<N, E>,
}

impl<N: Clone, E: Clone> FitnessEval<AdjGenome, MultiObjective2<f32>> for MyEval<N, E> {
    fn fitness(&mut self, pop: &[AdjGenome]) -> Vec<MultiObjective2<f32>> {
        pop.iter().map(|ind| fitness(&self.goal, ind)).collect()
    }
}

fn main() {
    const MATRIX_N: usize = 20;

    const MU: usize = 1000; // size of population
    const LAMBDA: usize = 500; // size of offspring population
    const NGEN: usize = 10000; // number of generations

    let mut evaluator = MyEval { goal: Goal::new(line_graph(MATRIX_N as u32)) };

    let seed: &[_] = &[45234341, 12343423, 123239];
    let mut rng: Isaac64Rng = SeedableRng::from_seed(seed);

    // create initial random population
    let initial_population: Vec<AdjGenome> = (0..MU)
                                                 .map(|_| AdjGenome::random(&mut rng, MATRIX_N))
                                                 .collect();

    // evaluate fitness
    let fitness: Vec<_> = evaluator.fitness(&initial_population[..]);
    assert!(fitness.len() == initial_population.len());

    let mut pop = initial_population;
    let mut fit = fitness;

    let mut mating = Mating {
        prob_bitflip: Probability::new(1.0 / ((MATRIX_N * MATRIX_N) as f32)),
    };

    for _ in 0..NGEN {
        let (new_pop, new_fit) = nsga2::iterate(&mut rng,
                                                pop,
                                                fit,
                                                &mut evaluator,
                                                MU,
                                                LAMBDA,
                                                2,
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
                           &format!("line_{}x{}_nsga2_n{}_f{}_i{}.svg",
                                    MATRIX_N,
                                    MATRIX_N,
                                    NGEN,
                                    fit[rd.idx].objectives[1] as usize,
                                    j));
                j += 1;
            }
        }
    }
}
