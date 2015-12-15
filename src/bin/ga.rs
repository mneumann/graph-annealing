extern crate rand;
extern crate evo;
extern crate petgraph;
extern crate graph_annealing;

use rand::{Rng, SeedableRng};
use rand::isaac::Isaac64Rng;

use evo::{Evaluator, Individual, MinFitness, OpCrossover1, OpMutate, OpSelect,
          OpSelectRandomIndividual, OpVariation, Probability, ProbabilityValue, RatedPopulation,
          UnratedPopulation, VariationMethod, ea_mu_plus_lambda};
use evo::selection::tournament_selection_fast;
use evo::bit_string::crossover_one_point;
use graph_annealing::repr::adj_genome::AdjGenome;
use graph_annealing::helper::{draw_graph, line_graph};
use graph_annealing::goal::Goal;

struct MyEval<N, E> {
    goal: Goal<N, E>,
}

impl<N: Clone + Sync, E: Clone + Sync> Evaluator<AdjGenome, MinFitness<f64>> for MyEval<N, E> {
    fn fitness(&self, ind: &AdjGenome) -> MinFitness<f64> {
        let g = ind.to_graph();

        let cc = self.goal.connected_components_distance(&g);
        MinFitness((cc + 1) as f64 * self.goal.triadic_distance(g))
    }
}

struct Toolbox {
    rng: Box<Rng>,
    prob_crossover: Probability,
    prob_mutation: Probability,
    prob_bitflip: Probability,
    tournament_size: usize,
}

// operate on Rng trait
impl OpCrossover1<AdjGenome> for Toolbox {
    fn crossover1(&mut self, male: &AdjGenome, female: &AdjGenome) -> AdjGenome {
        debug_assert!(male.matrix_n() == female.matrix_n());
        let rows = male.matrix_n();
        let total = rows * rows;
        assert!(male.bits.len() == total);
        assert!(female.bits.len() == total);
        let point = self.rng.gen_range(1, rows - 1); // point == 0 makes no sense, as
        // well as rows-1.

        // horizontal crossover is easy.

        let (c1, _c2) = crossover_one_point(point * rows, (&male.bits, &female.bits));
        AdjGenome::new(c1, rows)
    }
}

// operate on Rng trait
// XXX: pass probability to mutate function
impl OpMutate<AdjGenome> for Toolbox {
    fn mutate(&mut self, ind: &AdjGenome) -> AdjGenome {
        let mut genome: AdjGenome = ind.clone();
        genome.mutate(&mut self.rng, self.prob_bitflip);
        return genome;
    }
}

impl OpVariation for Toolbox {
    fn variation(&mut self) -> VariationMethod {
        let r = self.rng.gen::<ProbabilityValue>();
        if r.is_probable_with(self.prob_crossover) {
            VariationMethod::Crossover
        } else if r.is_probable_with(self.prob_crossover + self.prob_mutation) {
            VariationMethod::Mutation
        } else {
            VariationMethod::Reproduction
        }
    }
}

// XXX: No need for Fitness
impl<I: Individual, F: PartialOrd> OpSelectRandomIndividual<I, F> for Toolbox {
    fn select_random_individual(&mut self, population: &RatedPopulation<I, F>) -> usize {
        // let (idx, _) = tournament_selection(&mut self.rng, |idx|
        // {population.get_ref(idx).fitness().unwrap()}, population.len(), 1).unwrap();
        self.rng.gen_range(0, population.len())
    }
}

impl<I: Individual, F: PartialOrd + Clone> OpSelect<I, F> for Toolbox {
    fn select(&mut self, population: &RatedPopulation<I, F>, mu: usize) -> RatedPopulation<I, F> {
        let mut pop: RatedPopulation<I, F> = RatedPopulation::with_capacity(mu);
        for _ in 0..mu {
            let choice = tournament_selection_fast(&mut self.rng,
                                                   |i1, i2| population.fitter_than(i1, i2),
                                                   population.len(),
                                                   self.tournament_size);
            pop.add(population.get_individual(choice).clone(),
                    (*population.get_fitness(choice)).clone());
        }
        assert!(pop.len() == mu);
        return pop;
    }
}

fn print_stat(p: &RatedPopulation<AdjGenome, MinFitness<f64>>) {
    let mut fitnesses: Vec<f64> = Vec::new();
    for i in 0..p.len() {
        let f = p.get_fitness(i).0;
        fitnesses.push(f);
    }
    let min = fitnesses.iter().fold(fitnesses[0], |b, &i| {
        if b < i {
            b
        } else {
            i
        }
    });
    let max = fitnesses.iter().fold(fitnesses[0], |b, &i| {
        if b > i {
            b
        } else {
            i
        }
    });
    let sum = fitnesses.iter().fold(0.0, |b, &i| b + i);

    println!("min: {}, max: {}, sum: {}, avg: {}",
             min,
             max,
             sum,
             sum as f32 / fitnesses.len() as f32);
}

fn main() {
    const MATRIX_N: usize = 40;
    const BITS: usize = MATRIX_N * MATRIX_N;
    const MU: usize = 6000;
    const LAMBDA: usize = 60 * MATRIX_N;
    const NGEN: usize = 2000;

    let evaluator = MyEval { goal: Goal::new(line_graph(MATRIX_N as u32)) };

    let seed: &[_] = &[45234341, 12343423, 123239];
    let mut rng: Isaac64Rng = SeedableRng::from_seed(seed);

    let mut initial_population: UnratedPopulation<AdjGenome> = UnratedPopulation::with_capacity(MU);
    for _ in 0..MU {
        initial_population.add(AdjGenome::random(&mut rng, MATRIX_N));
    }

    let rated_population = initial_population.rate(&evaluator);

    let mut toolbox = Toolbox {
        rng: Box::new(rng),
        prob_crossover: Probability::new(0.3), // 0.7, 0.3 -> 20.
        prob_mutation: Probability::new(0.5),
        prob_bitflip: Probability::new(2.0 / BITS as f32),
        tournament_size: 3,
    };

    fn stat(gen: usize, nevals: usize, pop: &RatedPopulation<AdjGenome, MinFitness<f64>>) {
        print!("{:04} {:04}", gen, nevals);
        print_stat(&pop);

        if gen % 100 != 0 && gen != NGEN {
            return;
        }

        let filename = format!("line_{:04}.svg", gen);
        let ind = pop.get_individual(pop.fittest());
        draw_graph(&ind.to_graph(), &filename);
    }

    let _optimum = evo::ea_mu_plus_lambda(&mut toolbox,
                                          &evaluator,
                                          rated_population,
                                          MU,
                                          LAMBDA,
                                          NGEN,
                                          stat,
                                          8,
                                          10);
}
