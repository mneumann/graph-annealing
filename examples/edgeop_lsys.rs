extern crate rand;
extern crate evo;
extern crate petgraph;
#[macro_use]
extern crate graph_annealing;
extern crate pcg;
extern crate triadic_census;
extern crate lindenmayer_system;
extern crate graph_edge_evolution;
extern crate asexp;
extern crate expression;
extern crate expression_num;
extern crate expression_closed01;
extern crate matplotlib;
extern crate closed01;
extern crate graph_io_gml;
extern crate nsga2;

#[path="genome/genome_edgeop_lsys.rs"]
pub mod genome;

use std::str::FromStr;
use rand::{Rng, SeedableRng};
use rand::os::OsRng;
use pcg::PcgRng;
use evo::Probability;
use nsga2::{Driver, DriverConfig};
use genome::{Genome, Toolbox};
use graph_annealing::helper::to_weighted_vec;
use graph_annealing::goal::{FitnessFunction, Goal};
use graph_annealing::graph;
pub use graph_annealing::UniformDistribution;
use graph_annealing::stat::Stat;
use petgraph::{Directed, EdgeDirection, Graph};
use triadic_census::OptDenseDigraph;
use std::fs::File;
use genome::{RuleMutOp, RuleProductionMutOp, VarOp};
use genome::edgeop::{EdgeOp, edgeops_to_graph};
use genome::expr_op::{FlatExprOp, RecursiveExprOp, EXPR_NAME};
use std::io::Read;
use asexp::Sexp;
use asexp::sexp::pp;
use std::env;
use std::collections::BTreeMap;
use std::fmt::Debug;
use matplotlib::{Env, Plot};

struct ReseedRecorder {
    reseeds: Vec<(u64, u64)>
}

/*
impl Reseeder<pcg::RcgRng> for ReseedRecorder {
    fn reseed(&mut self, rng: &mut pcg::RcgRng) {
        let mut r = rand::thread_rng();
        let s1 = r.next_u64(); 
        let s2 = r.next_u64();
        self.reseeds.push((s1, s2));
        rng.reseed([s1, s2]);
    }
}
*/

const MAX_OBJECTIVES: usize = 3;

fn graph_to_sexp<N, E, F, G>(g: &Graph<N, E, Directed>,
                             node_weight_map: F,
                             edge_weight_map: G)
    -> Sexp
    where F: Fn(&N) -> Option<Sexp>,
          G: Fn(&E) -> Option<Sexp>
{
    let mut nodes = Vec::new();
    for node_idx in g.node_indices() {
        let edges: Vec<_> = g.edges_directed(node_idx, EdgeDirection::Outgoing)
            .map(|(target_node, edge_weight)| {
                match edge_weight_map(edge_weight) {
                    Some(w) => Sexp::from((target_node.index(), w)),
                    None => Sexp::from(target_node.index()),
                }
            })
        .collect();

        let mut def = vec![
            (Sexp::from("id"), Sexp::from(node_idx.index())),
            (Sexp::from("edges"), Sexp::Array(edges)),
        ];

        match node_weight_map(&g[node_idx]) {
            Some(w) => def.push((Sexp::from("weight"), w)),
            None => {}
        }

        nodes.push(Sexp::Map(def));
    }

    Sexp::Map(vec![
              (Sexp::from("version"), Sexp::from(1usize)),
              (Sexp::from("nodes"), Sexp::Array(nodes)),
    ])
}

#[derive(Debug)]
struct ConfigGenome {
    max_iter: usize,
    rules: usize,
    initial_len: usize,
    symbol_arity: usize,
    num_params: usize,
    prob_terminal: Probability,
}

#[derive(Debug)]
struct Config {
    ngen: usize,
    mu: usize,
    lambda: usize,
    k: usize,
    seed: Vec<u64>,
    objectives: Vec<Objective>,
    graph: Graph<f32, f32, Directed>,
    edge_ops: Vec<(EdgeOp, u32)>,
    var_ops: Vec<(VarOp, u32)>,
    rule_mut_ops: Vec<(RuleMutOp, u32)>,
    rule_prod_ops: Vec<(RuleProductionMutOp, u32)>,
    flat_expr_op: Vec<(FlatExprOp, u32)>,
    recursive_expr_op: Vec<(RecursiveExprOp, u32)>,
    genome: ConfigGenome,
    plot: bool,
    weight: f64,
}

#[derive(Debug)]
struct Objective {
    fitness_function: FitnessFunction,
    threshold: f32,
}

fn parse_ops<T, I>(map: &BTreeMap<String, Sexp>, key: &str) -> Vec<(T, u32)>
where T: FromStr<Err = I> + UniformDistribution,
      I: Debug
{
    if let Some(&Sexp::Map(ref list)) = map.get(key) {
        let mut ops: Vec<(T, u32)> = Vec::new();
        for &(ref k, ref v) in list.iter() {
            ops.push((T::from_str(k.get_str().unwrap()).unwrap(),
            v.get_uint().unwrap() as u32));
        }
        ops
    } else {
        T::uniform_distribution()
    }
}

fn convert_weight(w: Option<&Sexp>) -> Option<f32> {
    match w {
        Some(s) => s.get_float().map(|f| f as f32),
        None => {
            // use a default
            Some(0.0)
        }
    }
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

    let plot: bool = map.get("plot").map(|v| v.get_str() == Some("true")).unwrap_or(false);
    let weight: f64 = map.get("weight").and_then(|v| v.get_float()).unwrap_or(1.0);

    let seed: Vec<u64>;
    if let Some(seed_expr) = map.get("seed") {
        seed = seed_expr.get_uint_vec().unwrap();
    } else {
        println!("Use OsRng to generate seed..");
        let mut rng = OsRng::new().unwrap();
        seed = (0..2).map(|_| rng.next_u64()).collect();
    }

    // Parse objectives and thresholds
    let mut objectives: Vec<Objective> = Vec::new();
    if let Some(&Sexp::Map(ref list)) = map.get("objectives") {
        for &(ref k, ref v) in list.iter() {
            objectives.push(Objective {
                fitness_function: FitnessFunction::from_str(k.get_str().unwrap()).unwrap(),
                threshold: v.get_float().unwrap() as f32,
            });
        }
    } else {
        panic!("Map expected");
    }

    if objectives.len() > MAX_OBJECTIVES {
        panic!("Max {} objectives allowed", MAX_OBJECTIVES);
    }

    // read graph
    let graph_file = map.get("graph").unwrap().get_str().unwrap();
    println!("Using graph file: {}", graph_file);

    let graph_s = {
        let mut graph_file = File::open(graph_file).unwrap();
        let mut graph_s = String::new();
        let _ = graph_file.read_to_string(&mut graph_s).unwrap();
        graph_s
    };

    let graph = graph_io_gml::parse_gml(&graph_s,
                                        &convert_weight,
                                        &convert_weight)
        .unwrap();
    println!("graph: {:?}", graph);

    let graph = graph::normalize_graph(&graph);

    let genome_map = map.get("genome").unwrap().clone().into_map().unwrap();

    Config {
        ngen: ngen,
        mu: mu,
        lambda: lambda,
        k: k,
        plot: plot,
        weight: weight,
        seed: seed,
        objectives: objectives,
        graph: graph,
        edge_ops: parse_ops(&map, "edge_ops"),
        var_ops: parse_ops(&map, "var_ops"),
        rule_mut_ops: parse_ops(&map, "rule_mut_ops"),
        rule_prod_ops: parse_ops(&map, "rule_prod_mut_ops"),
        flat_expr_op: parse_ops(&map, "flat_expr_ops"),
        recursive_expr_op: parse_ops(&map, "recursive_expr_ops"),
        genome: ConfigGenome {
            rules: genome_map.get("rules").and_then(|v| v.get_uint()).unwrap() as usize,
            symbol_arity: genome_map.get("symbol_arity").and_then(|v| v.get_uint()).unwrap() as usize,
            num_params: genome_map.get("num_params").and_then(|v| v.get_uint()).unwrap() as usize,
            initial_len: genome_map.get("initial_len").and_then(|v| v.get_uint()).unwrap() as usize,
            max_iter: genome_map.get("max_iter").and_then(|v| v.get_uint()).unwrap() as usize,
            prob_terminal: Probability::new(genome_map.get("prob_terminal").and_then(|v| v.get_float()).unwrap() as f32),
        },
    }
}

fn main() {
    println!("Using expr system: {}", EXPR_NAME);
    let env = Env::new();
    let plot = Plot::new(&env);

    let mut s = String::new();
    let configfile = env::args().nth(1).unwrap();
    let _ = File::open(configfile).unwrap().read_to_string(&mut s).unwrap();
    let expr = asexp::Sexp::parse_toplevel(&s).unwrap();
    let config = parse_config(expr);

    println!("{:#?}", config);

    if config.plot {
        plot.interactive();
        plot.show();
    }


    let num_objectives = config.objectives.len();

    let driver_config = DriverConfig {
        mu: config.mu,
        lambda: config.lambda,
        k: config.k,
        ngen: config.ngen,
        num_objectives: num_objectives
    };

    let toolbox = Toolbox::new(Goal::new(OptDenseDigraph::from(config.graph.clone())),
    config.objectives
    .iter()
    .map(|o| o.threshold)
    .collect(),
    config.objectives
    .iter()
    .map(|o| o.fitness_function.clone())
    .collect(),
    config.genome.max_iter, // iterations
    config.genome.rules, // num_rules
    config.genome.initial_len, // initial rule length
    config.genome.symbol_arity, // we use 1-ary symbols
    config.genome.num_params,
    config.genome.prob_terminal,
    to_weighted_vec(&config.edge_ops),

    to_weighted_vec(&config.flat_expr_op),
    to_weighted_vec(&config.recursive_expr_op),

    to_weighted_vec(&config.var_ops),
    to_weighted_vec(&config.rule_mut_ops),
    to_weighted_vec(&config.rule_prod_ops));

    assert!(config.seed.len() == 2);
    let mut rng: PcgRng = SeedableRng::from_seed([config.seed[0], config.seed[1]]);
    //let mut rng = rand::thread_rng();

    let selected_population = toolbox.run(&mut rng, &driver_config, config.weight, &|iteration, duration, num_optima, population| {
        let duration_ms = (duration as f32) / 1_000_000.0;
        print!("# {:>6}", iteration);

        let fitness_values = population.fitness_to_vec();

        // XXX: Assume we have at least two objectives
        let mut x = Vec::new();
        let mut y = Vec::new();
        for f in fitness_values.iter() {
            x.push(f.objectives[0]);
            y.push(f.objectives[1]);
        }

        if config.plot {
            plot.clf();
            plot.title(&format!("Iteration: {}", iteration));
            plot.grid(true);
            plot.scatter(&x, &y);
            plot.draw();
        }

        // calculate a min/max/avg value for each objective.
        let stats: Vec<Stat<f32>> = (0..num_objectives)
            .into_iter()
            .map(|i| {
                Stat::from_iter(fitness_values.iter().map(|o| o.objectives[i]))
                    .unwrap()
            })
        .collect();

        for stat in stats.iter() {
            print!(" | ");
            print!("{:>8.2}", stat.min);
            print!("{:>9.2}", stat.avg);
            print!("{:>10.2}", stat.max);
        }

        print!(" | {:>5} | {:>8.0} ms", num_optima, duration_ms);
        println!("");

        if num_optima > 0 {
            println!("Found premature optimum in Iteration {}", iteration);
        }

    });

    println!("===========================================================");

    let mut best_solutions: Vec<(Genome, _)> = Vec::new();

    selected_population.all_of_rank(0, &mut |ind, fit| {
        if fit.objectives[0] < 0.1 && fit.objectives[1] < 0.1 {
            best_solutions.push((ind.clone(), fit.clone()));
        }
    });

    println!("Target graph");
    let sexp = graph_to_sexp(&graph::normalize_graph_closed01(&config.graph),
    |nw| Some(Sexp::from(nw.get())),
    |ew| Some(Sexp::from(ew.get())));
    println!("{}", pp(&sexp));

    let mut solutions: Vec<Sexp> = Vec::new();

    for (_i, &(ref ind, ref fitness)) in best_solutions.iter().enumerate() {
        let genome: Sexp = ind.into();

        let edge_ops = ind.to_edge_ops(&toolbox.axiom_args, toolbox.iterations);
        let g = edgeops_to_graph(&edge_ops);

        // output as sexp
        let graph_sexp = graph_to_sexp(g.ref_graph(),
        |&nw| Some(Sexp::from(nw)),
        |&ew| Some(Sexp::from(ew)));

        solutions.push(Sexp::Map(
                vec![
                (Sexp::from("fitness"), Sexp::from((fitness.objectives[0], fitness.objectives[1], fitness.objectives[2]))),
                (Sexp::from("genome"), genome),
                (Sexp::from("graph"), graph_sexp),
                ]
                ));

        /*
           draw_graph(g.ref_graph(),
        // XXX: name
        &format!("edgeop_lsys_g{}_f{}_i{}.svg",
        config.ngen,
        fitness.objectives[1] as usize,
        i));
        */
    }

    println!("{}", pp(&Sexp::from(("solutions", Sexp::Array(solutions)))));

    //println!("])");

    println!("{:#?}", config);
}
