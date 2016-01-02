#![feature(zero_one)]

extern crate rand;
extern crate evo;
extern crate petgraph;
#[macro_use]
extern crate graph_annealing;
//extern crate crossbeam;
//extern crate simple_parallel;
//extern crate num_cpus;
extern crate pcg;
extern crate triadic_census;
extern crate time;
extern crate lindenmayer_system;
extern crate graph_edge_evolution;
extern crate asexp;
extern crate expression;
extern crate expression_num;
extern crate matplotlib;
extern crate rayon;

#[path="genome/genome_edgeop_lsys.rs"]
pub mod genome;

use std::str::FromStr;
use rand::{Rng, SeedableRng};
use rand::os::OsRng;
use pcg::PcgRng;
use evo::Probability;
use evo::nsga2::{self, FitnessEval};
use genome::{Genome, Toolbox};
use graph_annealing::helper::{draw_graph, to_weighted_vec};
use graph_annealing::goal::{FitnessFunction, Goal};
use graph_annealing::goal;
pub use graph_annealing::UniformDistribution;
use graph_annealing::stat::Stat;
//use simple_parallel::Pool;
use petgraph::{Directed, EdgeDirection, Graph};
use triadic_census::OptDenseDigraph;
use std::fs::File;
use genome::{RuleMutOp, RuleProductionMutOp, VarOp};
use genome::edgeop::{EdgeOp, edgeops_to_graph};
use genome::expr_op::{FlatExprOp, RecursiveExprOp};
use std::io::Read;
use asexp::Sexp;
use asexp::sexp::prettyprint;
use std::env;
use std::collections::BTreeMap;
use petgraph::graph::NodeIndex;
use std::fmt::Debug;
use matplotlib::{Env, Plot};

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

fn read_graph<R, NodeWeightFn, EdgeWeightFn, NW, EW>
    (mut rd: R,
     mut node_weight_fn: NodeWeightFn,
     mut edge_weight_fn: EdgeWeightFn)
     -> Result<Graph<NW, EW, Directed>, &'static str>
    where R: Read,
          NodeWeightFn: FnMut(Option<&Sexp>) -> Option<NW>,
          EdgeWeightFn: FnMut(Option<&Sexp>) -> Option<EW>
{
    let mut s = String::new();
    match rd.read_to_string(&mut s) {
        Err(_) => return Err("failed to read file"),
        Ok(_) => {}
    }
    let sexp = match Sexp::parse_toplevel(&s) {
        Err(()) => return Err("failed to parse"),
        Ok(sexp) => sexp,
    };

    println!("{}", sexp);

    let mut map = match sexp.into_map() {
        Err(s) => return Err(s),
        Ok(m) => m,
    };

    println!("{:?}", map);

    match map["version"].get_uint() {
        Some(1) => {}
        _ => return Err("invalid version. Expect 1"),
    }

    let mut nodes;
    if let Some(Sexp::Array(v)) = map.remove("nodes") {
        nodes = Vec::new();
        for entry in v {
            nodes.push(entry.into_map().unwrap());
        }
    } else {
        return Err("no nodes given or invalid");
    }

    println!("{:?}", nodes);

    let mut graph = Graph::new();

    // maps nodes as defined in the graph file to node-ids as used in the graph
    let mut node_map: BTreeMap<u64, NodeIndex> = BTreeMap::new();

    // iterate once to add all nodes
    for node_info in nodes.iter() {
        // XXX: allow other id types
        if let Some(id) = node_info.get("id").and_then(|i| i.get_uint()) {
            println!("node-id: {}", id);
            let weight = match node_weight_fn(node_info.get("weight")) {
                Some(w) => w,
                None => {
                    return Err("invalid node weight");
                }
            };
            let idx = graph.add_node(weight);
            println!("node-idx: {:?}", idx);

            if let Some(_) = node_map.insert(id, idx) {
                return Err("duplicate node-id");
            }
        } else {
            return Err("non-existing or invalid non-integer node key");
        }
    }

    println!("node_map: {:?}", node_map);

    // iterate again, to add all edges
    for mut node_info in nodes.into_iter() {
        // XXX: allow other id types
        let id = node_info.get("id").and_then(|i| i.get_uint()).unwrap();
        println!("node-id: {}", id);

        // XXX: set node weight.

        let src_idx = node_map[&id];
        println!("src-node-idx: {:?}", src_idx);

        if let Some(node_weight) = node_info.remove("weight") {
            println!("node_weight: {}", node_weight);
        }

        if let Some(edges) = node_info.remove("edges") {
            match edges {
                Sexp::Array(edge_list) => {
                    for edge_def in edge_list.iter() {
                        if let Some(target_id) = edge_def.get_uint() {
                            let dst_idx = node_map[&target_id];
                            println!("dst-node-idx: {:?}", dst_idx);

                            let weight = match edge_weight_fn(None) {
                                Some(w) => w,
                                None => {
                                    return Err("invalid edge weight");
                                }
                            };

                            let _ = graph.add_edge(src_idx, dst_idx, weight);
                        } else {
                            let mut valid = false;

                            if let &Sexp::Tuple(ref list) = edge_def {
                                // (node-id weight) tuple
                                if list.len() == 2 {
                                    if let Some(target_id) = list[0].get_uint() {
                                        let dst_idx = node_map[&target_id];
                                        println!("dst-node-idx: {:?}", dst_idx);

                                        let weight = match edge_weight_fn(list.get(1)) {
                                            Some(w) => w,
                                            None => {
                                                return Err("invalid edge weight");
                                            }
                                        };

                                        let _ = graph.add_edge(src_idx, dst_idx, weight);
                                        valid = true;
                                    }
                                }
                            }
                            if !valid {
                                return Err("invalid edge node id");
                            }
                        }
                    }
                }
                _ => {
                    return Err("Invalid edge list");
                }
            }
        }
    }

    Ok(graph)
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
    let graph = read_graph(File::open(graph_file).unwrap(),
                           convert_weight,
                           convert_weight)
                    .unwrap();
    println!("graph: {:?}", graph);

    let genome_map = map.get("genome").unwrap().clone().into_map().unwrap();

    Config {
        ngen: ngen,
        mu: mu,
        lambda: lambda,
        k: k,
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

fn pp_sexp(s: &Sexp) {
    let mut st = String::new();
    let _ = prettyprint(&s, &mut st, 0, false).unwrap();
    println!("{}", st);
}

fn main() {
    let env = Env::new();
    let plot = Plot::new(&env);
    plot.interactive();
    plot.show();

    //let ncpus = num_cpus::get();
    //println!("Using {} CPUs", ncpus);

    let mut s = String::new();
    let configfile = env::args().nth(1).unwrap();
    let _ = File::open(configfile).unwrap().read_to_string(&mut s).unwrap();
    let expr = asexp::Sexp::parse_toplevel(&s).unwrap();
    let config = parse_config(expr);

    println!("{:#?}", config);

    let num_objectives = config.objectives.len();

    let mut toolbox = Toolbox::new(Goal::new(OptDenseDigraph::from(config.graph.clone())),
                                   //Pool::new(ncpus),
                                   config.objectives
                                         .iter()
                                         .map(|o| o.fitness_function.clone())
                                         .collect(),
                                   config.genome.max_iter, // iterations
                                   config.genome.rules, // num_rules
                                   config.genome.initial_len, // initial rule length
                                   config.genome.symbol_arity, // we use 2-ary symbols
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
        // XXX Would be nice if nsga2::iterate() would return us the
        // pareto front number. We could use it to color the output.
        let (new_pop, new_fit) = nsga2::iterate(&mut rng,
                                                pop,
                                                fit,
                                                config.mu,
                                                config.lambda,
                                                config.k,
                                                num_objectives,
                                                &mut toolbox);
        let duration = time::precise_time_ns() - before;
        pop = new_pop;
        fit = new_fit;
        assert!(fit.len() > 0);

        let duration_ms = (duration as f32) / 1_000_000.0;

        // XXX: Assume we have at least two objectives
        let mut x = Vec::new();
        let mut y = Vec::new();
        for f in fit.iter() {
            x.push(f.objectives[0]);
            y.push(f.objectives[1]);
        }

        plot.clf();
        plot.title(&format!("Iteration: {}", iteration));
        plot.grid(true);
        plot.scatter(&x, &y);
        plot.draw();

        let mut num_optima = 0;
        for f in fit.iter() {
            if config.objectives
                     .iter()
                     .enumerate()
                     .all(|(i, obj)| f.objectives[i] <= obj.threshold) {
                num_optima += 1;
            }
        }

        // calculate a min/max/avg value for each objective.
        let stats: Vec<Stat<f32>> = (0..num_objectives)
                                        .into_iter()
                                        .map(|i| {
                                            Stat::from_iter(fit.iter().map(|o| o.objectives[i]))
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
            break;
        }
    }
    println!("===========================================================");

    // finally evaluate rank and crowding distance (using select()).
    let rank_dist = nsga2::select(&fit[..], config.mu, num_objectives);
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

            // output as sexp
            println!("Found graph");
            let sexp = graph_to_sexp(g.ref_graph(),
                                     |&nw| Some(Sexp::from(nw)),
                                     |&ew| Some(Sexp::from(ew)));
            pp_sexp(&sexp);
            println!("Target graph");
            let sexp = graph_to_sexp(&goal::normalize_graph(&config.graph),
                                     |nw| Some(Sexp::from(nw.get())),
                                     |ew| Some(Sexp::from(ew.get())));
            pp_sexp(&sexp);

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

    println!("{:#?}", config);
}
