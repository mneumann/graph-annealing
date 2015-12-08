extern crate rand;
extern crate graph_annealing;
extern crate clap;
extern crate pcg;
extern crate sexp;

use sexp::{Sexp, atom_f, atom_i, atom_s, list};
use std::str::FromStr;
use clap::{App, Arg};
use rand::{Rng, SeedableRng};
use rand::distributions::{IndependentSample, Range};
use rand::os::OsRng;
use pcg::PcgRng;
use graph_annealing::repr::edge_ops_genome::{EdgeOpsGenome, parse_weighted_op_list, random_genome,
                                             to_weighted_vec};
use graph_annealing::owned_weighted_choice::OwnedWeightedChoice;
use std::io::Write;
use std::fs::File;
use std::string::ToString;

#[allow(non_snake_case)]
#[allow(unused_must_use)]
fn main() {
    let matches = App::new("edgeop_init")
                      .arg(Arg::with_name("OPS")
                               .long("ops")
                               .help("Edge operation and weight specification, e.g. \
                                      Dup:1,Split:3,Parent:2")
                               .takes_value(true)
                               .required(true))
                      .arg(Arg::with_name("MU")
                               .long("mu")
                               .help("Population size to create")
                               .takes_value(true)
                               .required(true))
                      .arg(Arg::with_name("SEED")
                               .long("seed")
                               .help("Seed value for Rng")
                               .takes_value(true))
                      .arg(Arg::with_name("ILEN")
                               .long("ilen")
                               .help("Initial genome length (random range)")
                               .takes_value(true)
                               .required(true))
                      .arg(Arg::with_name("OUT")
                               .long("out")
                               .help("Write population to file")
                               .takes_value(true)
                               .required(false))
                      .get_matches();

    let mut wr: Box<Write> = match matches.value_of("OUT") {
        Some(out) => {
            Box::new(File::create(out).unwrap())
        }
        None => {
            Box::new(std::io::stdout())
        }
    };

    writeln!(&mut wr, "(POPULATION");
    writeln!(&mut wr, "  (PARAMS");
     
    // size of population
    let MU: usize = FromStr::from_str(matches.value_of("MU").unwrap()).unwrap();
    writeln!(&mut wr, "    (MU {})", MU);

    let seed: Vec<u64>;
    if let Some(seed_str) = matches.value_of("SEED") {
        seed = seed_str.split(",").map(|s| FromStr::from_str(s).unwrap()).collect();
    } else {
        writeln!(&mut std::io::stderr(), "Use OsRng to generate seed..");
        let mut rng = OsRng::new().unwrap();
        seed = (0..2).map(|_| rng.next_u64()).collect();
    }
    writeln!(&mut wr,
             "    (SEED {})",
             Sexp::List(seed.iter().map(|&s| atom_i(s as i64)).collect()));

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
    writeln!(&mut wr,
             "    (ILEN {})",
             list(&[atom_i(ilen_from as i64), atom_i(ilen_to as i64)]));

    // Parse weighted operation choice from command line
    let ops = parse_weighted_op_list(matches.value_of("OPS").unwrap()).unwrap();
    let v = Sexp::List(ops.iter()
                          .map(|&(ref op, f)| {
                              list(&[atom_s(&ToString::to_string(op)), atom_f(f as f64)])
                          })
                          .collect());
    writeln!(&mut wr, "    (OPS {})", v);

    let w_ops = to_weighted_vec(&ops);
    assert!(w_ops.len() > 0);

    let owned_w_ops = OwnedWeightedChoice::new(w_ops);

    assert!(seed.len() == 2);
    let mut rng: PcgRng = SeedableRng::from_seed([seed[0], seed[1]]);

    
    writeln!(&mut wr, "  )");

    // we use the toolbox only for creating a new genome.
    // let mut toolbox = Toolbox::new(w_ops, vec![], vec![], Probability::new(0.0));

    // create initial random population
    let initial_population: Vec<EdgeOpsGenome> = (0..MU)
                                                     .map(|_| {
                                                         let len = if ilen_from == ilen_to {
                                                             ilen_from
                                                         } else {
                                                             Range::new(ilen_from, ilen_to)
                                                                 .ind_sample(&mut rng)
                                                         };
                                                         random_genome(&mut rng, len, &owned_w_ops)
                                                     })
                                                     .collect();


    // output initial population
    writeln!(&mut wr, "  (GENOMES");
    for ind in initial_population.iter() {
        writeln!(&mut wr, "    {}", ind.to_sexp());
    }
    writeln!(&mut wr, "  )");

    // evaluate fitness
    // let fitness: Vec<_> = evaluator.fitness(&initial_population[..]);
    // assert!(fitness.len() == initial_population.len());

    writeln!(&mut wr, ")");
}
