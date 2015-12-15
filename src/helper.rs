use petgraph::{Directed, Graph};
use petgraph::graph::NodeIndex;
use graph_layout::{P2d, fruchterman_reingold};
use graph_layout::svg_writer::{SvgCanvas, SvgWriter};
use std::fs::File;
use rand::isaac::Isaac64Rng;
use rand::{Closed01, Rng};
use rand::distributions::Weighted;
use std::str::FromStr;

pub fn draw_graph<N, E>(g: &Graph<N, E, Directed>, filename: &str) {
    let l = Some(g.edge_count() as f32 / (g.node_count() as f32 * g.node_count() as f32));
    let mut rng = Isaac64Rng::new_unseeded();
    let mut node_positions: Vec<P2d> = Vec::with_capacity(g.node_count());
    let mut node_neighbors: Vec<Vec<usize>> = Vec::with_capacity(g.node_count());

    for ni in 0..g.node_count() {
        node_positions.push(P2d(rng.gen::<Closed01<f32>>().0, rng.gen::<Closed01<f32>>().0));
        let mut neighbors: Vec<usize> = Vec::new();
        for (ei, _) in g.edges(NodeIndex::new(ni)) {
            neighbors.push(ei.index());
        }
        node_neighbors.push(neighbors);
    }

    fruchterman_reingold::layout_typical_2d(l, &mut node_positions, &node_neighbors);

    let mut file = File::create(filename).unwrap();
    let svg_wr = SvgWriter::new(SvgCanvas::default_for_unit_layout(), &mut file);
    svg_wr.draw_graph(&node_positions, &node_neighbors, true);
}

pub fn line_graph(n: u32) -> Graph<(), (), Directed> {
    let mut g: Graph<(), (), Directed> = Graph::new();

    let mut prev = g.add_node(());
    for _ in 0..n - 1 {
        let cur = g.add_node(());
        g.add_edge(prev, cur, ());
        prev = cur;
    }
    return g;
}

pub fn parse_weighted_op_list<T>(s: &str) -> Result<Vec<(T, u32)>, String>
    where T: FromStr<Err = String>
{
    let mut v = Vec::new();
    for opstr in s.split(",") {
        if opstr.is_empty() {
            continue;
        }
        let mut i = opstr.splitn(2, ":");
        if let Some(ops) = i.next() {
            match T::from_str(ops) {
                Ok(op) => {
                    let ws = i.next().unwrap_or("1");
                    if let Ok(weight) = u32::from_str(ws) {
                        v.push((op, weight));
                    } else {
                        return Err(format!("Invalid weight: {}", ws));
                    }
                }
                Err(s) => {
                    return Err(s);
                }
            }
        } else {
            return Err("missing op".to_string());
        }
    }
    return Ok(v);
}

#[test]
fn test_parse_weighted_op_choice_list() {
    #[derive(Copy, Clone, PartialEq, Eq, Debug)]
    enum Op {
        Dup,
        Split,
    }
    impl FromStr for Op {
        type Err = String;
        fn from_str(s: &str) -> Result<Self, Self::Err> {
            match s {
                "Dup" => Ok(Op::Dup),
                "Split" => Ok(Op::Split),
            }
        }
    }

    fn parse_weighted_op_choice_list(s: &str) -> Result<Vec<(Op, u32)>, String> {
        parse_weighted_op_list(s)
    }
    assert_eq!(Ok(vec![]), parse_weighted_op_choice_list(""));
    assert_eq!(Ok(vec![(Op::Dup, 1)]), parse_weighted_op_choice_list("Dup"));
    assert_eq!(Ok(vec![(Op::Dup, 1)]),
               parse_weighted_op_choice_list("Dup:1"));
    assert_eq!(Ok(vec![(Op::Dup, 2)]),
               parse_weighted_op_choice_list("Dup:2"));
    assert_eq!(Ok(vec![(Op::Dup, 2), (Op::Split, 1)]),
               parse_weighted_op_choice_list("Dup:2,Split"));
    assert_eq!(Err("Invalid weight: ".to_string()),
               parse_weighted_op_choice_list("Dup:2,Split:"));
    assert_eq!(Err("Invalid weight: a".to_string()),
               parse_weighted_op_choice_list("Dup:2,Split:a"));
    assert_eq!(Err("Invalid opcode: dup".to_string()),
               parse_weighted_op_choice_list("dup:2,Split:a"));
}
pub fn to_weighted_vec<T: Clone>(ops: &[(T, u32)]) -> Vec<Weighted<T>> {
    let mut w = Vec::with_capacity(ops.len());
    for &(ref op, weight) in ops {
        if weight > 0 {
            // an operation with weight=0 cannot be selected
            w.push(Weighted {
                weight: weight,
                item: op.clone(),
            });
        }
    }
    w
}
