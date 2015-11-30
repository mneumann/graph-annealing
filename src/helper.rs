use petgraph::{Directed, Graph};
use petgraph::graph::NodeIndex;
use graph_layout::{P2d, fruchterman_reingold};
use graph_layout::svg_writer::{SvgCanvas, SvgWriter};
use std::fs::File;
use rand::isaac::Isaac64Rng;
use rand::{Closed01, Rng};

pub fn draw_graph<N,E>(g: &Graph<N, E, Directed>, filename: &str) {
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
