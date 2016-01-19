use petgraph::{Directed, Graph};
use closed01::Closed01;
use std::f32::{INFINITY, NEG_INFINITY};

fn determine_node_value_range(g: &Graph<f32, f32, Directed>) -> (f32, f32) {
    let mut w_min = INFINITY;
    let mut w_max = NEG_INFINITY;
    for i in g.raw_nodes() {
        w_min = w_min.min(i.weight);
        w_max = w_max.max(i.weight);
    }
    (w_min, w_max)
}

fn determine_edge_value_range(g: &Graph<f32, f32, Directed>) -> (f32, f32) {
    let mut w_min = INFINITY;
    let mut w_max = NEG_INFINITY;
    for i in g.raw_edges() {
        w_min = w_min.min(i.weight);
        w_max = w_max.max(i.weight);
    }
    (w_min, w_max)
}

fn normalize_to_closed01(w: f32, range: (f32, f32)) -> Closed01<f32> {
    assert!(range.1 >= range.0);
    let dist = range.1 - range.0;
    if dist == 0.0 {
        Closed01::zero()
    } else {
        Closed01::new((w - range.0) / dist)
    }
}

// Normalize the node and edge weights into the range [0,1]
pub fn normalize_graph_closed01(g: &Graph<f32, f32, Directed>)
                       -> Graph<Closed01<f32>, Closed01<f32>, Directed> {
    // Determine value range range of node/edge weights
    let node_range = determine_node_value_range(g);
    let edge_range = determine_edge_value_range(g);

    g.map(|_, &nw| normalize_to_closed01(nw, node_range),
          |_, &ew| normalize_to_closed01(ew, edge_range))
}

// Normalize the node and edge weights into the range [0,1]
pub fn normalize_graph(g: &Graph<f32, f32, Directed>)
                       -> Graph<f32, f32, Directed> {
    // Determine value range range of node/edge weights
    let node_range = determine_node_value_range(g);
    let edge_range = determine_edge_value_range(g);

    g.map(|_, &nw| normalize_to_closed01(nw, node_range).get(),
          |_, &ew| normalize_to_closed01(ew, edge_range).get())
}
