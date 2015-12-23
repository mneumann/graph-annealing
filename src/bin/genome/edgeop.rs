use graph_edge_evolution::{EdgeOperation, GraphBuilder, NthEdgeF};
use triadic_census::OptDenseDigraph;
use std::collections::BTreeMap;
use asexp::{Sexp};

defops!{EdgeOp;
    Dup,
    Split,
    Loop,
    Merge,
    Next,
    Parent,
    Reverse,
    Save,
    Restore,
    Output
}

impl Into<Sexp> for EdgeOp {
    fn into(self) -> Sexp {
        Sexp::from(self.to_string())
    }
}

pub fn edgeops_to_graph(edgeops: &[(EdgeOp, f32)]) -> OptDenseDigraph<(), ()> {
    let mut builder: GraphBuilder<f32, ()> = GraphBuilder::new();
    for &(op, f) in edgeops {
        let graph_op = match op {
            EdgeOp::Dup => EdgeOperation::Duplicate { weight: f },
            EdgeOp::Split => EdgeOperation::Split { weight: f },
            EdgeOp::Loop => EdgeOperation::Loop { weight: f },
            EdgeOp::Output => EdgeOperation::Output { weight: f },
            EdgeOp::Merge => EdgeOperation::Merge { n: NthEdgeF(f.abs().fract()) },
            EdgeOp::Next => EdgeOperation::Next { n: NthEdgeF(f.abs().fract()) },
            EdgeOp::Parent => EdgeOperation::Parent { n: NthEdgeF(f.abs().fract()) },
            EdgeOp::Reverse => EdgeOperation::Reverse,
            EdgeOp::Save => EdgeOperation::Save,
            EdgeOp::Restore => EdgeOperation::Restore,
        };
        builder.apply_operation(graph_op);
    }

    let mut g: OptDenseDigraph<(), ()> = OptDenseDigraph::new(builder.count_nodes());

    // maps node_idx to index used within the graph.
    let mut node_map: BTreeMap<usize, usize> = BTreeMap::new(); // XXX: with_capacity

    builder.visit_nodes(|node_idx, _| {
        let graph_idx = g.add_node();
        node_map.insert(node_idx, graph_idx);
    });

    builder.visit_edges(|(a, b), _| {
        g.add_edge(node_map[&a], node_map[&b]);
    });

    return g;
}
