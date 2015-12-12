use std::str::FromStr;
use std::string::ToString;
use graph_edge_evolution::{EdgeOperation, GraphBuilder, NthEdgeF};
use triadic_census::OptDenseDigraph;
use std::collections::BTreeMap;

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum EdgeOp {
    Dup,
    Split,
    Loop,
    Merge,
    Next,
    Parent,
    Reverse,
    Save,
    Restore,
}

impl ToString for EdgeOp {
    fn to_string(&self) -> String {
        match *self {
            EdgeOp::Dup => "Dup".to_string(),
            EdgeOp::Split => "Split".to_string(),
            EdgeOp::Loop => "Loop".to_string(),
            EdgeOp::Merge => "Merge".to_string(),
            EdgeOp::Next => "Next".to_string(),
            EdgeOp::Parent => "Parent".to_string(),
            EdgeOp::Reverse => "Reverse".to_string(),
            EdgeOp::Save => "Save".to_string(),
            EdgeOp::Restore => "Restore".to_string(),
        }
    }
}

impl FromStr for EdgeOp {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "Dup" => Ok(EdgeOp::Dup),
            "Split" => Ok(EdgeOp::Split),
            "Loop" => Ok(EdgeOp::Loop),
            "Merge" => Ok(EdgeOp::Merge),
            "Next" => Ok(EdgeOp::Next),
            "Parent" => Ok(EdgeOp::Parent),
            "Reverse" => Ok(EdgeOp::Reverse),
            "Save" => Ok(EdgeOp::Save),
            "Restore" => Ok(EdgeOp::Restore),
            _ => Err(format!("Invalid opcode: {}", s)),
        }
    }
}

pub fn edgeops_to_graph(edgeops: &[(EdgeOp, f32)]) -> OptDenseDigraph<(), ()> {
    let mut builder: GraphBuilder<f32, ()> = GraphBuilder::new();
    for &(op, f) in edgeops {
        let graph_op = match op {
            EdgeOp::Dup => EdgeOperation::Duplicate { weight: f },
            EdgeOp::Split => EdgeOperation::Split { weight: f },
            EdgeOp::Loop => EdgeOperation::Loop { weight: f },
            EdgeOp::Merge => EdgeOperation::Merge { n: NthEdgeF(f) },
            EdgeOp::Next => EdgeOperation::Next { n: NthEdgeF(f) },
            EdgeOp::Parent => EdgeOperation::Parent { n: NthEdgeF(f) },
            EdgeOp::Reverse => EdgeOperation::Reverse,
            EdgeOp::Save => EdgeOperation::Save,
            EdgeOp::Restore => EdgeOperation::Restore,
        };
        builder.apply_operation(graph_op);
    }

    let mut g: OptDenseDigraph<(), ()> = OptDenseDigraph::new(builder.total_number_of_nodes()); // XXX: rename to real_number

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
