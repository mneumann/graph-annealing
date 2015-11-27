use petgraph::{Directed, EdgeDirection, Graph};
use petgraph::graph::NodeIndex;
use petgraph::algo::{connected_components, scc};
use triadic_census::{OptDenseDigraph, SimpleDigraph, TriadicCensus};
use graph_neighbor_matching::neighbor_matching_score;

pub struct Goal {
    target_graph: Graph<(), (), Directed>,
    target_census: TriadicCensus,
    target_connected_components: usize,
    target_strongly_connected_components: usize,
    target_in_a: Vec<Vec<usize>>,
    target_out_a: Vec<Vec<usize>>,
}

fn graph_to_edgelist(g: &Graph<(), (), Directed>) -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {
    let mut in_a: Vec<Vec<usize>> = Vec::with_capacity(g.node_count());
    let mut out_a: Vec<Vec<usize>> = Vec::with_capacity(g.node_count());
    for ni in 0..g.node_count() {
        let mut in_neighbors: Vec<usize> = Vec::new();
        let mut out_neighbors: Vec<usize> = Vec::new();
        for n in g.neighbors_directed(NodeIndex::new(ni), EdgeDirection::Incoming) {
            in_neighbors.push(n.index());
        }
        for n in g.neighbors_directed(NodeIndex::new(ni), EdgeDirection::Outgoing) {
            out_neighbors.push(n.index());
        }
        in_a.push(in_neighbors);
        out_a.push(out_neighbors);
    }

    (in_a, out_a)
}

impl Goal {
    pub fn new(g: Graph<(), (), Directed, u32>) -> Goal {
        let census = TriadicCensus::from(&OptDenseDigraph::from(SimpleDigraph::from(g.clone())));
        let (in_a, out_a) = graph_to_edgelist(&g);
        let cc = connected_components(&g);
        let scc = scc(&g).len();
        Goal {
            target_graph: g,
            target_census: census,
            target_connected_components: cc,
            target_strongly_connected_components: scc,
            target_in_a: in_a,
            target_out_a: out_a,
        }
    }

    pub fn triadic_distance(&self, g: Graph<(), (), Directed, u32>) -> f64 {
        let census = TriadicCensus::from(&OptDenseDigraph::from(SimpleDigraph::from(g)));
        TriadicCensus::distance(&self.target_census, &census)
    }

    pub fn connected_components_distance(&self, g: &Graph<(), (), Directed, u32>) -> usize {
        let cc = connected_components(g) as isize;
        ((self.target_connected_components as isize) - cc).abs() as usize
    }
    pub fn strongly_connected_components_distance(&self,
                                                  g: &Graph<(), (), Directed, u32>)
                                                  -> usize {
        let scc = scc(&g).len() as isize;
        ((self.target_strongly_connected_components as isize) - scc).abs() as usize
    }

    pub fn neighbor_matching_score(&self, g: &Graph<(), (), Directed, u32>) -> f32 {
        let (in_b, out_b) = graph_to_edgelist(g);
        let (_iter, score) = neighbor_matching_score(&self.target_in_a[..],
                                                     &in_b[..],
                                                     &self.target_out_a[..],
                                                     &out_b[..],
                                                     0.1,
                                                     20);
        assert!(score >= 0.0 && score <= 1.0);
        let score = 1.0 - score;
        score
    }
}
