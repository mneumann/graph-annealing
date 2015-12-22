use petgraph::{Directed, EdgeDirection, Graph};
use petgraph::graph::NodeIndex;
use petgraph::algo::{connected_components, scc};
use triadic_census::{OptDenseDigraph, TriadicCensus};
use graph_neighbor_matching::{Edge, GraphSimilarityMatrix, IgnoreNodeColors};
use super::fitness_function::FitnessFunction;
use closed01::Closed01;
use std::fmt::Debug;

use graph_neighbor_matching::Graph as NGraph;

//#[derive(Debug)]
pub struct Goal<N: Debug, E: Debug> {
    _target_graph: OptDenseDigraph<N, E>,
    target_census: TriadicCensus,
    target_connected_components: usize,
    target_strongly_connected_components: usize,
    target_in_a: Vec<Vec<Edge>>,
    target_out_a: Vec<Vec<Edge>>,
}

fn graph_to_edgelist<N: Debug, E: Debug>(g: &Graph<N, E, Directed>) -> (Vec<Vec<Edge>>, Vec<Vec<Edge>>) {
    let mut in_a: Vec<Vec<Edge>> = Vec::with_capacity(g.node_count());
    let mut out_a: Vec<Vec<Edge>> = Vec::with_capacity(g.node_count());
    for ni in 0..g.node_count() {
        let mut in_neighbors: Vec<Edge> = Vec::new();
        let mut out_neighbors: Vec<Edge> = Vec::new();
        for n in g.neighbors_directed(NodeIndex::new(ni), EdgeDirection::Incoming) {
            in_neighbors.push(Edge::new(n.index(), Closed01::zero()));
        }
        for n in g.neighbors_directed(NodeIndex::new(ni), EdgeDirection::Outgoing) {
            out_neighbors.push(Edge::new(n.index(), Closed01::zero()));
        }
        in_a.push(in_neighbors);
        out_a.push(out_neighbors);
    }

    (in_a, out_a)
}

impl<N: Clone + Default + Debug, E: Clone + Default + Debug> Goal<N, E> {
    pub fn new(g: OptDenseDigraph<N, E>) -> Goal<N, E> {
        let census = TriadicCensus::from(&g);
        let (in_a, out_a) = graph_to_edgelist(g.ref_graph());
        let cc = connected_components(g.ref_graph());
        let scc = scc(g.ref_graph()).len();
        Goal {
            _target_graph: g,
            target_census: census,
            target_connected_components: cc,
            target_strongly_connected_components: scc,
            target_in_a: in_a,
            target_out_a: out_a,
        }
    }

    pub fn apply_fitness_function(&self,
                                  fitfun: FitnessFunction,
                                  g: &OptDenseDigraph<(), ()>)
                                  -> f32 {
        match fitfun {
            FitnessFunction::Null => 0.0,
            FitnessFunction::ConnectedComponents => self.connected_components_distance(g) as f32,
            FitnessFunction::StronglyConnectedComponents => {
                self.strongly_connected_components_distance(g) as f32
            }
            FitnessFunction::NeighborMatching => self.neighbor_matching_score(g) as f32,
            FitnessFunction::TriadicDistance => self.triadic_distance(g) as f32,
        }
    }

    pub fn triadic_distance<A: Debug, B: Debug>(&self, g: &OptDenseDigraph<A, B>) -> f64 {
        let census = TriadicCensus::from(g);
        TriadicCensus::distance(&self.target_census, &census)
    }

    pub fn connected_components_distance<A: Default+Debug, B: Default+Debug>(&self,
                                                                 g: &OptDenseDigraph<A, B>)
                                                                 -> usize {
        let cc = connected_components(g.ref_graph()) as isize;
        ((self.target_connected_components as isize) - cc).abs() as usize
    }
    pub fn strongly_connected_components_distance<A: Default+Debug, B: Default+Debug>(&self,
                                                                          g: &OptDenseDigraph<A, B>)
                                                                          -> usize {
        let scc = scc(g.ref_graph()).len() as isize;
        ((self.target_strongly_connected_components as isize) - scc).abs() as usize
    }

    pub fn neighbor_matching_score<A: Default+Debug, B: Default+Debug>(&self,
                                                           g: &OptDenseDigraph<A, B>)
                                                           -> f32 {
        let (in_b, out_b) = graph_to_edgelist(g.ref_graph());
        let mut sim = GraphSimilarityMatrix::new(NGraph::new(&self.target_in_a[..],
                                                             &self.target_out_a[..]),
                                                 NGraph::new(&in_b[..], &out_b[..]),
                                                 IgnoreNodeColors);
        // XXX
        sim.iterate(20, 0.1);

        let score = sim.score_sum_norm_min_degree(None);

        let score = 1.0 - score.get();
        score
    }
}
