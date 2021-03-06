use petgraph::{Directed, Graph};
use petgraph::algo::{connected_components, scc};
use triadic_census::{OptDenseDigraph, TriadicCensus};
use graph_neighbor_matching::{Edge, GraphSimilarityMatrix, IgnoreNodeColors, ScoreNorm};
use closed01::Closed01;
use std::fmt::Debug;

use graph_neighbor_matching::Graph as NGraph;

defops!{FitnessFunction;
    ConnectedComponents,
    StronglyConnectedComponents,
    NeighborMatchingMinDeg,
    NeighborMatchingMaxDeg,
    NeighborMatchingAvg,
    NeighborMatchingEdgeWeightsMaxDeg,
    NeighborMatchingEdgeWeightsMinDeg,
    TriadicDistance,
    GenomeComplexity
}

pub struct Goal<N: Debug, E: Debug> {
    _target_graph: OptDenseDigraph<N, E>,
    target_census: TriadicCensus,
    target_connected_components: usize,
    target_strongly_connected_components: usize,
    target_in_a: Vec<Vec<Edge>>,
    target_out_a: Vec<Vec<Edge>>,
}

// We need to normalize the edge weights into the range [0,1]
#[cfg(feature = "expr_num")]
fn graph_to_edgelist(g: &Graph<f32, f32, Directed>) -> (Vec<Vec<Edge>>, Vec<Vec<Edge>>) {
    let mut in_a: Vec<Vec<Edge>> = (0..g.node_count()).map(|_| Vec::new()).collect();
    let mut out_a: Vec<Vec<Edge>> = (0..g.node_count()).map(|_| Vec::new()).collect();

    // Determine value range range of node/edge weights
    // let node_range = determine_node_value_range(g);
    let edge_range = determine_edge_value_range(g);

    for edge in g.raw_edges() {
        in_a[edge.target().index()].push(Edge::new(edge.source().index(),
                                                   normalize_to_closed01(edge.weight, edge_range)));
        out_a[edge.source().index()].push(Edge::new(edge.target().index(),
                                                    normalize_to_closed01(edge.weight,
                                                                          edge_range)));
    }

    (in_a, out_a)
}


#[cfg(feature = "expr_closed01")]
fn graph_to_edgelist(g: &Graph<f32, f32, Directed>) -> (Vec<Vec<Edge>>, Vec<Vec<Edge>>) {
    let mut in_a: Vec<Vec<Edge>> = (0..g.node_count()).map(|_| Vec::new()).collect();
    let mut out_a: Vec<Vec<Edge>> = (0..g.node_count()).map(|_| Vec::new()).collect();

    for edge in g.raw_edges() {
        in_a[edge.target().index()]
            .push(Edge::new(edge.source().index(), Closed01::new(edge.weight)));
        out_a[edge.source().index()]
            .push(Edge::new(edge.target().index(), Closed01::new(edge.weight)));
    }

    (in_a, out_a)
}

/// This is used to cache and reuse some heavy calculations done
/// by some FitnessFunctions.
pub struct Cache {
    edge_list: Option<(Vec<Vec<Edge>>, Vec<Vec<Edge>>)>,
}

impl Cache {
    pub fn new() -> Cache {
        Cache { edge_list: None }
    }
}

const NEIGHBORMATCHING_ITERATIONS: usize = 20;
const NEIGHBORMATCHING_EPS: f32 = 0.05;

pub type N = f32;
pub type E = f32;

impl Goal<N, E> {
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
                                  g: &OptDenseDigraph<N, E>,
                                  cache: &mut Cache)
                                  -> f32 {
        match fitfun {
            FitnessFunction::GenomeComplexity => unreachable!(),
            FitnessFunction::ConnectedComponents => self.connected_components_distance(g) as f32,
            FitnessFunction::StronglyConnectedComponents => {
                self.strongly_connected_components_distance(g) as f32
            }
            FitnessFunction::NeighborMatchingMinDeg => {
                self.neighbor_matching_score_sum_norm(g, cache, ScoreNorm::MinDegree) as f32
            }
            FitnessFunction::NeighborMatchingMaxDeg => {
                self.neighbor_matching_score_sum_norm(g, cache, ScoreNorm::MaxDegree) as f32
            }
            FitnessFunction::NeighborMatchingAvg => {
                self.neighbor_matching_score_avg(g, cache) as f32
            }
            FitnessFunction::NeighborMatchingEdgeWeightsMaxDeg => {
                self.neighbor_matching_score_edge_weights_sum_norm(g, cache, ScoreNorm::MaxDegree) as f32
            }
            FitnessFunction::NeighborMatchingEdgeWeightsMinDeg => {
                self.neighbor_matching_score_edge_weights_sum_norm(g, cache, ScoreNorm::MinDegree) as f32
            }
            FitnessFunction::TriadicDistance => self.triadic_distance(g) as f32,
        }
    }

    pub fn triadic_distance(&self, g: &OptDenseDigraph<N, E>) -> f64 {
        let census = TriadicCensus::from(g);
        TriadicCensus::distance(&self.target_census, &census)
    }

    pub fn connected_components_distance(&self, g: &OptDenseDigraph<N, E>) -> usize {
        let cc = connected_components(g.ref_graph()) as isize;
        ((self.target_connected_components as isize) - cc).abs() as usize
    }

    pub fn strongly_connected_components_distance(&self, g: &OptDenseDigraph<N, E>) -> usize {
        let scc = scc(g.ref_graph()).len() as isize;
        ((self.target_strongly_connected_components as isize) - scc).abs() as usize
    }

    pub fn neighbor_matching_score_sum_norm(&self,
                                            g: &OptDenseDigraph<N, E>,
                                            cache: &mut Cache,
                                            norm: ScoreNorm)
                                            -> f32 {
        if let None = cache.edge_list {
            cache.edge_list = Some(graph_to_edgelist(g.ref_graph()));
        }

        let &(ref in_b, ref out_b) = cache.edge_list.as_ref().unwrap();
        let mut sim = GraphSimilarityMatrix::new(NGraph::new(&self.target_in_a[..],
                                                             &self.target_out_a[..]),
                                                 NGraph::new(&in_b[..], &out_b[..]),
                                                 IgnoreNodeColors);
        sim.iterate(NEIGHBORMATCHING_ITERATIONS, NEIGHBORMATCHING_EPS);
        sim.score_optimal_sum_norm(None, norm).inv().get()
    }

    pub fn neighbor_matching_score_avg(&self, g: &OptDenseDigraph<N, E>, cache: &mut Cache) -> f32 {
        if let None = cache.edge_list {
            cache.edge_list = Some(graph_to_edgelist(g.ref_graph()));
        }

        let &(ref in_b, ref out_b) = cache.edge_list.as_ref().unwrap();
        let mut sim = GraphSimilarityMatrix::new(NGraph::new(&self.target_in_a[..],
                                                             &self.target_out_a[..]),
                                                 NGraph::new(&in_b[..], &out_b[..]),
                                                 IgnoreNodeColors);
        sim.iterate(NEIGHBORMATCHING_ITERATIONS, NEIGHBORMATCHING_EPS);
        sim.score_average().inv().get()
    }

    pub fn neighbor_matching_score_edge_weights_sum_norm(&self,
                                                         g: &OptDenseDigraph<N, E>,
                                                         cache: &mut Cache,
                                                         norm: ScoreNorm)
                                                         -> f32 {
        if let None = cache.edge_list {
            cache.edge_list = Some(graph_to_edgelist(g.ref_graph()));
        }

        let &(ref in_b, ref out_b) = cache.edge_list.as_ref().unwrap();
        let mut sim = GraphSimilarityMatrix::new(NGraph::new(&self.target_in_a[..],
                                                             &self.target_out_a[..]),
                                                 NGraph::new(&in_b[..], &out_b[..]),
                                                 IgnoreNodeColors);
        sim.iterate(NEIGHBORMATCHING_ITERATIONS, NEIGHBORMATCHING_EPS);

        let assignment = sim.optimal_node_assignment();

        sim.score_outgoing_edge_weights_sum_norm(&assignment, norm).inv().get()
    }
}
