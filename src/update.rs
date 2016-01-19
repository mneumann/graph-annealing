use closed01::Closed01;
use petgraph::{Directed, Graph};
use graph_neighbor_matching::{Edge, GraphSimilarityMatrix, IgnoreNodeColors, ScoreNorm};
use graph_neighbor_matching::Graph as NGraph;

pub type WeightedDigraph = Graph<Closed01<f32>, Closed01<f32>, Directed>;

pub trait Objective {
    type Input;
    type Output;

    fn eval(&self, inp: &Self::Input) -> Self::Output; 
}

pub struct NeighborMatching {
    score_norm: ScoreNorm,
    edge_list: (Vec<Vec<Edge>>, Vec<Vec<Edge>>), 

    // number of iterations
    iters: usize,

    // eps
    eps: f32,
}

fn graph_to_edgelist(g: &WeightedDigraph) -> (Vec<Vec<Edge>>, Vec<Vec<Edge>>) {
    let mut in_a: Vec<Vec<Edge>> = (0..g.node_count()).map(|_| Vec::new()).collect();
    let mut out_a: Vec<Vec<Edge>> = (0..g.node_count()).map(|_| Vec::new()).collect();

    for edge in g.raw_edges() {
        in_a[edge.target().index()].push(Edge::new(edge.source().index(), edge.weight));
        out_a[edge.source().index()].push(Edge::new(edge.target().index(), edge.weight));
    }

    (in_a, out_a)
}

impl NeighborMatching {
    pub fn new(target_graph: WeightedDigraph, score_norm: ScoreNorm, iters: usize, eps: f32) -> NeighborMatching {
        NeighborMatching {
            score_norm: score_norm,
            edge_list: graph_to_edgelist(&target_graph),
            iters: iters,
            eps: eps,
        }
    }
}

impl Objective for NeighborMatching
{
    type Input = Graph<Closed01<f32>, Closed01<f32>, Directed>;
    type Output = Closed01<f32>;

    fn eval(&self, input: &Self::Input) -> Self::Output {
        let (in_b, out_b) = graph_to_edgelist(input);

        let mut sim = GraphSimilarityMatrix::new(NGraph::new(&self.edge_list.0[..],
                                                             &self.edge_list.1[..]),
                                                 NGraph::new(&in_b[..], &out_b[..]),
                                                 IgnoreNodeColors);
        sim.iterate(self.iters, self.eps);
        sim.score_optimal_sum_norm(None, self.score_norm).inv()
    }
}
