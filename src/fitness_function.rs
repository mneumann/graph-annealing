use std::str::FromStr;

#[derive(Debug, Copy, Clone)]
pub enum FitnessFunction {
    Null,
    ConnectedComponents,
    StronglyConnectedComponents,
    NeighborMatching,
    TriadicDistance,
}

impl FromStr for FitnessFunction {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "null" => Ok(FitnessFunction::Null),
            "cc" => Ok(FitnessFunction::ConnectedComponents),
            "scc" => Ok(FitnessFunction::StronglyConnectedComponents),
            "nm" => Ok(FitnessFunction::NeighborMatching),
            "td" => Ok(FitnessFunction::TriadicDistance),
            _ => Err(format!("Invalid fitness function: {}", s)),
        }
    }
}
