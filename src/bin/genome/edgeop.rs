use std::str::FromStr;
use std::string::ToString;

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
