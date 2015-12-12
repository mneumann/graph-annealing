use graph_annealing::owned_weighted_choice::OwnedWeightedChoice;
use lindenmayer_system::expr::Expr;
use rand::{Closed01, Open01, Rng};
use rand::distributions::IndependentSample;
use std::num::{One, Zero};
use std::f32::consts;
use std::str::FromStr;
use std::string::ToString;

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum ExprOp {
    /// 0.0
    Zero,

    /// 1.0
    One,

    /// Eulers number
    Euler,

    /// Pi
    Pi,

    /// A constant value in [0, 1]
    ConstClosed01,

    /// A constant value in (1, inf), i.e. 1.0 / (0, 1)
    ConstOpen01Reciproc,

    /// References a parameter
    Param,

    /// 1.0 / x using safe division.
    Reciprocz,

    /// Addition
    Add,

    /// Subtraction
    Sub,

    /// Multiplication
    Mul,

    /// Safe division
    Divz,
}

impl ToString for ExprOp {
    fn to_string(&self) -> String {
        match *self {
            ExprOp::Zero => "Zero".to_string(),
            ExprOp::One => "One".to_string(),
            ExprOp::Euler => "Euler".to_string(),
            ExprOp::Pi => "Pi".to_string(),
            ExprOp::ConstClosed01 => "ConstClosed01".to_string(),
            ExprOp::ConstOpen01Reciproc => "ConstOpen01Reciproc".to_string(),
            ExprOp::Param => "Param".to_string(),
            ExprOp::Reciprocz => "Reciprocz".to_string(),
            ExprOp::Add => "Add".to_string(),
            ExprOp::Sub => "Sub".to_string(),
            ExprOp::Mul => "Mul".to_string(),
            ExprOp::Divz => "Divz".to_string(),
        }
    }
}

impl FromStr for ExprOp {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "Zero" => Ok(ExprOp::Zero),
            "One" => Ok(ExprOp::One),
            "Euler" => Ok(ExprOp::Euler),
            "Pi" => Ok(ExprOp::Pi),
            "ConstClosed01" => Ok(ExprOp::ConstClosed01),
            "ConstOpen01Reciproc" => Ok(ExprOp::ConstOpen01Reciproc),
            "Param" => Ok(ExprOp::Param),
            "Reciprocz" => Ok(ExprOp::Reciprocz),
            "Add" => Ok(ExprOp::Add),
            "Sub" => Ok(ExprOp::Sub),
            "Mul" => Ok(ExprOp::Mul),
            "Divz" => Ok(ExprOp::Divz),
            _ => Err(format!("Invalid expression opcode: {}", s)),
        }
    }
}

/// Generates a random expression according to the parameters:
///
///     rng: Random number generator to use
///     num_params: number of parameters.
///     max_depth: maximum recursion depth.
///     weighted_op: Used to choose an expression when the max recursion depth is NOT reached.
///     weighted_op_max_depth: Used to choose an expression when the max recursion depth is reached.
///
pub fn random_expr<R>(rng: &mut R,
                      num_params: usize,
                      max_depth: usize,
                      weighted_op: &OwnedWeightedChoice<ExprOp>,
                      weighted_op_max_depth: &OwnedWeightedChoice<ExprOp>)
                      -> Expr<f32>
    where R: Rng
{
    let choose_from = if max_depth > 0 {
        weighted_op
    } else {
        weighted_op_max_depth
    };

    match choose_from.ind_sample(rng) {
        ExprOp::Zero => Expr::Const(Zero::zero()),

        ExprOp::One => Expr::Const(One::one()),

        ExprOp::Euler => Expr::Const(consts::E),

        ExprOp::Pi => Expr::Const(consts::PI),

        ExprOp::ConstClosed01 => {
            let n = rng.gen::<Closed01<f32>>().0;
            debug_assert!(n >= 0.0 && n <= 1.0);
            Expr::Const(n)
        }

        ExprOp::ConstOpen01Reciproc => {
            let n = rng.gen::<Open01<f32>>().0;
            debug_assert!(n > 0.0 && n < 1.0);
            Expr::Const(1.0 / n)
        }

        ExprOp::Param => Expr::Arg(rng.gen_range(0, num_params)),

        ExprOp::Reciprocz => {
            if max_depth > 0 {
                let op = Box::new(random_expr(rng,
                                              num_params,
                                              max_depth - 1,
                                              weighted_op,
                                              weighted_op_max_depth));
                Expr::Recipz(op)
            } else {
                Expr::Const(Zero::zero())
            }
        }

        ExprOp::Add => {
            if max_depth > 0 {
                let op1 = Box::new(random_expr(rng,
                                               num_params,
                                               max_depth - 1,
                                               weighted_op,
                                               weighted_op_max_depth));
                let op2 = Box::new(random_expr(rng,
                                               num_params,
                                               max_depth - 1,
                                               weighted_op,
                                               weighted_op_max_depth));
                Expr::Add(op1, op2)
            } else {
                Expr::Const(Zero::zero())
            }
        }

        ExprOp::Sub => {
            if max_depth > 0 {
                let op1 = Box::new(random_expr(rng,
                                               num_params,
                                               max_depth - 1,
                                               weighted_op,
                                               weighted_op_max_depth));
                let op2 = Box::new(random_expr(rng,
                                               num_params,
                                               max_depth - 1,
                                               weighted_op,
                                               weighted_op_max_depth));
                Expr::Sub(op1, op2)
            } else {
                Expr::Const(Zero::zero())
            }
        }

        ExprOp::Mul => {
            if max_depth > 0 {
                let op1 = Box::new(random_expr(rng,
                                               num_params,
                                               max_depth - 1,
                                               weighted_op,
                                               weighted_op_max_depth));
                let op2 = Box::new(random_expr(rng,
                                               num_params,
                                               max_depth - 1,
                                               weighted_op,
                                               weighted_op_max_depth));
                Expr::Mul(op1, op2)
            } else {
                Expr::Const(Zero::zero())
            }
        }

        ExprOp::Divz => {
            if max_depth > 0 {
                let op1 = Box::new(random_expr(rng,
                                               num_params,
                                               max_depth - 1,
                                               weighted_op,
                                               weighted_op_max_depth));
                let op2 = Box::new(random_expr(rng,
                                               num_params,
                                               max_depth - 1,
                                               weighted_op,
                                               weighted_op_max_depth));
                Expr::Divz(op1, op2)
            } else {
                Expr::Const(Zero::zero())
            }
        }
    }
}
