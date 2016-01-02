use graph_annealing::owned_weighted_choice::OwnedWeightedChoice;
pub use expression_num::NumExpr as ExprT;
use rand::{Closed01, Open01, Rng};
use rand::distributions::IndependentSample;
use std::num::{One, Zero};
use std::f32::consts;

pub type Expr = ExprT<f32>;

/// FlatExprOp is a non-recursive expression.
defops!{FlatExprOp;
    // 0.0
    Zero,

    // 1.0
    One,

    // Eulers number
    Euler,

    // Pi
    Pi,

    // A constant value in [0, 1]
    ConstClosed01,

    // A constant value in (1, inf), i.e. 1.0 / (0, 1)
    ConstOpen01Reciproc,

    // References a parameter
    Param
}

/// Contains all recursive (unary or binary) expressions.
defops!{RecursiveExprOp;
    // 1.0 / x using safe division.
    Reciprocz,

    // Addition
    Add,

    // Subtraction
    Sub,

    // Multiplication
    Mul,

    // Safe division
    Divz
}

/// Generates a random flat expression according to the parameters:
///
///     rng: Random number generator to use
///     num_params: number of parameters (if == 0, this turns Param into Zero).
///     weighted_flat_op: Used to choose a flat expression
///
pub fn random_flat_expr<R>(rng: &mut R,
                           weighted_flat_op: &OwnedWeightedChoice<FlatExprOp>,
                           num_params: usize)
                           -> Expr
    where R: Rng
{
    let op = weighted_flat_op.ind_sample(rng);
    flat_expr_op_to_expr(rng, op, num_params)
}

pub fn build_recursive_expr<F>(expr: Expr, op: RecursiveExprOp, f: F) -> Expr
where F:FnOnce() -> Expr {
    match op {
        RecursiveExprOp::Reciprocz => expr.op_recipz(),
        RecursiveExprOp::Add => expr.op_add(f()),
        RecursiveExprOp::Sub => expr.op_sub(f()),
        RecursiveExprOp::Mul => expr.op_mul(f()),
        RecursiveExprOp::Divz => expr.op_divz(f()),
    }
}


fn flat_expr_op_to_expr<R: Rng>(rng: &mut R, op: FlatExprOp, num_params: usize) -> Expr {
    match op {
        FlatExprOp::Zero => ExprT::Const(Zero::zero()),

        FlatExprOp::One => ExprT::Const(One::one()),

        FlatExprOp::Euler => ExprT::Const(consts::E),

        FlatExprOp::Pi => ExprT::Const(consts::PI),

        FlatExprOp::ConstClosed01 => {
            let n = rng.gen::<Closed01<f32>>().0;
            debug_assert!(n >= 0.0 && n <= 1.0);
            ExprT::Const(n)
        }

        FlatExprOp::ConstOpen01Reciproc => {
            let n = rng.gen::<Open01<f32>>().0;
            debug_assert!(n > 0.0 && n < 1.0);
            ExprT::Const(1.0 / n)
        }

        FlatExprOp::Param => {
            if num_params > 0 {
                ExprT::Var(rng.gen_range(0, num_params))
            } else {
                ExprT::Const(Zero::zero())
            }
        }
    }
}
