use graph_annealing::owned_weighted_choice::OwnedWeightedChoice;
use expression_num::NumExpr as Expr;
use rand::{Closed01, Open01, Rng};
use rand::distributions::IndependentSample;
use std::num::{One, Zero};
use std::f32::consts;

defops!{ConstExprOp;
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
    ConstOpen01Reciproc
}

// FlatExprOp is a non-recursive expression. It contains all
// expressions of ConstExprOp, plus Param.
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

/// Generates a random constant expression according to the parameters:
///
///     rng: Random number generator to use
///     weighted_const_op: Used to choose a const expression
///
pub fn random_const_expr<R: Rng>(rng: &mut R,
                                 weighted_const_op: &OwnedWeightedChoice<ConstExprOp>)
                                 -> Expr<f32> {
    let op = weighted_const_op.ind_sample(rng);
    const_expr_op_to_expr(rng, op)
}

fn const_expr_op_to_expr<R: Rng>(rng: &mut R, op: ConstExprOp) -> Expr<f32> {
    match op {
        ConstExprOp::Zero => Expr::Const(Zero::zero()),

        ConstExprOp::One => Expr::Const(One::one()),

        ConstExprOp::Euler => Expr::Const(consts::E),

        ConstExprOp::Pi => Expr::Const(consts::PI),

        ConstExprOp::ConstClosed01 => {
            let n = rng.gen::<Closed01<f32>>().0;
            debug_assert!(n >= 0.0 && n <= 1.0);
            Expr::Const(n)
        }

        ConstExprOp::ConstOpen01Reciproc => {
            let n = rng.gen::<Open01<f32>>().0;
            debug_assert!(n > 0.0 && n < 1.0);
            Expr::Const(1.0 / n)
        }
    }
}


/// Generates a random flat expression according to the parameters:
///
///     rng: Random number generator to use
///     num_params: number of parameters.
///     weighted_const_op: Used to choose a const expression
///
pub fn random_flat_expr<R>(rng: &mut R,
                           weighted_flat_op: &OwnedWeightedChoice<FlatExprOp>,
                           num_params: usize)
                           -> Expr<f32>
    where R: Rng
{
    let op = weighted_flat_op.ind_sample(rng);
    flat_expr_op_to_expr(rng, op, num_params)
}

fn flat_expr_op_to_expr<R: Rng>(rng: &mut R, op: FlatExprOp, num_params: usize) -> Expr<f32> {
    match op {
        FlatExprOp::Zero => Expr::Const(Zero::zero()),

        FlatExprOp::One => Expr::Const(One::one()),

        FlatExprOp::Euler => Expr::Const(consts::E),

        FlatExprOp::Pi => Expr::Const(consts::PI),

        FlatExprOp::ConstClosed01 => {
            let n = rng.gen::<Closed01<f32>>().0;
            debug_assert!(n >= 0.0 && n <= 1.0);
            Expr::Const(n)
        }

        FlatExprOp::ConstOpen01Reciproc => {
            let n = rng.gen::<Open01<f32>>().0;
            debug_assert!(n > 0.0 && n < 1.0);
            Expr::Const(1.0 / n)
        }

        FlatExprOp::Param => {
            if num_params > 0 {
                Expr::Var(rng.gen_range(0, num_params))
            } else {
                Expr::Const(Zero::zero())
            }
        }
    }
}
