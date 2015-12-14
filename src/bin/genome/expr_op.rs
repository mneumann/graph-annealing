use graph_annealing::owned_weighted_choice::OwnedWeightedChoice;
use lindenmayer_system::expr::Expr;
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


defops!{ExprOp;
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
    Param,

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
                Expr::Arg(rng.gen_range(0, num_params))
            } else {
                Expr::Const(Zero::zero())
            }
        }
    }
}




/// Generates a random expression according to the parameters:
///
///     rng: Random number generator to use
///     num_params: number of parameters.
///     max_depth: maximum recursion depth.
///     weighted_expr_op: Used to choose an expression when the max recursion depth is NOT reached.
///     weighted_flat_op: Used to choose an expression when the max recursion depth is reached.
///                       Only flat expressions (non-recursive).
///
pub fn random_expr<R>(rng: &mut R,
                      num_params: usize,
                      max_depth: usize,
                      weighted_expr_op: &OwnedWeightedChoice<ExprOp>,
                      weighted_flat_op: &OwnedWeightedChoice<FlatExprOp>)
                      -> Expr<f32>
    where R: Rng
{
    if max_depth > 0 {
        let expr_op = weighted_expr_op.ind_sample(rng);
        match expr_op {
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

            ExprOp::Param => {
                if num_params > 0 {
                    Expr::Arg(rng.gen_range(0, num_params))
                } else {
                    Expr::Const(Zero::zero())
                }
            }

            ExprOp::Reciprocz => {
                let op = Box::new(random_expr(rng,
                                              num_params,
                                              max_depth - 1,
                                              weighted_expr_op,
                                              weighted_flat_op));
                Expr::Recipz(op)
            }

            ExprOp::Add => {
                let op1 = Box::new(random_expr(rng,
                                               num_params,
                                               max_depth - 1,
                                               weighted_expr_op,
                                               weighted_flat_op));
                let op2 = Box::new(random_expr(rng,
                                               num_params,
                                               max_depth - 1,
                                               weighted_expr_op,
                                               weighted_flat_op));
                Expr::Add(op1, op2)
            }

            ExprOp::Sub => {
                let op1 = Box::new(random_expr(rng,
                                               num_params,
                                               max_depth - 1,
                                               weighted_expr_op,
                                               weighted_flat_op));
                let op2 = Box::new(random_expr(rng,
                                               num_params,
                                               max_depth - 1,
                                               weighted_expr_op,
                                               weighted_flat_op));
                Expr::Sub(op1, op2)
            }

            ExprOp::Mul => {
                let op1 = Box::new(random_expr(rng,
                                               num_params,
                                               max_depth - 1,
                                               weighted_expr_op,
                                               weighted_flat_op));
                let op2 = Box::new(random_expr(rng,
                                               num_params,
                                               max_depth - 1,
                                               weighted_expr_op,
                                               weighted_flat_op));
                Expr::Mul(op1, op2)
            }

            ExprOp::Divz => {
                let op1 = Box::new(random_expr(rng,
                                               num_params,
                                               max_depth - 1,
                                               weighted_expr_op,
                                               weighted_flat_op));
                let op2 = Box::new(random_expr(rng,
                                               num_params,
                                               max_depth - 1,
                                               weighted_expr_op,
                                               weighted_flat_op));
                Expr::Divz(op1, op2)
            }
        }

    } else {
        random_flat_expr(rng, weighted_flat_op, num_params)
    }
}
