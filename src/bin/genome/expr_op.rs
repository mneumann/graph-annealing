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
pub fn random_const_expr<R>(rng: &mut R,
                            weighted_const_op: &OwnedWeightedChoice<ConstExprOp>)
                            -> Expr<f32>
    where R: Rng
{
    match weighted_const_op.ind_sample(rng) {
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

        ExprOp::Param => {
            if num_params > 0 {
                Expr::Arg(rng.gen_range(0, num_params))
            } else {
                Expr::Const(Zero::zero())
            }
        }

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
