use graph_annealing::owned_weighted_choice::OwnedWeightedChoice;
pub use expression_closed01::Closed01Expr as ExprT;
use rand::distributions::IndependentSample;
use closed01::Closed01;
use rand::{Rand, Rng};

pub type ExprScalar = Closed01<f32>;
pub type Expr = ExprT;

pub fn expr_zero() -> ExprScalar {
    ExprScalar::zero()
}

pub const EXPR_NAME: &'static str = "ExprClosed01";

pub fn expr_conv_to_f32(s: &ExprScalar) -> f32 {
    s.get().fract()
}

/// FlatExprOp is a non-recursive expression.
defops!{FlatExprOp;
    // A constant value in [0, 1]
    Const,

    // References a parameter
    Param
}

/// Contains all recursive (unary or binary) expressions.
defops!{RecursiveExprOp;
    Min,
    Max,
    Distance,
    Avg,
    SatAdd,
    SatSub,
    Mul,
    ScaleUp,
    ScaleDown,
    Inv,
    Round,
    PertubeUp,
    PertubeDown
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


fn flat_expr_op_to_expr<R: Rng>(rng: &mut R, op: FlatExprOp, num_params: usize) -> Expr {
    match op {
        FlatExprOp::Const => {
            ExprT::Const(rng.gen::<Closed01<f32>>())
        }

        FlatExprOp::Param => {
            if num_params > 0 {
                ExprT::Var(rng.gen_range(0, num_params))
            } else {
                ExprT::Const(Closed01::zero())
            }
        }
    }
}

pub fn build_recursive_expr<F>(expr: Expr, op: RecursiveExprOp, f: F) -> Expr
where F:FnOnce() -> Expr {
    match op {
        RecursiveExprOp::Min => expr.op_min(f()),
        RecursiveExprOp::Max => expr.op_max(f()),
        RecursiveExprOp::Distance => expr.op_distance(f()),
        RecursiveExprOp::Avg => expr.op_avg(f()),
        RecursiveExprOp::SatAdd => expr.op_sat_add(f()),
        RecursiveExprOp::SatSub => expr.op_sat_sub(f()),
        RecursiveExprOp::Mul => expr.op_mul(f()),
        RecursiveExprOp::ScaleUp => expr.op_scale_up(f()),
        RecursiveExprOp::ScaleDown => expr.op_scale_down(f()),
        RecursiveExprOp::Inv => expr.op_inv(),
        RecursiveExprOp::Round => expr.op_round(),
        RecursiveExprOp::PertubeUp => {
            expr.op_scale_up(ExprT::Const(Closed01::new(0.1)))
            /*let r = rng.gen::<Closed01<f32>>();
            if r.get() > 0.5 {
                expr.op_sat_add(r.mul(Closed01::new(0.1)))
            } else {
                expr.op_sat_sub(r.mul(Closed01::new(0.1)))
            }*/
        }
        RecursiveExprOp::PertubeDown => {
            expr.op_scale_down(ExprT::Const(Closed01::new(0.1)))
        }

    }
}


