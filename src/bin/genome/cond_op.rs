use graph_annealing::owned_weighted_choice::OwnedWeightedChoice;
use lindenmayer_system::expr::{Condition, Expr};
use rand::{Closed01, Open01, Rng};
use rand::distributions::IndependentSample;
use std::num::{One, Zero};
use std::f32::consts;

defops!{CondOp;
    True,
    False,
    Not,
    And,
    Or,
    Equal,
    Less,
    Greater,
    LessEqual,
    GreaterEqual
}

/// Generates a random condition according to the parameters:
///
///     rng: Random number generator to use
///     max_depth: maximum recursion depth.
///     weighted_op: Used to choose an expression when the max recursion depth is NOT reached.
///     weighted_op_max_depth: Used to choose an expression when the max recursion depth is reached.
///     expr_fn: A function that returns an expression, passing the Rng and the current depth of
///     the condition.
///
pub fn random_cond<R, F>(rng: &mut R,
                         max_depth: usize,
                         weighted_op: &OwnedWeightedChoice<CondOp>,
                         weighted_op_max_depth: &OwnedWeightedChoice<CondOp>,
                         expr_fn: &mut F)
                         -> Condition<f32>
    where R: Rng,
          F: FnMut(&mut R, usize) -> Expr<f32>
{
    let choose_from = if max_depth > 0 {
        weighted_op
    } else {
        weighted_op_max_depth
    };

    match choose_from.ind_sample(rng) {
        CondOp::True => Condition::True,

        CondOp::False => Condition::False,

        CondOp::Not => {
            if max_depth > 0 {
                let op = Box::new(random_cond(rng,
                                              max_depth - 1,
                                              weighted_op,
                                              weighted_op_max_depth,
                                              expr_fn));
                Condition::Not(op)
            } else {
                Condition::False
            }
        }

        CondOp::And => {
            if max_depth > 0 {
                let op1 = Box::new(random_cond(rng,
                                               max_depth - 1,
                                               weighted_op,
                                               weighted_op_max_depth,
                                               expr_fn));
                let op2 = Box::new(random_cond(rng,
                                               max_depth - 1,
                                               weighted_op,
                                               weighted_op_max_depth,
                                               expr_fn));
                Condition::And(op1, op2)
            } else {
                Condition::False
            }
        }

        CondOp::Or => {
            if max_depth > 0 {
                let op1 = Box::new(random_cond(rng,
                                               max_depth - 1,
                                               weighted_op,
                                               weighted_op_max_depth,
                                               expr_fn));
                let op2 = Box::new(random_cond(rng,
                                               max_depth - 1,
                                               weighted_op,
                                               weighted_op_max_depth,
                                               expr_fn));
                Condition::Or(op1, op2)
            } else {
                Condition::False
            }
        }

        CondOp::Equal => {
            let ex1 = Box::new(expr_fn(rng, max_depth));
            let ex2 = Box::new(expr_fn(rng, max_depth));
            Condition::Equal(ex1, ex2)
        }

        CondOp::Less => {
            let ex1 = Box::new(expr_fn(rng, max_depth));
            let ex2 = Box::new(expr_fn(rng, max_depth));
            Condition::Less(ex1, ex2)
        }

        CondOp::Greater => {
            let ex1 = Box::new(expr_fn(rng, max_depth));
            let ex2 = Box::new(expr_fn(rng, max_depth));
            Condition::Greater(ex1, ex2)
        }

        CondOp::LessEqual => {
            let ex1 = Box::new(expr_fn(rng, max_depth));
            let ex2 = Box::new(expr_fn(rng, max_depth));
            Condition::LessEqual(ex1, ex2)
        }

        CondOp::GreaterEqual => {
            let ex1 = Box::new(expr_fn(rng, max_depth));
            let ex2 = Box::new(expr_fn(rng, max_depth));
            Condition::GreaterEqual(ex1, ex2)
        }
    }
}
