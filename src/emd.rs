use nalgebra as na;
use minilp::{LinearExpr, Problem, Solution};


fn check_dimension(vec_from: &na::DVector<f64>, vec_to: &na::DVector<f64>) -> usize {
    let ndim = vec_from.len();
    assert!(ndim > 0);
    assert!(vec_to.len() == ndim);

    ndim
}

fn generate_empty_exprs(capacity: usize) -> Vec<LinearExpr> {
    let mut vec = Vec::<LinearExpr>::with_capacity(capacity);
    (0..capacity).for_each(|_| vec.push(LinearExpr::empty()));
    vec
}

pub fn calc_emd_for_mechanism(vec_from: &na::DVector<f64>, vec_to: &na::DVector<f64>) -> Solution {
    let ndim = check_dimension(vec_from, vec_to);
    let mut problem = Problem::new(minilp::OptimizationDirection::Minimize);
    let mut sum_rows = generate_empty_exprs(ndim);
    let mut sum_columns = generate_empty_exprs(ndim);

    (0..ndim).for_each(|i| {
        (0..ndim).for_each(|j| {
            let d = (i ^ j).count_ones() as f64;
            let e = problem.add_var(d, (0.0, f64::INFINITY));

            sum_rows[i].add(e, 1.0);
            sum_columns[j].add(e, 1.0);
        });
    });

    vec_from.iter().zip(sum_rows).for_each(|(&p, expr)| problem.add_constraint(expr, minilp::ComparisonOp::Eq, p));
    vec_to.iter().zip(sum_columns).for_each(|(&q, expr)| problem.add_constraint(expr, minilp::ComparisonOp::Eq, q));

    problem.solve().unwrap()
}
