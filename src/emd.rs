use minilp::{LinearExpr, Problem, Solution};


fn check_arguments(P: &Vec<f64>, Q: &Vec<f64>) -> usize {
    let ndim = P.len();
    assert!(ndim > 0);
    assert!(Q.len() == ndim);

    assert!(P.iter().all(|&v| v >= 0.0));
    assert!(Q.iter().all(|&v| v >= 0.0));

    ndim
}

fn generate_empty_exprs(capacity: usize) -> Vec<LinearExpr> {
    let mut vec = Vec::<LinearExpr>::with_capacity(capacity);
    (0..capacity).for_each(|_| vec.push(LinearExpr::empty()));
    vec
}

pub fn calc_emd_for_mechanism(P: &Vec<f64>, Q: &Vec<f64>) -> Solution {
    let ndim = check_arguments(P, Q);
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

    P.iter().zip(sum_rows).for_each(|(&p, expr)| problem.add_constraint(expr, minilp::ComparisonOp::Eq, p));
    Q.iter().zip(sum_columns).for_each(|(&q, expr)| problem.add_constraint(expr, minilp::ComparisonOp::Eq, q));

    problem.solve().unwrap()
}
