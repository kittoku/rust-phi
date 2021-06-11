use nalgebra as na;
use minilp::{LinearExpr, Problem};

use crate::mechanism::Constellation;


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

pub fn calc_repertoire_emd(vec_from: &na::DVector<f64>, vec_to: &na::DVector<f64>) -> f64 {
    let ndim = check_dimension(vec_from, vec_to);
    let mut problem = Problem::new(minilp::OptimizationDirection::Minimize);
    let mut horizontal_sums = generate_empty_exprs(ndim);
    let mut vertical_sums = generate_empty_exprs(ndim);

    (0..ndim).for_each(|i| {
        (0..ndim).for_each(|j| {
            let d = (i ^ j).count_ones() as f64;
            let e = problem.add_var(d, (0.0, f64::INFINITY));

            horizontal_sums[i].add(e, 1.0);
            vertical_sums[j].add(e, 1.0);
        });
    });

    vec_from.iter().zip(horizontal_sums).for_each(|(&p, expr)| problem.add_constraint(expr, minilp::ComparisonOp::Eq, p));
    vec_to.iter().zip(vertical_sums).for_each(|(&q, expr)| problem.add_constraint(expr, minilp::ComparisonOp::Eq, q));

    problem.solve().unwrap().objective()
}

pub fn calc_constellation_emd(constellation_from: &Constellation, constellation_to: &Constellation) -> f64 {
    let from_concepts_size = constellation_from.concepts.len();
    let to_concepts_size = constellation_to.concepts.len();

    if from_concepts_size == 0 {
        return 0.0;
    }

    let total_from_phi = constellation_from.concepts.iter().fold(0.0, |acc, x| acc + x.phi);
    let total_to_phi = constellation_to.concepts.iter().fold(0.0, |acc, x| acc + x.phi);
    let oversupply = total_from_phi - total_to_phi;

    let mut problem = Problem::new(minilp::OptimizationDirection::Minimize);
    let mut horizontal_sums = generate_empty_exprs(from_concepts_size);
    let mut vertical_sums = generate_empty_exprs(to_concepts_size);
    let mut null_sum = LinearExpr::empty();

    (0..from_concepts_size).for_each(|i| {
        (0..to_concepts_size).for_each(|j| {
            let d = constellation_from.concepts[i].distance_from(&constellation_to.concepts[j]);
            let e = problem.add_var(d, (0.0, f64::INFINITY));

            horizontal_sums[i].add(e, 1.0);
            vertical_sums[j].add(e, 1.0);
        });

        let null_distance = constellation_from.concepts[i].distance_from(&constellation_to.null_concept);
        let null_earth = problem.add_var(null_distance, (0.0, f64::INFINITY));
        null_sum.add(null_earth, 1.0);
        horizontal_sums[i].add(null_earth, 1.0);
    });

    constellation_from.concepts.iter().zip(horizontal_sums).for_each(|(concept, expr)| {
        problem.add_constraint(expr, minilp::ComparisonOp::Eq, concept.phi)
    });

    constellation_to.concepts.iter().zip(vertical_sums).for_each(|(concept, expr)| {
        problem.add_constraint(expr, minilp::ComparisonOp::Eq, concept.phi)
    });

    problem.add_constraint(null_sum, minilp::ComparisonOp::Eq, oversupply);


    problem.solve().unwrap().objective()
}
