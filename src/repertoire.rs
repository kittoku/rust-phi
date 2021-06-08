use std::usize;
use nalgebra as na;
use crate::basis::BitBasis;


pub fn normalize_repertoire(repertoire: &mut na::DVector<f64>, sum: Option<f64>) {
    let norm_term = match sum {
        Some(v) => v / repertoire.sum(),
        None => 1.0 / repertoire.sum(),
    };

    repertoire.apply(|x| x * norm_term);
}

fn calc_unconstrained_cause(purview: &BitBasis, tpm: &na::DMatrix<f64>) -> na::DVector<f64> {
    let ndim = tpm.nrows();
    let p = 1.0 / purview.image_size() as f64;
    na::DVector::<f64>::from_element(ndim, p)
}

fn calc_elementary_cause(purview: &BitBasis, mechanism: &BitBasis, current_state: usize, tpm: &na::DMatrix<f64>) -> na::DVector<f64> {
    let ndim = tpm.nrows();

    let c_mechanism = mechanism.generate_complement_basis();

    let mut accumulated = na::DVector::<f64>::zeros(ndim);
    c_mechanism.span(mechanism.fixed_state(current_state)).for_each(|col| {
        accumulated += tpm.column(col);
    });


    let c_purview = purview.generate_complement_basis();

    let mut marginal = na::DVector::<f64>::zeros(ndim);
    purview.span(0).for_each(|eq_class| {
        let mut marginal_rows = Vec::<usize>::with_capacity(c_purview.image_size());

        if c_purview.dim == 0 {
            marginal_rows.push(eq_class);
        } else {
            c_purview.span(eq_class).for_each(|row| {
                marginal_rows.push(row);
            });
        }

        let eq_prob = marginal_rows.iter().fold(0.0, |acc, &row| {
            acc + accumulated[row]
        });

        marginal_rows.iter().for_each(|&row| {
            marginal[row] = eq_prob;
        });
    });

    normalize_repertoire(&mut marginal, Some(c_purview.image_size() as f64));
    marginal
}

pub fn calc_cause_repertoire(purview: &BitBasis, mechanism: &BitBasis, current_state: usize, tpm: &na::DMatrix<f64>) -> na::DVector<f64> {
    if purview.dim == 0 {
        return na::DVector::<f64>::from_element(tpm.nrows(), 1.0);
    }

    match mechanism.dim {
        0 => calc_unconstrained_cause(purview, tpm),
        1 => calc_elementary_cause(purview, mechanism, current_state, tpm),
        _ => {
            let mut joint = calc_elementary_cause(purview, &mechanism.sub_basis(&[0]), current_state, tpm);

            (1..mechanism.dim).for_each(|i| {
                joint.component_mul_assign(&calc_elementary_cause(purview, &mechanism.sub_basis(&[i]), current_state, tpm));
            });

            normalize_repertoire(&mut joint, Some(purview.codim_image_size() as f64));
            joint
        }
    }
}

fn calc_elementary_effect(purview: &BitBasis, mechanism: &BitBasis, current_state: usize, tpm: &na::DMatrix<f64>) -> na::DVector<f64> {
    let ndim = tpm.nrows();

    let mut accumulated = na::DVector::<f64>::zeros(ndim).transpose();
    let c_mechanism = mechanism.generate_complement_basis();
    c_mechanism.span(mechanism.fixed_state(current_state)).for_each(|row| {
        accumulated += tpm.row(row);
    });


    let mut marginal = na::DVector::<f64>::zeros(ndim);
    let c_purview = purview.generate_complement_basis();
    purview.span(0).for_each(|eq_class| {
        let mut cols = Vec::<usize>::with_capacity(c_purview.image_size());
        c_purview.span(eq_class).for_each(|col| {
            cols.push(col);
        });

        let eq_prob = cols.iter().fold(0.0, |acc, &col| acc + accumulated[col]);

        cols.iter().for_each(|&col| {
            marginal[col] = eq_prob;
        });
    });

    normalize_repertoire(&mut marginal, Some(c_purview.image_size() as f64));
    marginal
}

pub fn calc_effect_repertoire(purview: &BitBasis, mechanism: &BitBasis, current_state: usize, tpm: &na::DMatrix<f64>) -> na::DVector<f64> {
    if purview.dim == 0 {
        return na::DVector::<f64>::from_element(tpm.nrows(), 1.0);
    }

    if purview.dim == 1 {
        calc_elementary_effect(purview, mechanism, current_state, tpm)
    } else {
        let mut joint = calc_elementary_effect(&purview.sub_basis(&[0]), mechanism, current_state, tpm);

        (1..purview.dim).for_each(|i| {
            joint.component_mul_assign(&calc_elementary_effect(&purview.sub_basis(&[i]), mechanism, current_state, tpm));
        });

        normalize_repertoire(&mut joint, Some(purview.codim_image_size() as f64));
        joint
    }
}
