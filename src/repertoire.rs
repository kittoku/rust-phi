use std::usize;
use nalgebra as na;
use crate::bases::BitBases;


pub fn normalize_repertoire(repertoire: &mut na::DVector<f64>, sum: Option<f64>) {
    let norm_term = match sum {
        Some(v) => v / repertoire.sum(),
        None => 1.0 / repertoire.sum(),
    };

    repertoire.apply(|x| x * norm_term);
}

fn calc_unconstrained_cause(past_bases: &BitBases, tpm: &na::DMatrix<f64>) -> na::DVector<f64> {
    let ndim = tpm.nrows();
    let p = 1.0 / past_bases.image_size() as f64;
    na::DVector::<f64>::from_element(ndim, p)
}

fn calc_elementary_cause(past_bases: &BitBases, current_bases: &BitBases, current_state: usize, tpm: &na::DMatrix<f64>) -> na::DVector<f64> {
    let ndim = tpm.nrows();

    let c_current_bases = current_bases.generate_complement_bases();

    let mut accumulated = na::DVector::<f64>::zeros(ndim);
    let fixed_current = current_bases.bases[0] & current_state;
    c_current_bases.span(fixed_current).for_each(|col| {
        accumulated += tpm.column(col);
    });


    let c_past_bases = past_bases.generate_complement_bases();

    let mut marginal = na::DVector::<f64>::zeros(ndim);
    past_bases.span(0).for_each(|eq_class| {
        let mut marginal_rows = Vec::<usize>::with_capacity(c_past_bases.image_size());

        if c_past_bases.dim == 0 {
            marginal_rows.push(eq_class);
        } else {
            c_past_bases.span(eq_class).for_each(|row| {
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

    normalize_repertoire(&mut marginal, Some(c_past_bases.image_size() as f64));
    marginal
}

pub fn calc_cause_repertoire(past_bases: &BitBases, current_bases: &BitBases, current_state: usize, tpm: &na::DMatrix<f64>) -> na::DVector<f64> {
    if past_bases.dim == 0 {
        return na::DVector::<f64>::from_element(tpm.nrows(), 1.0);
    }

    match current_bases.dim {
        0 => calc_unconstrained_cause(past_bases, tpm),
        1 => calc_elementary_cause(past_bases, current_bases, current_state, tpm),
        _ => {
            let mut joint = calc_elementary_cause(past_bases, &current_bases.sub_bases(&[0]), current_state, tpm);

            (1..current_bases.dim).for_each(|i| {
                joint.component_mul_assign(&calc_elementary_cause(past_bases, &current_bases.sub_bases(&[i]), current_state, tpm));
            });

            normalize_repertoire(&mut joint, Some(past_bases.codim_image_size() as f64));
            joint
        }
    }
}

fn calc_elementary_effect(future_bases: &BitBases, current_bases: &BitBases, current_state: usize, tpm: &na::DMatrix<f64>) -> na::DVector<f64> {
    let ndim = tpm.nrows();

    let mut accumulated = na::DVector::<f64>::zeros(ndim).transpose();
    let c_current_bases = current_bases.generate_complement_bases();
    c_current_bases.span(current_bases.fixed_state(current_state)).for_each(|row| {
        accumulated += tpm.row(row);
    });


    let mut marginal = na::DVector::<f64>::zeros(ndim);
    let c_future_bases = future_bases.generate_complement_bases();
    future_bases.span(0).for_each(|eq_class| {
        let mut cols = Vec::<usize>::with_capacity(c_future_bases.image_size());
        c_future_bases.span(eq_class).for_each(|col| {
            cols.push(col);
        });

        let eq_prob = cols.iter().fold(0.0, |acc, &col| acc + accumulated[col]);

        cols.iter().for_each(|&col| {
            marginal[col] = eq_prob;
        });
    });

    normalize_repertoire(&mut marginal, Some(c_future_bases.image_size() as f64));
    marginal
}

pub fn calc_effect_repertoire(future_bases: &BitBases, current_bases: &BitBases, current_state: usize, tpm: &na::DMatrix<f64>) -> na::DVector<f64> {
    if future_bases.dim == 0 {
        return na::DVector::<f64>::from_element(tpm.nrows(), 1.0);
    }

    if future_bases.dim == 1 {
        calc_elementary_effect(future_bases, current_bases, current_state, tpm)
    } else {
        let mut joint = calc_elementary_effect(&future_bases.sub_bases(&[0]), current_bases, current_state, tpm);

        (1..future_bases.dim).for_each(|i| {
            joint.component_mul_assign(&calc_elementary_effect(&future_bases.sub_bases(&[i]), current_bases, current_state, tpm));
        });

        normalize_repertoire(&mut joint, Some(future_bases.codim_image_size() as f64));
        joint
    }
}
