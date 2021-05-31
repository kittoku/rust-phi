use std::usize;
use nalgebra as na;
use crate::bitwise::{change_bases, generate_bases_from_mask, get_masked_counter};


pub fn normalize_repertoire(repertoire: &mut na::DVector<f64>) {
    let norm_term = 1.0 / repertoire.sum();
    repertoire.apply(|x| x * norm_term);
}

fn calc_unconstrained_cause(tpm: &na::DMatrix<f64>) -> na::DVector<f64> {
    let ndim = tpm.ncols();
    let p = 1.0 / ndim as f64;
    na::DVector::<f64>::from_element(ndim, p)
}

fn calc_elementary_cause(past_mask: usize, current_mask: usize, current_state: usize, tpm: &na::DMatrix<f64>) -> na::DVector<f64> {
    let ndim = tpm.nrows();

    let column_indices = get_masked_counter((ndim - 1) & !current_mask , current_state & current_mask);
    let mut accumulated = na::DVector::<f64>::zeros(ndim);
    column_indices.for_each(|ncol| accumulated += tpm.column(ncol));

    let past_mask_size = past_mask.count_ones();
    let margined_size = 1 << past_mask_size;
    let mut margined = na::DVector::<f64>::zeros(margined_size);
    let margined_bases = generate_bases_from_mask(!(usize::MAX << past_mask_size));
    let original_bases = generate_bases_from_mask(past_mask);

    let mut eq_classes = Vec::<usize>::with_capacity(ndim);
    (0..ndim).for_each(|nrow| {
        eq_classes.push(change_bases(nrow, &original_bases, &margined_bases))
    });

    eq_classes.iter().enumerate().for_each(|(original_row, &margined_row)| {
        margined[margined_row] += accumulated[original_row];
    });


    let mut result = na::DVector::<f64>::zeros(ndim);
    eq_classes.iter().enumerate().for_each(|(result_row, &margined_row)| {
        result[result_row] += margined[margined_row];
    });


    normalize_repertoire(&mut result);
    result
}

pub fn calc_cause_repertoire(past_mask: usize, current_mask: usize, current_state: usize, tpm: &na::DMatrix<f64>) -> na::DVector<f64> {
    let past_mask_size = past_mask.count_ones();
    let current_mask_size = current_mask.count_ones();

   if past_mask_size == 0 || current_mask_size == 0 {
       calc_unconstrained_cause(tpm)
   } else {
       if current_mask_size == 1 {
           calc_elementary_cause(past_mask, current_mask, current_state, tpm)
       } else {
           let mut current_bases = generate_bases_from_mask(current_mask).into_iter();
           let mut joint = calc_elementary_cause(past_mask, current_bases.next().unwrap(), current_state, tpm);

           current_bases.for_each(|each_mask| {
               let each = calc_elementary_cause(past_mask, each_mask, current_state, tpm);
               joint.component_mul_assign(&each);
           });

           normalize_repertoire(&mut joint);
           joint
       }
   }
}
