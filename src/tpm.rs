use std::{sync::{Arc, Mutex}, thread::{self, JoinHandle}, usize};
use nalgebra as na;
use crate::{bitwise::{USIZE_BITS, change_bases, generate_bases_from_mask, get_masked_counter}, link_fn::LinkFn};


struct RowCounter {
    current: usize,
    size: usize,
}

impl RowCounter {
    fn get_nrow(&mut self) -> Option<usize> {
        if self.current >= self.size {
            return None
        }

        let nrow = self.current;

        self.current += 1;

        Some(nrow)
    }
}

fn get_assigned_nrow(counter: &Arc<Mutex<RowCounter>>) -> Option<usize> {
    counter.lock().unwrap().get_nrow()
}

fn update_row(nrow: usize, row: &na::DMatrix::<f64>, matrix: &Arc<Mutex<na::DMatrix<f64>>>) {
    let mut unwrapped = matrix.lock().unwrap();
    let mut target = unwrapped.row_mut(nrow);

    target.copy_from(row);
}

fn predict_each_elements(env: usize, fns: &Vec<(LinkFn, usize)>, probs: &mut na::DMatrix<f64>) {
    fns.iter().zip(probs.row_iter_mut()).for_each(|((link_fn, mask) ,mut row)| {
        row[1] = link_fn(env, *mask);
        row[0] = 1.0 - row[1]
    });
}

fn calc_joint_prob(state: usize, probs: &na::DMatrix<f64>) -> f64 {
    let mut prob = 1.0;
    let mut mask = 1;

    probs.row_iter().for_each(|row| {
        prob *= if state & mask == 0 {
            row[0]
        } else {
            row[1]
        };

        mask <<= 1;
    });

    prob
}

pub fn calculate_tpm(fns: Vec<(LinkFn, usize)>, num_threads: usize) -> na::DMatrix<f64> {
    let element_size: usize = fns.len();
    let matrix_size: usize = 1 << element_size;

    let counter = RowCounter { current: 0, size: matrix_size };
    let tpm: na::DMatrix<f64> = na::DMatrix::<f64>::from_element(matrix_size, matrix_size, 1.0);

    let shared_fns = Arc::new(fns);
    let shared_counter = Arc::new(Mutex::new(counter));
    let shared_tpm = Arc::new(Mutex::new(tpm));

    let mut handles = Vec::<JoinHandle<()>>::new();

    (0..num_threads).for_each(|_| {
        let cloned_fns = Arc::clone(&shared_fns);
        let cloned_counter = Arc::clone(&shared_counter);
        let cloned_tpm  = Arc::clone(&shared_tpm);

        let handle = thread::spawn(move || {
            let mut row_buffer = na::DMatrix::<f64>::from_element(1, matrix_size, 0.0);
            let mut prob_buffer = na::DMatrix::<f64>::from_element(element_size, 2, 0.0);

            loop {
                if let Some(env) = get_assigned_nrow(&cloned_counter) {
                    predict_each_elements(env, &cloned_fns, &mut prob_buffer);
                    (0..matrix_size).for_each(|state| {
                        row_buffer[state] = calc_joint_prob(state, &prob_buffer);
                    });

                    update_row(env, &row_buffer, &cloned_tpm);
                } else {
                    break;
                }
            };
        });

        handles.push(handle);
    });

    loop {
        if let Some(handle) = handles.pop() {
            handle.join().unwrap();
        } else {
            break;
        }
    }

    Arc::try_unwrap(shared_tpm).unwrap().into_inner().unwrap()
}

pub fn margin_tpm(state: usize, mask: usize, matrix: &na::DMatrix<f64>) -> na::DMatrix<f64> {
    let mask_size = mask.count_ones() as usize;
    let margined_size = 1 << mask_size;
    let fixed = state & !mask;

    let mut margined = na::DMatrix::<f64>::from_element(margined_size, margined_size, 0.0);

    let row_counter = get_masked_counter(mask, fixed);
    let mask_bases = generate_bases_from_mask(mask);
    let margined_bases = generate_bases_from_mask(usize::MAX >> USIZE_BITS - mask_size);


    row_counter.enumerate().for_each(|(margined_nrow, original_nrow)| {
        let mut margined_row = margined.row_mut(margined_nrow);
        let original_row = matrix.row(original_nrow);

        (0..original_row.ncols()).for_each(|original_ncol| {
            let eq_class = change_bases(original_ncol, &mask_bases, &margined_bases);
            margined_row[eq_class] += original_row[original_ncol];
        });
    });

    margined
}
