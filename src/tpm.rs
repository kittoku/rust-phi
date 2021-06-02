use std::{sync::{Arc, Mutex}, thread::{self, JoinHandle}, usize};
use nalgebra as na;
use crate::{bases::BitBases, link_fn::LinkFn};


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

pub fn calc_tpm(fns: Vec<(LinkFn, usize)>, num_threads: usize) -> na::DMatrix<f64> {
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

pub fn marginalize_tpm(surviving_bases: &BitBases, state: usize, tpm: &na::DMatrix<f64>) -> na::DMatrix<f64> {
    let c_bases = surviving_bases.generate_complement_bases();

    let maginal_dim = surviving_bases.image_size();
    let mut marginal = na::DMatrix::<f64>::zeros(maginal_dim, maginal_dim);

    let fixed_state = c_bases.bases.iter().fold(0, |acc, &base| {
        acc | (state & base)
    });

    surviving_bases.span(fixed_state).enumerate().for_each(|(marginal_row, original_row)| {
        surviving_bases.span(0).enumerate().for_each(|(marginal_col, eq_class)| {
            let mut marginal_vec = marginal.row_mut(marginal_row);
            let original_vec = tpm.row(original_row);

            marginal_vec[marginal_col] = c_bases.span(eq_class).fold(0.0, |acc, original_col| {
                acc + original_vec[original_col]
            });
        });
    });

    marginal
}
