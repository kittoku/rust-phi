use std::{sync::{Arc, Mutex}, thread::{self, JoinHandle}, usize};
use nalgebra as na;
use crate::{basis::BitBasis, link_fn::LinkFn, partition::SystemPartition};


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

pub fn calc_fixed_marginal_tpm(surviving_basis: &BitBasis, state: usize, tpm: &na::DMatrix<f64>) -> na::DMatrix<f64> {
    let c_basis = surviving_basis.generate_complement_basis();

    let maginal_dim = surviving_basis.image_size();
    let mut marginal = na::DMatrix::<f64>::zeros(maginal_dim, maginal_dim);

    surviving_basis.span(c_basis.fixed_state(state)).enumerate().for_each(|(marginal_row, original_row)| {
        surviving_basis.span(0).enumerate().for_each(|(marginal_col, eq_class)| {
            let mut marginal_vec = marginal.row_mut(marginal_row);
            let original_vec = tpm.row(original_row);

            marginal_vec[marginal_col] = c_basis.span(eq_class).fold(0.0, |acc, original_col| {
                acc + original_vec[original_col]
            });
        });
    });

    marginal
}

pub fn calc_elementary_marginal_tpm(target_basis: &BitBasis, surviving_basis: &BitBasis, tpm: &na::DMatrix<f64>) -> na::DMatrix<f64> {
    let target_vector = target_basis.vectors[0];
    let c_target_basis = target_basis.generate_complement_basis();
    let c_surviving_basis = surviving_basis.generate_complement_basis();

    let marginal_dim = tpm.nrows();
    let mut marginal = na::DMatrix::<f64>::zeros(marginal_dim, marginal_dim);

    let mut acc = na::DVector::<f64>::zeros(marginal_dim).transpose();

    surviving_basis.span(0).for_each(|eq_class| {
        acc.fill(0.0);

        c_surviving_basis.span(eq_class).for_each(|row| {
            acc += tpm.row(row);
        });

        let mut p_0 = c_target_basis.span(0).fold(0.0, |p, col| {
            p + acc[col]
        });

        let mut p_1 = c_target_basis.span(target_vector).fold(0.0, |p, col| {
            p + acc[col]
        });

        let norm_term = 1.0 / (p_0 + p_1);

        p_0 *= norm_term;
        p_1 *= norm_term;


        surviving_basis.span(0).for_each(|_eq_class| {
            c_surviving_basis.span(eq_class).for_each(|_row| {
                let mut marginal_row = marginal.row_mut(_row);

                c_target_basis.span(0).for_each(|col| {
                    marginal_row[col] = p_0;
                });

                c_target_basis.span(target_vector).for_each(|col| {
                    marginal_row[col] = p_1;
                });
            });
        });
    });

    marginal
}

pub fn calc_partitioned_marginal_tpm(partition: &SystemPartition, tpm: &na::DMatrix<f64>) -> na::DMatrix<f64> {
    let system_basis = BitBasis::construct_from_max_image_size(tpm.nrows());
    let template_basis = system_basis.sub_basis(&[partition.cut_from[0]]);
    let mut marginal = calc_elementary_marginal_tpm(&template_basis, &system_basis, tpm);

    (1..partition.cut_from.len()).for_each(|i| {
        let intact_basis = system_basis.sub_basis(&[partition.cut_from[i]]);
        marginal.component_mul_assign(&calc_elementary_marginal_tpm(&intact_basis, &system_basis, tpm));
    });

    let isolated_basis = system_basis.sub_basis(&partition.cut_to);
    (0..partition.cut_to.len()).for_each(|i| {
        let noised_basis = system_basis.sub_basis(&[partition.cut_to[i]]);
        marginal.component_mul_assign(&calc_elementary_marginal_tpm(&noised_basis, &isolated_basis, tpm))
    });

    marginal
}
