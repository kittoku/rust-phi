use std::{iter::Rev, ops::Range, sync::{Arc, Mutex}, thread::{self, JoinHandle}};

use nalgebra as na;
use crate::{basis::BitBasis, bitwise::USIZE_BASIS, compare::{Comparison, compare_roughly}, emd::calc_constellation_emd, mechanism::{Concept, CoreRepertoire, construct_vector_from_row, generate_all_repertoire_parts, search_concept_with_parts}, partition::{MechanismPartition, SystemPartition, SystemPartitionIterator}, tpm::{calc_fixed_marginal_tpm, calc_partitioned_marginal_tpm}};


#[derive(Debug)]
pub struct MinimumInformationPartition {
    pub partition: SystemPartition,
    pub phi: f64,
}

#[derive(Debug)]
pub struct Constellation {
    pub concepts: Vec<Concept>,
    pub null_concept: Concept,
    pub mip: MinimumInformationPartition,
}

pub fn search_constellation_with_parts(cause_parts: &na::DMatrix<f64>, effect_parts: &na::DMatrix<f64>) -> Constellation {
    let system_basis = BitBasis::construct_from_max_image_size(cause_parts.ncols());
    let mut concepts = Vec::<Concept>::new();

    (1..system_basis.max_image_size()).for_each(|mask| {
        let mechanism = BitBasis::construct_from_mask(mask, system_basis.max_dim);

        let concept = search_concept_with_parts(&mechanism, cause_parts, effect_parts);
        if concept.phi > 0.0 {
            concepts.push(concept);
        };
    });

    let unconstrained_mask = system_basis.to_mask() << system_basis.max_dim;
    let unconstrained_cause = construct_vector_from_row(unconstrained_mask, cause_parts);
    let unconstrained_effect = construct_vector_from_row(unconstrained_mask, effect_parts);

    let null_concept = Concept {
        mechanism: BitBasis::null_basis(system_basis.max_dim),
        core_cause: CoreRepertoire {
            purview: BitBasis::null_basis(system_basis.max_dim),
            repertoire: unconstrained_cause,
            partition: MechanismPartition::null_partition(),
            phi: 0.0,
        },
        core_effect: CoreRepertoire {
            purview: BitBasis::null_basis(system_basis.max_dim),
            repertoire: unconstrained_effect,
            partition: MechanismPartition::null_partition(),
            phi: 0.0,
        },
        phi: 0.0,
    };

    Constellation {
        concepts: concepts,
        null_concept: null_concept,
        mip: MinimumInformationPartition {
            partition: SystemPartition::null_partition(),
            phi: 0.0,
        },
    }
}

pub fn search_constellation_with_mip(current_state: usize, tpm: &na::DMatrix<f64>) -> Constellation {
    let system_basis = BitBasis::construct_from_max_image_size(tpm.ncols());

    let cause_parts = generate_all_repertoire_parts(crate::mechanism::RepertoireType::CAUSE, current_state, &tpm);
    let effect_parts = generate_all_repertoire_parts(crate::mechanism::RepertoireType::EFFECT, current_state, &tpm);
    let mut criterion = search_constellation_with_parts(&cause_parts, &effect_parts);
    criterion.mip = MinimumInformationPartition {
        partition: SystemPartition::null_partition(),
        phi: f64::INFINITY,
    };

    let partitions = SystemPartitionIterator::construct(system_basis.max_dim);
    for partition in partitions {
        let partitioned_tpm = calc_partitioned_marginal_tpm(&partition, &tpm);
        let partitioned_cause_parts = generate_all_repertoire_parts(crate::mechanism::RepertoireType::CAUSE, current_state, &partitioned_tpm);
        let partitioned_effect_parts = generate_all_repertoire_parts(crate::mechanism::RepertoireType::EFFECT, current_state, &partitioned_tpm);
        let partitioned = search_constellation_with_parts(&partitioned_cause_parts, &partitioned_effect_parts);

        let emd = calc_constellation_emd(&criterion, &partitioned);

        if emd < criterion.mip.phi {
            criterion.mip.partition = partition;
            criterion.mip.phi = emd;
        }

        if let Comparison::AlmostEqual = compare_roughly(emd, 0.0) {
            criterion.mip.phi = 0.0;
            break;
        }
    }

    if criterion.mip.phi == f64::INFINITY { // no possible partition found
        criterion.mip.phi = 0.0;
    }

    criterion
}

#[derive(Debug)]
pub struct Complex {
    pub elements: Vec<usize>,
    pub marginal_tpm: na::DMatrix<f64>,
    pub constellation: Constellation
}

fn get_assigned_mask(masks: &Arc<Mutex<Rev<Range<usize>>>>) -> Option<usize> {
    masks.lock().unwrap().next()
}

fn notify_progress(complex: &Complex, current_count: &Arc<Mutex<usize>>, total_count: usize) {
    let mut locked = current_count.lock().unwrap();
    *locked += 1;

    let progress = format!("PROGRESS={}/{}", locked, total_count);
    let candidate = format!("CANDIDATE={:?}", complex.elements);
    let phi = format!("BIG_PHI={}", complex.constellation.mip.phi);

    println!("{}, {}, {}", progress, candidate, phi);
}

fn challenge_complex(complex: Complex, maximum: &Arc<Mutex<Option<Complex>>>) {
    let mut locked = maximum.lock().unwrap();

    let update = if let Some(v) = locked.as_ref() {
        complex.constellation.mip.phi > v.constellation.mip.phi
    } else {
        true
    };

    if update {
        *locked = Some(complex);
    };
}

pub fn search_complex(current_state: usize, tpm: na::DMatrix<f64>, num_threads: usize, log: bool) -> Complex {
    let system_basis = Arc::new(BitBasis::construct_from_max_image_size(tpm.ncols()));
    let max_image_size = system_basis.max_image_size();

    let shared_tpm = Arc::new(tpm);
    let candidate_masks = Arc::new(Mutex::new((1..max_image_size).rev()));
    let current_complex = Arc::new(Mutex::new(None));
    let current_count = Arc::new(Mutex::new(0));
    let total_count = max_image_size - 1;

    let mut handles = Vec::<JoinHandle<()>>::new();

    (0..num_threads).for_each(|_| {
        let cloned_basis = system_basis.clone();
        let cloned_tpm = shared_tpm.clone();
        let cloned_masks = candidate_masks.clone();
        let cloned_complex = current_complex.clone();
        let cloned_count = current_count.clone();

        let handle = thread::spawn(move || {
            loop {
                if let Some(mask) = get_assigned_mask(&cloned_masks) {
                    let candidate_elements: Vec<usize> = (0..cloned_basis.max_dim).filter(|&i| mask & USIZE_BASIS[i] != 0).collect();
                    let candidate_basis = cloned_basis.sub_basis(candidate_elements.as_slice());

                    let marginal = calc_fixed_marginal_tpm(&candidate_basis, current_state, &cloned_tpm);

                    let constellation = search_constellation_with_mip(current_state, &marginal);
                    let complex = Complex {
                        elements: candidate_elements,
                        marginal_tpm: marginal,
                        constellation: constellation,
                    };

                    if log {
                        notify_progress(&complex, &cloned_count, total_count)
                    };

                    challenge_complex(complex, &cloned_complex);
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
    };


    Arc::try_unwrap(current_complex).unwrap().into_inner().unwrap().unwrap()
}
