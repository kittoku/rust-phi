use std::{sync::{Arc, Mutex}, thread::{self, JoinHandle}, time::SystemTime};

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

fn get_assigned_partition(partitions: &Arc<Mutex<SystemPartitionIterator>>) -> Option<SystemPartition> {
    partitions.lock().unwrap().next()
}

fn challenge_update(emd: f64, partition: SystemPartition, mip: &Arc<Mutex<MinimumInformationPartition>>) -> bool {
    // return false if MIP can fully reduce the system
    let mut locked = mip.lock().unwrap();

    if emd < locked.phi {
        locked.partition = partition;
        locked.phi = emd;
    };

    if let Comparison::AlmostEqual = compare_roughly(locked.phi, 0.0) {
        locked.phi = 0.0;
        false
    } else {
        true
    }
}

pub fn search_constellation_with_mip(current_state: usize, tpm: &Arc<na::DMatrix<f64>>, num_threads: usize) -> Constellation {
    let system_basis = BitBasis::construct_from_max_image_size(tpm.ncols());

    let cause_parts = Arc::new(generate_all_repertoire_parts(crate::mechanism::RepertoireType::CAUSE, current_state, &tpm));
    let effect_parts = Arc::new(generate_all_repertoire_parts(crate::mechanism::RepertoireType::EFFECT, current_state, &tpm));
    let criterion = Arc::new(search_constellation_with_parts(&cause_parts, &effect_parts));

    let mip = Arc::new(Mutex::new(MinimumInformationPartition {
        partition: SystemPartition::null_partition(),
        phi: f64::INFINITY,
    }));

    let partitions = Arc::new(Mutex::new(SystemPartitionIterator::construct(system_basis.max_dim)));

    let mut handles = Vec::<JoinHandle<()>>::new();

    (0..num_threads).for_each(|_| {
        let cloned_tpm = tpm.clone();
        let cloned_criterion = criterion.clone();
        let cloned_mip = mip.clone();
        let cloned_partitions = partitions.clone();

        let handle = thread::spawn(move || {
            loop {
                if let Some(partition) = get_assigned_partition(&cloned_partitions) {
                    let partitioned_tpm = calc_partitioned_marginal_tpm(&partition, &cloned_tpm);
                    let partitioned_cause_parts = generate_all_repertoire_parts(crate::mechanism::RepertoireType::CAUSE, current_state, &partitioned_tpm);
                    let partitioned_effect_parts = generate_all_repertoire_parts(crate::mechanism::RepertoireType::EFFECT, current_state, &partitioned_tpm);
                    let partitioned = search_constellation_with_parts(&partitioned_cause_parts, &partitioned_effect_parts);

                    let emd = calc_constellation_emd(&cloned_criterion, &partitioned);

                    if !challenge_update(emd, partition, &cloned_mip) {
                        break;
                    };
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

    let mut final_mip = Arc::try_unwrap(mip).unwrap().into_inner().unwrap();
    if final_mip.phi == f64::INFINITY { // no possible partition found
        final_mip.phi = 0.0;
    }

    let mut unwrapped = Arc::try_unwrap(criterion).unwrap();
    unwrapped.mip = final_mip;
    unwrapped
}

#[derive(Debug)]
pub struct Complex {
    pub elements: Vec<usize>,
    pub marginal_tpm: na::DMatrix<f64>,
    pub constellation: Constellation
}

fn notify_progress(candidate: &Vec<usize>, phi: f64, current_count: usize, total_count: usize, start_time: SystemTime) {
    let progress = format!("PROGRESS={}/{}", current_count, total_count);
    let candidate = format!("CANDIDATE={:?}", candidate);
    let phi = format!("BIG_PHI={}", phi);
    let time = format!("TIME={:.2e}", start_time.elapsed().unwrap().as_secs_f64());

    println!("{}, {}, {}, {}", progress, candidate, phi, time);
}

pub fn search_complex(current_state: usize, tpm: &na::DMatrix<f64>, num_threads: usize, log: bool) -> Complex {
    let system_basis = BitBasis::construct_from_max_image_size(tpm.ncols());
    let max_image_size = system_basis.max_image_size();

    let total_count = max_image_size - 1;

    let mut current_complex: Option<Complex> = None;

    (1..max_image_size).for_each(|mask| {
        let start_time = SystemTime::now();

        let candidate_elements: Vec<usize> = (0..system_basis.max_dim).filter(|&i| mask & USIZE_BASIS[i] != 0).collect();
        let candidate_basis = system_basis.sub_basis(candidate_elements.as_slice());

        let marginal = Arc::new(calc_fixed_marginal_tpm(&candidate_basis, current_state, &tpm));

        let constellation = search_constellation_with_mip(current_state, &marginal, num_threads);

        if log {
            notify_progress(&&candidate_elements, constellation.mip.phi, mask, total_count, start_time);
        };

        let update = if let Some(complex) = &current_complex {
            constellation.mip.phi > complex.constellation.mip.phi
        } else {
            true
        };

        if update {
            current_complex = Some(Complex {
                elements: candidate_elements,
                marginal_tpm: Arc::try_unwrap(marginal).unwrap(),
                constellation: constellation,
            });
        };
    });

    current_complex.unwrap()
}
