use nalgebra as na;
use crate::{basis::BitBasis, compare::{Comparison, compare_roughly}, emd::calc_constellation_emd, mechanism::{Concept, CoreRepertoire, construct_vector_from_row, generate_all_repertoire_parts, search_concept_with_parts}, partition::{MechanismPartition, SystemPartition, SystemPartitionIterator}, tpm::calc_partitioned_marginal_tpm};


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
