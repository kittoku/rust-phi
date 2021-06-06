use nalgebra as na;
use crate::{bases::BitBases, emd::calc_emd_for_mechanism, partition::MechanismPartitionIterator, repertoire::{calc_cause_repertoire, calc_effect_repertoire}};


pub enum RepertoireType {
    CAUSE,
    EFFECT,
}

pub fn generate_all_repertoire_parts(repertoire_type: RepertoireType, current_state: usize, tpm: &na::DMatrix<f64>) -> na::DMatrix<f64> {
    let ndim = tpm.nrows();
    let combination_ndim = ndim * ndim;

    let mut result = na::DMatrix::<f64>::zeros(combination_ndim, ndim);

    let calc_repertoire = match repertoire_type {
        RepertoireType::CAUSE => calc_cause_repertoire,
        RepertoireType::EFFECT => calc_effect_repertoire,
    };

    let max_dim = (ndim - 1).count_ones() as usize;
    (0..ndim).for_each(|p| {
        (0..ndim).for_each(|m| {
            let purview = BitBases::construct_from_mask(p, max_dim);
            let mechanism = BitBases::construct_from_mask(m, max_dim);

            let repertoire = calc_repertoire(&purview, &mechanism, current_state, tpm);

            result.row_mut((p << max_dim) | m).tr_copy_from(&repertoire);
        });
    });

    result
}

#[derive(Debug)]
pub struct CoreRepertoire {
    pub purview: BitBases,
    pub repertoire: na::DVector<f64>,
    pub phi: f64,
}

pub fn construct_vector_from_row(row: usize, matrix: &na::DMatrix<f64>) -> na::DVector<f64> {
    na::DVector::<f64>::from_iterator(matrix.ncols(), matrix.row(row).iter().map(|x| *x))
}

pub fn search_core_with_parts(mechanism: &BitBases, parts: &na::DMatrix<f64>) -> Option<CoreRepertoire> {
    let mechanism_mask = mechanism.to_mask();

    let mut max_phi_repertoire: Option<CoreRepertoire> = None;

    for purview_mask in 0..mechanism.max_image_size() {
        let candidate = BitBases::construct_from_mask(purview_mask, mechanism.max_dim);
        if candidate.dim + mechanism.dim == 1 {
            // No possible partition
            continue;
        }

        let c_candidate = candidate.generate_complement_bases();

        let unconstrained_row = c_candidate.to_mask() << mechanism.max_dim;
        let unconstrained = construct_vector_from_row(unconstrained_row, parts);

        let criterion_row = (purview_mask << mechanism.max_dim) | mechanism_mask;
        let mut criterion = construct_vector_from_row(criterion_row, parts);
        criterion.component_mul_assign(&unconstrained);

        let mut min_emd = f64::INFINITY;

        let partitions = MechanismPartitionIterator::construct(candidate.dim, mechanism.dim);
        for partition in partitions {
            let left_purview_mask = candidate.sub_bases(&partition.left_purview).to_mask() << mechanism.max_dim;
            let right_purview_mask = candidate.sub_bases(&partition.right_purview).to_mask() << mechanism.max_dim;
            let left_mechanism_mask = mechanism.sub_bases(&partition.left_mechanism).to_mask();
            let right_mechanism_mask = mechanism.sub_bases(&partition.right_mechanism).to_mask();

            let mut joint = unconstrained.clone();
            joint.component_mul_assign(&construct_vector_from_row(left_purview_mask | left_mechanism_mask, parts));
            joint.component_mul_assign(&construct_vector_from_row(right_purview_mask | right_mechanism_mask, parts));

            let emd = calc_emd_for_mechanism(&criterion, &joint).objective();

            if emd < min_emd {
                min_emd = emd;
            }

            if min_emd == 0.0 {
                break;
            }
        }

        if min_emd != 0.0 {
            let update = if let Some(v) = &max_phi_repertoire {
                if min_emd > v.phi {
                    true
                } else {
                    false
                }
            } else {
                true
            };

            if update {
                max_phi_repertoire = Some(CoreRepertoire {
                    purview: candidate,
                    repertoire: criterion,
                    phi: min_emd,
                });
            }
        }
    };

    max_phi_repertoire
}

#[derive(Debug)]
pub struct Concept {
    pub mechanism: BitBases,
    pub core_cause: CoreRepertoire,
    pub core_effect: CoreRepertoire,
}

pub fn search_concept_with_parts(mechanism: &BitBases, cause_parts: &na::DMatrix<f64>, effect_parts: &na::DMatrix<f64>) -> Option<Concept> {
    let core_cause = if let Some(v) = search_core_with_parts(mechanism, cause_parts) {
        v
    } else {
        return None;
    };

    let core_effect = if let Some(v) = search_core_with_parts(mechanism, effect_parts) {
        v
    } else {
        return None;
    };

    Some(Concept {
        mechanism: mechanism.clone(),
        core_cause: core_cause,
        core_effect: core_effect,
    })
}
