use nalgebra as na;
use crate::{basis::BitBasis, emd::calc_repertoire_emd, partition::MechanismPartitionIterator, repertoire::{calc_cause_repertoire, calc_effect_repertoire}};


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
            let purview = BitBasis::construct_from_mask(p, max_dim);
            let mechanism = BitBasis::construct_from_mask(m, max_dim);

            let repertoire = calc_repertoire(&purview, &mechanism, current_state, tpm);

            result.row_mut((p << max_dim) | m).tr_copy_from(&repertoire);
        });
    });

    result
}

#[derive(Debug)]
pub struct CoreRepertoire {
    pub purview: BitBasis,
    pub repertoire: na::DVector<f64>,
    pub phi: f64,
}

pub fn construct_vector_from_row(row: usize, matrix: &na::DMatrix<f64>) -> na::DVector<f64> {
    na::DVector::<f64>::from_iterator(matrix.ncols(), matrix.row(row).iter().map(|x| *x))
}

pub fn search_core_with_parts(mechanism: &BitBasis, parts: &na::DMatrix<f64>) -> Option<CoreRepertoire> {
    let mechanism_mask = mechanism.to_mask();

    let mut max_phi_repertoire: Option<CoreRepertoire> = None;
    let mut max_null_distance = 0.0;

    let unconstrained_row = !(usize::MAX << mechanism.max_dim) << mechanism.max_dim;
    let unconstrained = construct_vector_from_row(unconstrained_row, parts);

    for purview_mask in 0..mechanism.max_image_size() {
        let candidate = BitBasis::construct_from_mask(purview_mask, mechanism.max_dim);
        if candidate.dim + mechanism.dim == 1 {
            // No possible partition
            continue;
        }

        let c_candidate = candidate.generate_complement_basis();

        let unconstrained_part_row = c_candidate.to_mask() << mechanism.max_dim;
        let unconstrained_part = construct_vector_from_row(unconstrained_part_row, parts);

        let criterion_row = (purview_mask << mechanism.max_dim) | mechanism_mask;
        let mut criterion = construct_vector_from_row(criterion_row, parts);
        criterion.component_mul_assign(&unconstrained_part);

        let mut min_emd = f64::INFINITY;
        let mut null_distance: Option<f64> = None;

        let partitions = MechanismPartitionIterator::construct(candidate.dim, mechanism.dim);
        for partition in partitions {
            let left_purview_mask = candidate.sub_basis(&partition.left_purview).to_mask() << mechanism.max_dim;
            let right_purview_mask = candidate.sub_basis(&partition.right_purview).to_mask() << mechanism.max_dim;
            let left_mechanism_mask = mechanism.sub_basis(&partition.left_mechanism).to_mask();
            let right_mechanism_mask = mechanism.sub_basis(&partition.right_mechanism).to_mask();

            let mut joint = unconstrained_part.clone();
            joint.component_mul_assign(&construct_vector_from_row(left_purview_mask | left_mechanism_mask, parts));
            joint.component_mul_assign(&construct_vector_from_row(right_purview_mask | right_mechanism_mask, parts));

            let emd = calc_repertoire_emd(&criterion, &joint);

            if emd < min_emd {
                min_emd = emd;
            }

            if min_emd == 0.0 {
                break;
            }
        }

        if min_emd != 0.0 {
            let update = if let Some(v) = &max_phi_repertoire {
                match min_emd {
                    x if x > v.phi => true,
                    x if x == v.phi && {
                        null_distance = Some(calc_repertoire_emd(&criterion, &unconstrained));
                        null_distance.unwrap() > max_null_distance
                    }  => true,
                    _ => false,
                }
            } else {
                true
            };

            if update {
                max_null_distance = if let Some(v) = null_distance {
                    v
                } else {
                    calc_repertoire_emd(&criterion, &unconstrained)
                };

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
    pub mechanism: BitBasis,
    pub core_cause: CoreRepertoire,
    pub core_effect: CoreRepertoire,
    pub phi: f64,
}

impl Concept {
    pub fn distance_from(&self, other: &Concept) -> f64 {
        let mut distance = calc_repertoire_emd(&self.core_cause.repertoire, &other.core_cause.repertoire);

        distance += calc_repertoire_emd(&self.core_effect.repertoire, &other.core_effect.repertoire);

        distance
    }
}

pub fn search_concept_with_parts(mechanism: &BitBasis, cause_parts: &na::DMatrix<f64>, effect_parts: &na::DMatrix<f64>) -> Option<Concept> {
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

    let phi = core_cause.phi.min(core_effect.phi);

    Some(Concept {
        mechanism: mechanism.clone(),
        core_cause: core_cause,
        core_effect: core_effect,
        phi: phi,
    })
}

#[derive(Debug)]
pub struct Constellation {
    pub concepts: Vec<Concept>,
    pub null_concept: Concept,
}

pub fn search_constellation_with_parts(system_basis: &BitBasis, cause_parts: &na::DMatrix<f64>, effect_parts: &na::DMatrix<f64>) -> Constellation {
    let mut concepts = Vec::<Concept>::new();

    (1..system_basis.max_image_size()).for_each(|mask| {
        let mechanism = BitBasis::construct_from_mask(mask, system_basis.max_dim);

        if let Some(concept) = search_concept_with_parts(&mechanism, cause_parts, effect_parts) {
            concepts.push(concept);
        }
    });

    let unconstrained_mask = system_basis.to_mask() << system_basis.max_dim;
    let unconstrained_cause = construct_vector_from_row(unconstrained_mask, cause_parts);
    let unconstrained_effect = construct_vector_from_row(unconstrained_mask, effect_parts);

    let null_concept = Concept {
        mechanism: BitBasis::null_basis(system_basis.max_dim),
        core_cause: CoreRepertoire {
            purview: BitBasis::null_basis(system_basis.max_dim),
            repertoire: unconstrained_cause,
            phi: 0.0,
        },
        core_effect: CoreRepertoire {
            purview: BitBasis::null_basis(system_basis.max_dim),
            repertoire: unconstrained_effect,
            phi: 0.0,
        },
        phi: 0.0,
    };

    Constellation {
        concepts: concepts,
        null_concept: null_concept,
    }
}
