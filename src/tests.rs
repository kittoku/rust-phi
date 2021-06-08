use std::usize;
use nalgebra as na;
use crate::{basis::BitBasis, emd::calc_emd_for_mechanism, mechanism::{generate_all_repertoire_parts, search_concept_with_parts}, repertoire::{calc_cause_repertoire, calc_effect_repertoire, normalize_repertoire}};


const EPSILON : f64 = 1.0e-7;

fn notify_pass(case_number: usize) {
    println!("CASE_{} ... PASSED", case_number)
}

fn assert_almost_equal_scalar(actual: f64, expected: f64) {
    let diff = (actual - expected).abs();

    if diff > EPSILON {
        let mut message = String::from("The given scalars are too different: \n");
        message += &format!("actual -> {:?}\n", actual);
        message += &format!("expected -> {:?}\n", expected);
        message += &format!("diff -> {:?}\n", diff);
        panic!("{}", message);
    }
}

fn assert_almost_equal_vec(actual: &na::DVector<f64>, expected: &na::DVector<f64>) {
    let diff = (actual - expected).abs();

    diff.iter().for_each(|&x| {
        if x > EPSILON {
            let mut message = String::from("The given vectors are too different: \n");
            message += &format!("actual -> {:?}\n", actual);
            message += &format!("expected -> {:?}\n", expected);
            message += &format!("diff -> {:?}\n", diff);
            panic!("{}", message);
        }
    });
}

#[test]
fn test_calc_emd_for_mechanism() {
    let mut vec_from = Vec::<na::DVector<f64>>::new();
    let mut vec_to = Vec::<na::DVector<f64>>::new();
    let mut expected = Vec::<f64>::new();

    // CASE 0
    vec_from.push(na::DVector::<f64>::from_element(8, 1.0 / 8.0));
    vec_to.push(na::DVector::<f64>::from_element(8, 1.0 / 8.0));
    expected.push(0.0);

    // CASE 1
    let mut from_1 = na::DVector::<f64>::zeros(8);
    from_1[7] = 1.0;
    vec_from.push(from_1);
    vec_to.push(na::DVector::<f64>::from_element(8, 1.0 / 8.0));
    expected.push(1.5);

    // CASE 2
    let mut from_2 = na::DVector::<f64>::from_element(8, 1.0 / 7.0);
    from_2[7] = 0.0;
    vec_from.push(from_2);
    vec_to.push(na::DVector::<f64>::from_element(8, 1.0 / 8.0));
    let earth_unit = 1.0 / 56.0;
    expected.push(12.0 * earth_unit); // 3 * 1 * earth_unit + 3 * 2 * earth_unit + 1 * 3 * earth_unit


    expected.into_iter().enumerate().for_each(|(i, e)| {
        let actual = calc_emd_for_mechanism(&vec_from[i], &vec_to[i]).objective();
        assert_almost_equal_scalar(actual, e);
        notify_pass(i);
    });
}

fn generate_reference_state() -> usize {
    // A=ON, B=OFF, C=OFF
    0b001
}

fn generate_reference_tpm() -> na::DMatrix::<f64> {
    let values = &[
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
    ];

    na::DMatrix::<f64>::from_row_slice(8, 8, values)
}

#[test]
fn test_calc_cause_repertoire() {
    let tpm = generate_reference_tpm();
    let current_state = generate_reference_state();

    let mut purview_masks = Vec::<usize>::new();
    let mut mechanism_masks = Vec::<usize>::new();
    let mut expected = Vec::<na::DVector<f64>>::new();

    // CASE 0, Fig.4 p(ABC^p|A^c=1)
    purview_masks.push(0b111);
    mechanism_masks.push(0b001);
    let mut e_0 = na::DVector::<f64>::from_column_slice(&[0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    normalize_repertoire(&mut e_0, None);
    expected.push(e_0);

    // CASE 1, Fig.10 p(AB^p|C^c), ,
    purview_masks.push(0b011);
    mechanism_masks.push(0b100);
    let mut e_1 = na::DVector::<f64>::from_column_slice(&[1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]);
    normalize_repertoire(&mut e_1, None);
    expected.push(e_1);

    // CASE 2, Fig.10 p(AC^p|B^c)
    purview_masks.push(0b101);
    mechanism_masks.push(0b010);
    let mut e_2 = na::DVector::<f64>::from_column_slice(&[1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0]);
    normalize_repertoire(&mut e_2, None);
    expected.push(e_2);

    // CASE 3, Fig.10 p(BC^p|A^c)
    purview_masks.push(0b110);
    mechanism_masks.push(0b001);
    let mut e_3 = na::DVector::<f64>::from_column_slice(&[0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    normalize_repertoire(&mut e_3, None);
    expected.push(e_3);

    // CASE 4, Fig.10 p(ABC^p|ABC^c)
    purview_masks.push(0b111);
    mechanism_masks.push(0b111);
    let mut e_4 = na::DVector::<f64>::from_column_slice(&[0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]);
    normalize_repertoire(&mut e_4, None);
    expected.push(e_4);

    // CASE 5, Fig.10 p(AB^p|BC^c)
    purview_masks.push(0b011);
    mechanism_masks.push(0b110);
    let mut e_5 = na::DVector::<f64>::from_column_slice(&[2.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 1.0]);
    normalize_repertoire(&mut e_5, None);
    expected.push(e_5);

    // CASE 6, Fig.10 p(ABC^p|AB^c)
    purview_masks.push(0b111);
    mechanism_masks.push(0b011);
    let mut e_6 = na::DVector::<f64>::from_column_slice(&[0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0]);
    normalize_repertoire(&mut e_6, None);
    expected.push(e_6);

    // CASE 7, p(ABC^p)
    purview_masks.push(0b111);
    mechanism_masks.push(0b000);
    let mut e_7 = na::DVector::<f64>::from_column_slice(&[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    normalize_repertoire(&mut e_7, None);
    expected.push(e_7);

    // CASE 8, p([])
    purview_masks.push(0b000);
    mechanism_masks.push(0b000);
    let mut e_8 = na::DVector::<f64>::from_column_slice(&[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    normalize_repertoire(&mut e_8, None);
    expected.push(e_8);

    // CASE 9, p([]|ABC^c)
    purview_masks.push(0b000);
    mechanism_masks.push(0b111);
    let mut e_9 = na::DVector::<f64>::from_column_slice(&[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    normalize_repertoire(&mut e_9, None);
    expected.push(e_9);


    let max_dim = (tpm.nrows() - 1).count_ones() as usize;
    expected.iter().enumerate().for_each(|(i, e)| {
        let purview = BitBasis::construct_from_mask(purview_masks[i], max_dim);
        let c_purview = purview.generate_complement_basis();
        let mechanism =BitBasis::construct_from_mask(mechanism_masks[i], max_dim);

        let marginal = calc_cause_repertoire(&purview, &mechanism, current_state, &tpm);
        let unconstrained = calc_cause_repertoire(&c_purview, &BitBasis::null_basis(max_dim), current_state, &tpm);

        let actual  = marginal.component_mul(&unconstrained);

        assert_almost_equal_vec(&actual, e);
        notify_pass(i);
    });
}

#[test]
fn test_calc_effect_repertoire() {
    let tpm = generate_reference_tpm();
    let current_state = generate_reference_state();

    let mut purview_masks = Vec::<usize>::new();
    let mut mechanism_masks = Vec::<usize>::new();
    let mut expected = Vec::<na::DVector<f64>>::new();

    // CASE 0, p(ABC^f)
    purview_masks.push(0b111);
    mechanism_masks.push(0b000);
    let (a_0, a_1) = (1.0 / 4.0, 3.0 / 4.0);
    let uc_a = na::DVector::<f64>::from_column_slice(&[a_0, a_1, a_0, a_1, a_0, a_1, a_0, a_1]);
    let (b_0, b_1) = (3.0 / 4.0, 1.0 / 4.0);
    let uc_b = na::DVector::<f64>::from_column_slice(&[b_0, b_0, b_1, b_1, b_0, b_0, b_1, b_1]);
    let uc_c = na::DVector::<f64>::from_element(tpm.nrows(), 1.0 / 2.0);
    let uc_abc = uc_a.component_mul(&uc_b).component_mul(&uc_c);
    expected.push(uc_abc.clone());

    // CASE 1, p([])
    purview_masks.push(0b000);
    mechanism_masks.push(0b000);
    expected.push(uc_abc.clone());

    // CASE 2, Fig.10 p(AC^f|ABC^c)
    purview_masks.push(0b101);
    mechanism_masks.push(0b111);
    let mut m_2 = na::DVector::<f64>::from_column_slice(&[0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0]);
    normalize_repertoire(&mut m_2, Some(2.0));
    expected.push(m_2.component_mul(&uc_b));

    // CASE 3, Fig.10 p(A^f|BC^c)
    purview_masks.push(0b001);
    mechanism_masks.push(0b110);
    let mut m_3 = na::DVector::<f64>::from_column_slice(&[2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0]);
    normalize_repertoire(&mut m_3, Some(4.0));
    expected.push(m_3.component_mul(&uc_b).component_mul(&uc_c));

    // CASE 4, Fig.10 p(C^f|AB^c)
    purview_masks.push(0b100);
    mechanism_masks.push(0b011);
    let mut m_4 = na::DVector::<f64>::from_column_slice(&[0.0, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 2.0]);
    normalize_repertoire(&mut m_4, Some(4.0));
    expected.push(m_4.component_mul(&uc_a).component_mul(&uc_b));

    // CASE 5, Fig.10 p(AB^f|C^c)
    purview_masks.push(0b011);
    mechanism_masks.push(0b100);
    let mut m_5 = na::DVector::<f64>::from_column_slice(&[2.0, 2.0, 0.0, 0.0, 2.0, 2.0, 0.0, 0.0]);
    normalize_repertoire(&mut m_5, Some(2.0));
    expected.push(m_5.component_mul(&uc_c));

    // CASE 6, Fig.10 p(A^f|B^c)
    purview_masks.push(0b001);
    mechanism_masks.push(0b100);
    let mut m_6 = na::DVector::<f64>::from_column_slice(&[2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]);
    normalize_repertoire(&mut m_6, Some(4.0));
    expected.push(m_6.component_mul(&uc_b).component_mul(&uc_c));

    // CASE 7, Fig.10 p(B^f|A^c)
    purview_masks.push(0b010);
    mechanism_masks.push(0b001);
    let mut m_7 = na::DVector::<f64>::from_column_slice(&[2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]);
    normalize_repertoire(&mut m_7, Some(4.0));
    expected.push(m_7.component_mul(&uc_a).component_mul(&uc_c));


    let max_dim = (tpm.nrows() - 1).count_ones() as usize;
    expected.iter().enumerate().for_each(|(i, e)| {
        let purview = BitBasis::construct_from_mask(purview_masks[i], max_dim);
        let c_purview = purview.generate_complement_basis();
        let mechanism =BitBasis::construct_from_mask(mechanism_masks[i], max_dim);

        let marginal = calc_effect_repertoire(&purview, &mechanism, current_state, &tpm);
        let unconstrained = calc_effect_repertoire(&c_purview, &BitBasis::null_basis(max_dim), current_state, &tpm);

        let actual  = marginal.component_mul(&unconstrained);

        assert_almost_equal_vec(&actual, e);
        notify_pass(i);
    });
}

#[test]
fn test_search_concept_with_parts() {
    let tpm = generate_reference_tpm();
    let current_state = generate_reference_state();

    let cause_parts = generate_all_repertoire_parts(crate::mechanism::RepertoireType::CAUSE, current_state, &tpm);
    let effect_parts = generate_all_repertoire_parts(crate::mechanism::RepertoireType::EFFECT, current_state, &tpm);

    let mut mechanisms = Vec::<usize>::new();
    let mut expected = Vec::<f64>::new();

    // CASE 0, Fig.10 ABC
    mechanisms.push(0b111);
    expected.push(0.5);

    // CASE 1, Fig.10 BC
    mechanisms.push(0b110);
    expected.push(0.3333333333);

    // CASE 2, Fig.10 AB
    mechanisms.push(0b011);
    expected.push(0.25);

    // CASE 3, Fig.10 C
    mechanisms.push(0b100);
    expected.push(0.25);

    // CASE 4, Fig.10 B
    mechanisms.push(0b010);
    expected.push(0.1666666666);

    // CASE 5, Fig.10 A
    mechanisms.push(0b001);
    expected.push(0.1666666666);

    // CASE 6, Fig.10 AC (to be fully reduced)
    mechanisms.push(0b101);
    expected.push(0.0);


    (0..mechanisms.len()).for_each(|i| {
        let mechanism = BitBasis::construct_from_mask(mechanisms[i], 3);
        let concept = search_concept_with_parts(&mechanism, &cause_parts, &effect_parts);
        let actual = if let Some(v) = concept {
            v.core_cause.phi.min(v.core_effect.phi)
        } else {
            0.0
        };

        assert_almost_equal_scalar(actual, expected[i]);
        notify_pass(i);
    });
}
