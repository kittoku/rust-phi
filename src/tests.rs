use std::usize;
use nalgebra as na;
use crate::{bases::BitBases, emd::calc_emd_for_mechanism, repertoire::{calc_cause_repertoire, normalize_repertoire}};


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
    let values = vec![
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
    ];

    na::DMatrix::<f64>::from_vec(8, 8, values).transpose()
}

#[test]
fn test_calc_cause_repertoire() {
    let tpm = generate_reference_tpm();
    let current_state = generate_reference_state();

    let mut past_masks = Vec::<usize>::new();
    let mut current_masks = Vec::<usize>::new();
    let mut expected = Vec::<na::DVector<f64>>::new();

    // CASE 0, Fig.4 p(ABC^p|A^c=1)
    past_masks.push(0b111);
    current_masks.push(0b001);
    let mut e_0 = na::DVector::<f64>::from_vec(vec![0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    normalize_repertoire(&mut e_0, None);
    expected.push(e_0);

    // CASE 1, Fig.10 p(AB^p|C^c), ,
    past_masks.push(0b011);
    current_masks.push(0b100);
    let mut e_1 = na::DVector::<f64>::from_vec(vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]);
    normalize_repertoire(&mut e_1, None);
    expected.push(e_1);

    // CASE 2, Fig.10 p(AC^p|B^c)
    past_masks.push(0b101);
    current_masks.push(0b010);
    let mut e_2 = na::DVector::<f64>::from_vec(vec![1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0]);
    normalize_repertoire(&mut e_2, None);
    expected.push(e_2);

    // CASE 3, Fig.10 p(BC^p|A^c)
    past_masks.push(0b110);
    current_masks.push(0b001);
    let mut e_3 = na::DVector::<f64>::from_vec(vec![0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    normalize_repertoire(&mut e_3, None);
    expected.push(e_3);

    // CASE 4, Fig.10 p(ABC^p|ABC^c)
    past_masks.push(0b111);
    current_masks.push(0b111);
    let mut e_4 = na::DVector::<f64>::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]);
    normalize_repertoire(&mut e_4, None);
    expected.push(e_4);

    // CASE 5, Fig.10 p(AB^p|BC^c)
    past_masks.push(0b011);
    current_masks.push(0b110);
    let mut e_5 = na::DVector::<f64>::from_vec(vec![2.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 1.0]);
    normalize_repertoire(&mut e_5, None);
    expected.push(e_5);

    // CASE 6, Fig.10 p(ABC^p|AB^c)
    past_masks.push(0b111);
    current_masks.push(0b011);
    let mut e_6 = na::DVector::<f64>::from_vec(vec![0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0]);
    normalize_repertoire(&mut e_6, None);
    expected.push(e_6);

    // CASE 7, p(ABC^p)
    past_masks.push(0b111);
    current_masks.push(0b000);
    let mut e_7 = na::DVector::<f64>::from_vec(vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    normalize_repertoire(&mut e_7, None);
    expected.push(e_7);

    // CASE 8, p([])
    past_masks.push(0b000);
    current_masks.push(0b000);
    let mut e_8 = na::DVector::<f64>::from_vec(vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    normalize_repertoire(&mut e_8, None);
    expected.push(e_8);

    // CASE 9, p([]|ABC^c)
    past_masks.push(0b000);
    current_masks.push(0b111);
    let mut e_9 = na::DVector::<f64>::from_vec(vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    normalize_repertoire(&mut e_9, None);
    expected.push(e_9);


    let max_dim = (tpm.nrows() - 1).count_ones() as usize;
    expected.iter().enumerate().for_each(|(i, e)| {
        let past_bases = BitBases::construct_from_mask(past_masks[i], max_dim);
        let c_past_bases = past_bases.generate_complement_bases();
        let current_bases =BitBases::construct_from_mask(current_masks[i], max_dim);

        let marginal = calc_cause_repertoire(&past_bases, &current_bases, current_state, &tpm);
        let unconstrained = calc_cause_repertoire(&c_past_bases, &BitBases::null_bases(max_dim), current_state, &tpm);

        let actual  = marginal.component_mul(&unconstrained);

        assert_almost_equal_vec(&actual, e);
        notify_pass(i);
    });
}
