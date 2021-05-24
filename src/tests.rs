use nalgebra as na;
use crate::emd::calc_emd_for_mechanism;


#[test]
fn calc_emd() {
    let P_0 = na::DVector::<f64>::from_element(8, 1.0 / 8.0);
    let Q_0 = na::DVector::<f64>::from_element(8, 1.0 / 8.0);

    let mut P_1 = na::DVector::<f64>::from_element(8, 0.0);
    P_1[7] = 1.0;
    let Q_1 = na::DVector::<f64>::from_element(8, 1.0 / 8.0);

    let mut P_2 = na::DVector::<f64>::from_element(8, 1.0 / 7.0);
    P_2[7] = 0.0;
    let Q_2 = na::DVector::<f64>::from_element(8, 1.0 / 8.0);


    assert!(calc_emd_for_mechanism(&P_0, &Q_0).objective() == 0.0);
    assert!(calc_emd_for_mechanism(&P_1, &Q_1).objective() == 1.5);
    assert!(calc_emd_for_mechanism(&P_2, &Q_2).objective() == 0.2142857142857143);
}
