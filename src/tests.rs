use crate::emd::calc_emd_for_mechanism;


#[test]
fn calc_emd() {
    let P_0 = vec![1.0 / 8.0; 8];
    let Q_0 = vec![1.0 / 8.0; 8];

    let P_1 = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0];
    let Q_1 = vec![1.0 / 8.0; 8];

    let mut P_2 = vec![1.0 / 7.0; 7];
    P_2.push(0.0);
    let Q_2 = vec![1.0 / 8.0; 8];


    assert!(calc_emd_for_mechanism(&P_0, &Q_0).objective() == 0.0);
    assert!(calc_emd_for_mechanism(&P_1, &Q_1).objective() == 1.5);
    assert!(calc_emd_for_mechanism(&P_2, &Q_2).objective() == 0.2142857142857143);
}
