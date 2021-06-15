const PRECISION : f64 = 1.0e-7;

pub enum Comparison {
    AlmostEqual,
    NotEqual(f64),
}

pub fn compare_roughly(left: f64, right: f64) -> Comparison {
    let diff = left - right;

    if diff.abs() < PRECISION {
        return Comparison::AlmostEqual;
    }

    Comparison::NotEqual(diff)
}
