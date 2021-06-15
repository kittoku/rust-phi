extern crate rust_phi;


fn main() {
    // simulate the example in
    // Oizumi M, Albantakis L, Tononi G (2014)
    // From the Phenomenology to the Mechanisms of Consciousness: Integrated Information Theory 3.0.
    // PLOS Computational Biology 10(5): e1003588. https://doi.org/10.1371/journal.pcbi.1003588

    // `link.sif` defines a system in Figure 1 (A)
    let link_path = r"src/../link.sif";

    // parse link.sif
    let infos = rust_phi::sif::read_sif(&link_path);

    // generate each functions of elements
    let fns = rust_phi::link_fn::get_link_fns(infos);

    // calculate a transition probability matrix of the whole system
    let tpm = rust_phi::tpm::calc_tpm(fns, 4);
    println!("TPM of the whole system: {}", tpm);

    let state = 0b010001; // means the current state A=ON, B=OFF, C=OFF, D=OFF, E=ON, F=OFF
    let mask = 0b000111; // means we now consider ABC as a candidate set

    // calculate marginal distribution for ABC
    let surviving_basis = rust_phi::basis::BitBasis::construct_from_mask(mask, 6);
    let marginal = rust_phi::tpm::calc_fixed_marginal_tpm(&surviving_basis, state, &tpm);
    println!("TPM of ABC: {}", marginal); // == Figure 1 (B)

    // get all parts used in mechanism partition
    let cause_parts = rust_phi::mechanism::generate_all_repertoire_parts(rust_phi::mechanism::RepertoireType::CAUSE, state, &marginal);
    let effect_parts = rust_phi::mechanism::generate_all_repertoire_parts(rust_phi::mechanism::RepertoireType::EFFECT, state, &marginal);

    // search a concept
    let mechanism_ab = rust_phi::basis::BitBasis::construct_from_mask(0b011, 3);
    let concept_ab = rust_phi::mechanism::search_concept_with_parts(&mechanism_ab, &cause_parts, &effect_parts);
    println!("Concept AB:");
    println!("CAUSE -> {}", concept_ab.core_cause.repertoire);
    println!("EFFECT -> {}", concept_ab.core_effect.repertoire);
    println!("phi -> {}", concept_ab.phi);

    let mechanism_ac = rust_phi::basis::BitBasis::construct_from_mask(0b101, 3);
    let concept_ac = rust_phi::mechanism::search_concept_with_parts(&mechanism_ac, &cause_parts, &effect_parts);
    println!("Concept AC"); // means AC is fully reduced
    println!("phi -> {}", concept_ac.phi); // means AC is fully reduced

    // partition a system
    let partition = rust_phi::partition::SystemPartition { // cut A -> BC
        cut_from: vec![0],
        cut_to: vec![1, 2],
    };

    let partitioned_tpm = rust_phi::tpm::calc_partitioned_marginal_tpm(&partition, &marginal);
    let partitioned_cause_parts = rust_phi::mechanism::generate_all_repertoire_parts(rust_phi::mechanism::RepertoireType::CAUSE, state, &partitioned_tpm);
    let partitioned_effect_parts = rust_phi::mechanism::generate_all_repertoire_parts(rust_phi::mechanism::RepertoireType::EFFECT, state, &partitioned_tpm);

    // calculate extended EMD
    let constellation = rust_phi::system::search_constellation_with_parts(&cause_parts, &effect_parts);
    let partitioned_constellation = rust_phi::system::search_constellation_with_parts(&partitioned_cause_parts, &partitioned_effect_parts);

    let extended_emd = rust_phi::emd::calc_constellation_emd(&constellation, &partitioned_constellation);
    println!("Big phi when A =/=> BC: {}", extended_emd);

    // search MIP
    let constellation_mip = rust_phi::system::search_constellation_with_mip(state, &marginal);
    let mip = constellation_mip.mip;
    println!("MIP: {:?} =/=> {:?}", mip.partition.cut_from, mip.partition.cut_to); // [0, 1] =/=> [2] equivalent to AB =/=> C
    println!("Max big phi: {}", mip.phi);
}
