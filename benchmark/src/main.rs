extern crate rust_phi;


fn main() {
    // simulate the example in
    // Oizumi M, Albantakis L, Tononi G (2014)
    // From the Phenomenology to the Mechanisms of Consciousness: Integrated Information Theory 3.0.
    // PLOS Computational Biology 10(5): e1003588. https://doi.org/10.1371/journal.pcbi.1003588

    const NUM_THREADS: usize = 4;

    // `link.sif` defines a system in Figure 1 (A)
    let link_path = r"src/../link.sif";

    // parse link.sif
    let infos = rust_phi::sif::read_sif(&link_path);

    // generate each functions of elements
    let fns = rust_phi::link_fn::get_link_fns(infos);

    // calculate a transition probability matrix of the whole system
    let full_state_tpm = rust_phi::tpm::calc_tpm(fns, NUM_THREADS);
    println!("TPM of the whole system: {}", full_state_tpm);

    let full_state = 0b010001; // means the current state A=ON, B=OFF, C=OFF, D=OFF, E=ON, F=OFF

    // search complex
    let enable_log = true;
    let complex = rust_phi::system::search_complex(full_state, full_state_tpm, NUM_THREADS, enable_log);
    /*
        This takes relatively short time since `marginal_tpm` is used as a full-state tpm.
        If you want to search a complex among a system of ABCDEF, you need use `full_state_tpm`.
    */

    println!("Complex: {:?}", complex.elements)
}
