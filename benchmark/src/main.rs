use std::time::SystemTime;
extern crate rust_phi;


fn main() {
    const NUM_THREADS: usize = 4;

    let link_path = r"src/../link.sif";
    let infos = rust_phi::sif::read_sif(&link_path);
    let fns = rust_phi::link_fn::get_link_fns(infos);
    let full_state_tpm = rust_phi::tpm::calc_tpm(fns, NUM_THREADS);
    let full_state = 0b010001;

    let start_time = SystemTime::now();

    let complex = rust_phi::system::search_complex(full_state, &full_state_tpm, NUM_THREADS, true);

    println!("\nComplex: {:?}", complex.elements);
    println!("Total elapsed time: {:.2e}", start_time.elapsed().unwrap().as_secs_f64());
}
