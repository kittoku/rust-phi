# rust-phi
This library is made for my understanding Rust and Integrated Information Theory (IIT3.0).

## How to use
See [this example](example/src/main.rs).  
  
You can calculate things like *concept*, *MIP* and *complex*.

## Benchmark
Using the system of ABCDEF elements appeared in Figure 1 of [the original paper](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003588), 
you can search a complex as a benchmark.
Execute [benchmark crate](benchmark/src/main.rs) with `cargo run --release`.
The default setting uses 4 threads.  

In my environment, it took about 240 seconds and needed about 30MB memory usage.
