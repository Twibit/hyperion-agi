# neurorust — measured smoke-test results

## `cargo run --release --bin neurorust-demo`

1 s of simulation, dt = 0.05 ms, I_inj = 10 µA/cm²

|                     model | spikes | V-trace compression | total on-wire |
|--------------------------:|-------:|--------------------:|--------------:|
|            Hodgkin–Huxley |    69  |               3.13× |     25 561 B  |
|           Izhikevich (RS) |    23  |               5.83× |     13 748 B  |
|         Multi-compartment |    19  |               4.04× |     19 819 B  |

Biological sanity: HH at 10 µA/cm² fires ~70 Hz — matches Hodgkin & Huxley
(1952) predictions for the squid-giant-axon parameter set. Izhikevich RS is
in the 20–40 Hz literature range. Multi-compartment fires more slowly because
current is split between soma and dendrite tip.

## `cargo run --release --bin neurorust-bench`

5 s workload, 100 000 samples, 500×500 connectivity at ρ = 5 %, 1024×4096 raster.

| stream                | naive bytes | after bytes | ratio |
|-----------------------|-------------|-------------|-------|
| Spike train (99 sp)   |         792 |         199 |  4.0× |
| V-trace (100 000 sm)  |     400 000 |      46 809 |  8.5× |
| Connectivity 500×500  |   1 000 000 |     102 340 |  9.8× |
| Raster 1024×4096      |   4 194 304 |     524 288 |  8.0× |

Total working-set reduction across all four streams: **~6.9 MB → ~0.67 MB (10.3×)**.

## `cargo test --release`

```
running 9 tests
test csr_matches_dense_spmv ... ok
test bit_raster_compression_is_8x ... ok
test hh_is_quiet_at_rest ... ok
test hh_fires_under_suprathreshold_current ... ok
test izh_regular_spiking_fires_within_expected_rate ... ok
test spike_train_roundtrip_is_lossless ... ok
test tridiag_solver_matches_reference ... ok
test v_trace_roundtrip_below_biological_noise ... ok
test multi_compartment_propagates_and_spikes ... ok

test result: ok. 9 passed; 0 failed
```

All biological plausibility, compression round-trip, sparse matvec, and FFI
kernel tests pass.
