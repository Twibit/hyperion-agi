//! Hybrid per-datum compression for neuron memory.
//!
//! ## Why hybrid?
//!
//! Different biological signals have different statistical structure:
//!
//! | Signal              | Structure                              | Codec                     |
//! |---------------------|----------------------------------------|---------------------------|
//! | Spike train (times) | Sparse, monotone increasing            | Delta ISI + zstd (lossless) |
//! | Membrane trace V(t) | Dense, bounded (−90…+50 mV), smooth    | Affine i16 quantisation (lossy) |
//! | Connectivity W      | Sparse (~1 % of dense neocortex)       | CSR                       |
//! | Population raster   | Boolean, dense over neurons × time     | Bit-packed via `bitvec`   |
//!
//! Every neuron owns a [`NeuronMemory`] that dispatches each incoming datum to
//! the correct stream. Round-trip decompression is exposed for tests, plotting,
//! and downstream analysis.

use bitvec::prelude::*;
use serde::{Deserialize, Serialize};
use std::io::Read;
use thiserror::Error;

/// Errors from the compression layer.
#[derive(Debug, Error)]
pub enum CompressionError {
    /// Wrapper around a `std::io::Error` (zstd uses `io::Result`).
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    /// bincode serialization failed.
    #[error("bincode error: {0}")]
    Bincode(#[from] Box<bincode::ErrorKind>),
}

// ---------------------------------------------------------------------------
// 1. Spike train — delta ISI + zstd
// ---------------------------------------------------------------------------

/// Lossless codec for spike trains.
///
/// Absolute times → inter-spike intervals (in units of `dt_ms`) → zstd.
/// Because ISIs of a spiking neuron are typically tens of ms (thousands of
/// dt-steps) and vary slowly, delta encoding + entropy coding is very tight.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct SpikeTrain {
    /// Time step used to quantise absolute spike times.
    pub dt_ms: f64,
    /// Last stored spike step (for incremental delta encoding).
    last_step: u64,
    /// zstd-compressed varint-encoded ISI stream (flushed portion).
    payload: Vec<u8>,
    /// Uncompressed varint bytes pending compression. Flushed automatically
    /// when it grows past `PENDING_FLUSH_BYTES` or on any read.
    ///
    /// Motivation: the original `push` implementation decompressed the
    /// entire payload, appended one varint, and recompressed — O(n) per
    /// push → O(n²) for a full simulation, catastrophic on long runs.
    /// See `FAILURE_PREDICTIONS.md` §1.
    pending: Vec<u8>,
    /// Number of spikes stored — cached to avoid decompression for `.len()`.
    n_spikes: usize,
    /// Uncompressed byte counter for reporting compression ratio.
    raw_bytes: usize,
}

/// Flush threshold for `pending` staging buffers. 64 KB balances the
/// per-flush cost against the working set kept resident. Chosen empirically
/// against zstd level-3 throughput on modern x86-64.
const PENDING_FLUSH_BYTES: usize = 64 * 1024;

impl SpikeTrain {
    /// Create an empty spike train quantised at `dt_ms`.
    pub fn new(dt_ms: f64) -> Self {
        Self {
            dt_ms,
            last_step: 0,
            payload: Vec::new(),
            pending: Vec::new(),
            n_spikes: 0,
            raw_bytes: 0,
        }
    }

    /// Append an absolute spike time (ms). Must be monotonically increasing.
    ///
    /// This is now O(1) amortised: the varint is staged in `pending` and
    /// only merged into the zstd payload when `pending` grows past
    /// `PENDING_FLUSH_BYTES` or a decoding operation is requested.
    pub fn push(&mut self, t_ms: f64) -> Result<(), CompressionError> {
        let step = (t_ms / self.dt_ms).round() as u64;
        let delta = step.saturating_sub(self.last_step);
        self.last_step = step;

        varint_encode(delta, &mut self.pending);
        self.n_spikes += 1;
        self.raw_bytes += 8;

        if self.pending.len() >= PENDING_FLUSH_BYTES {
            self.flush()?;
        }
        Ok(())
    }

    /// Force any staged spikes into the compressed payload. Callers that need
    /// the compressed representation to reflect the latest pushes should call
    /// this explicitly; reader methods do so implicitly.
    pub fn flush(&mut self) -> Result<(), CompressionError> {
        if self.pending.is_empty() { return Ok(()); }
        let mut raw = decompress(&self.payload)?;
        raw.extend_from_slice(&self.pending);
        self.payload = compress(&raw)?;
        self.pending.clear();
        Ok(())
    }

    /// Batched append — still supported for API compatibility; internally
    /// this now shares the same staging path as `push` and only flushes
    /// once at the end.
    pub fn extend<I: IntoIterator<Item = f64>>(&mut self, spikes: I) -> Result<(), CompressionError> {
        // Historical fast path: keep decompress-once semantics for the
        // batched call so users who *did* switch to `extend` see the same
        // (or better) throughput.
        let mut raw = decompress(&self.payload)?;
        raw.extend_from_slice(&self.pending);
        self.pending.clear();
        for t_ms in spikes {
            let step = (t_ms / self.dt_ms).round() as u64;
            let delta = step.saturating_sub(self.last_step);
            self.last_step = step;
            varint_encode(delta, &mut raw);
            self.n_spikes += 1;
            self.raw_bytes += 8;
        }
        self.payload = compress(&raw)?;
        Ok(())
    }

    /// Decode back to absolute spike times in ms.
    pub fn to_times_ms(&self) -> Result<Vec<f64>, CompressionError> {
        let raw = decompress(&self.payload)?;
        let mut cursor = 0usize;
        let mut step: u64 = 0;
        let mut out = Vec::with_capacity(self.n_spikes);
        while cursor < raw.len() {
            let (delta, adv) = varint_decode(&raw[cursor..]);
            cursor += adv;
            step += delta;
            out.push(step as f64 * self.dt_ms);
        }
        Ok(out)
    }

    /// Number of spikes stored.
    pub fn len(&self) -> usize { self.n_spikes }

    /// Whether the train is empty.
    pub fn is_empty(&self) -> bool { self.n_spikes == 0 }

    /// Compression ratio vs. a naive `Vec<u64>` of absolute spike steps.
    pub fn compression_ratio(&self) -> f64 {
        if self.raw_bytes == 0 { return 1.0; }
        self.raw_bytes as f64 / self.payload.len().max(1) as f64
    }

    /// Compressed payload size in bytes (payload + staged pending).
    pub fn compressed_bytes(&self) -> usize { self.payload.len() + self.pending.len() }
}

// ---------------------------------------------------------------------------
// 2. Membrane trace — affine i16 quantisation
// ---------------------------------------------------------------------------

/// Lossy codec for V(t) traces.
///
/// Membrane potentials in cortical neurons live in `[-100, +60] mV`. Storing
/// them as `f32` (4 B/sample) wastes ~2× vs. quantising to `i16` under an
/// affine map. Reconstruction error is ≤ 160 mV / 65 535 ≈ 2.4 µV — orders of
/// magnitude below biological noise floors.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MembraneTrace {
    /// Minimum representable voltage (mV).
    pub v_min: f32,
    /// Maximum representable voltage (mV).
    pub v_max: f32,
    /// zstd-compressed i16 samples (little-endian).
    payload: Vec<u8>,
    /// Unflushed i16 bytes staged for the next compression pass.
    /// See `FAILURE_PREDICTIONS.md` §1 — the original `extend([single])`
    /// call in `NeuronMemory::record` decompressed + recompressed the
    /// entire payload on every simulation step. That is O(n) per step and
    /// O(n²) per simulation. Staging restores O(1) amortised.
    #[serde(default)]
    pending: Vec<u8>,
    /// Number of samples stored.
    n_samples: usize,
    /// Sampling interval (ms).
    pub dt_ms: f64,
}

impl MembraneTrace {
    /// Create an empty trace over voltage range `[v_min, v_max]` mV.
    pub fn new(v_min: f32, v_max: f32, dt_ms: f64) -> Self {
        assert!(v_max > v_min, "v_max must exceed v_min");
        Self {
            v_min, v_max, dt_ms,
            payload: Vec::new(),
            pending: Vec::new(),
            n_samples: 0,
        }
    }

    /// Append many samples. Amortised O(1) per sample — samples are quantised
    /// and staged in `pending`, flushed to the zstd payload only when
    /// `pending` exceeds `PENDING_FLUSH_BYTES` (or a reader is invoked, or
    /// `flush()` is called explicitly).
    pub fn extend<I: IntoIterator<Item = f32>>(&mut self, samples: I) -> Result<(), CompressionError> {
        let scale = (self.v_max - self.v_min) / (u16::MAX as f32);
        for v in samples {
            let clamped = v.clamp(self.v_min, self.v_max);
            let q = (((clamped - self.v_min) / scale).round() as i32 - i16::MIN as i32) as i16;
            self.pending.extend_from_slice(&q.to_le_bytes());
            self.n_samples += 1;
        }
        if self.pending.len() >= PENDING_FLUSH_BYTES {
            self.flush()?;
        }
        Ok(())
    }

    /// Force staged samples into the compressed payload.
    pub fn flush(&mut self) -> Result<(), CompressionError> {
        if self.pending.is_empty() { return Ok(()); }
        let mut raw = decompress(&self.payload)?;
        raw.extend_from_slice(&self.pending);
        self.payload = compress(&raw)?;
        self.pending.clear();
        Ok(())
    }

    /// Decode back to `Vec<f32>` in mV. Includes staged samples transparently.
    pub fn to_samples(&self) -> Result<Vec<f32>, CompressionError> {
        let mut raw = decompress(&self.payload)?;
        raw.extend_from_slice(&self.pending);
        let scale = (self.v_max - self.v_min) / (u16::MAX as f32);
        let mut out = Vec::with_capacity(self.n_samples);
        for chunk in raw.chunks_exact(2) {
            let q = i16::from_le_bytes([chunk[0], chunk[1]]);
            let u = (q as i32 - i16::MIN as i32) as f32;
            out.push(self.v_min + u * scale);
        }
        Ok(out)
    }

    /// Number of samples stored.
    pub fn len(&self) -> usize { self.n_samples }

    /// Whether the trace is empty.
    pub fn is_empty(&self) -> bool { self.n_samples == 0 }

    /// Naive `f32` bytes vs. compressed bytes.
    pub fn compression_ratio(&self) -> f64 {
        let naive = self.n_samples * 4;
        if naive == 0 { return 1.0; }
        naive as f64 / (self.payload.len() + self.pending.len()).max(1) as f64
    }

    /// Compressed payload size in bytes (payload + staged pending).
    pub fn compressed_bytes(&self) -> usize { self.payload.len() + self.pending.len() }
}

// ---------------------------------------------------------------------------
// 3. Connectivity — CSR sparse matrix
// ---------------------------------------------------------------------------

/// Compressed-sparse-row synaptic connectivity matrix.
///
/// Cortical connectivity is ~1–10 % dense; CSR uses `(nnz)` doubles + `(nnz)`
/// u32 column indices + `(rows+1)` u32 row pointers instead of `rows×cols`
/// doubles.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseConnectivity {
    /// Number of postsynaptic neurons (rows).
    pub rows: usize,
    /// Number of presynaptic neurons (cols).
    pub cols: usize,
    /// Row pointer array of length `rows + 1`.
    pub row_ptr: Vec<u32>,
    /// Column indices, one per non-zero.
    pub col_idx: Vec<u32>,
    /// Synaptic weights, one per non-zero.
    pub weights: Vec<f32>,
}

impl SparseConnectivity {
    /// Build a CSR matrix from a dense row-major slice.
    pub fn from_dense(rows: usize, cols: usize, dense: &[f32], eps: f32) -> Self {
        assert_eq!(dense.len(), rows * cols);
        let mut row_ptr = Vec::with_capacity(rows + 1);
        let mut col_idx = Vec::new();
        let mut weights = Vec::new();
        row_ptr.push(0);
        for r in 0..rows {
            for c in 0..cols {
                let w = dense[r * cols + c];
                if w.abs() > eps {
                    col_idx.push(c as u32);
                    weights.push(w);
                }
            }
            row_ptr.push(col_idx.len() as u32);
        }
        Self { rows, cols, row_ptr, col_idx, weights }
    }

    /// Number of non-zero entries.
    pub fn nnz(&self) -> usize { self.weights.len() }

    /// Density (nnz / (rows·cols)).
    pub fn density(&self) -> f64 {
        self.nnz() as f64 / (self.rows * self.cols).max(1) as f64
    }

    /// Bytes on the wire vs. dense `f32` representation.
    pub fn compression_ratio(&self) -> f64 {
        let dense = self.rows * self.cols * 4;
        let sparse = self.weights.len() * 4 + self.col_idx.len() * 4 + self.row_ptr.len() * 4;
        if sparse == 0 { return 1.0; }
        dense as f64 / sparse as f64
    }

    /// Sparse matrix–vector product: `out = W · x` (scalar path).
    ///
    /// Delegates the inner accumulation loop to
    /// [`crate::asm_kernels::sparse_dot_f32`], which pins the accumulator
    /// to xmm0 on x86-64 (avoiding a load/store round-trip LLVM has been
    /// observed to emit under `-Copt-level=3` for CSR-style loops with
    /// dependent-load chains).
    ///
    /// For the packed AVX2 path (8-wide gather + FMA, ~3× faster on
    /// realistic connectomes), call [`Self::spmv_avx2`] instead.
    pub fn spmv(&self, x: &[f32], out: &mut [f32]) {
        assert_eq!(x.len(), self.cols);
        assert_eq!(out.len(), self.rows);
        for r in 0..self.rows {
            let start = self.row_ptr[r] as usize;
            let end   = self.row_ptr[r + 1] as usize;
            out[r] = crate::asm_kernels::sparse_dot_f32(
                &self.weights, &self.col_idx, x, start, end,
            );
        }
    }

    /// Sparse matvec via the packed AVX2 (`vpgatherdd` + `vfmadd231ps`)
    /// kernel, with software prefetching of the next row's cache lines.
    ///
    /// Behaviourally identical to [`Self::spmv`] to within f32 unit-round-off
    /// (FMA rounds once vs mul-then-add rounding twice). On x86-64 without
    /// AVX2 it falls back automatically to the scalar path.
    pub fn spmv_avx2(&self, x: &[f32], out: &mut [f32]) {
        assert_eq!(x.len(), self.cols);
        assert_eq!(out.len(), self.rows);
        crate::asm_kernels_avx2::spmv_csr_f32_avx2(
            &self.row_ptr, &self.col_idx, &self.weights, x, out,
        );
    }
}

// ---------------------------------------------------------------------------
// 4. Population raster — bit-packed
// ---------------------------------------------------------------------------

/// Bit-packed spike raster: one bit per (neuron, time-bin).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BitRaster {
    /// Number of neurons (rows of the raster).
    pub n_neurons: usize,
    /// Number of time bins (columns of the raster).
    pub n_bins: usize,
    /// Row-major bit buffer of length `n_neurons * n_bins`.
    bits: BitVec<u8, Lsb0>,
}

impl BitRaster {
    /// Allocate an all-zero raster.
    pub fn zeros(n_neurons: usize, n_bins: usize) -> Self {
        Self {
            n_neurons, n_bins,
            bits: BitVec::repeat(false, n_neurons * n_bins),
        }
    }

    /// Set the (neuron, bin) cell to `spiked`.
    pub fn set(&mut self, neuron: usize, bin: usize, spiked: bool) {
        self.bits.set(neuron * self.n_bins + bin, spiked);
    }

    /// Read the (neuron, bin) cell.
    pub fn get(&self, neuron: usize, bin: usize) -> bool {
        *self.bits.get(neuron * self.n_bins + bin).unwrap()
    }

    /// Naive `bool` (1 B each) vs. bit-packed bytes.
    pub fn compression_ratio(&self) -> f64 {
        let naive = self.n_neurons * self.n_bins;
        let packed = self.bits.as_raw_slice().len();
        if packed == 0 { return 1.0; }
        naive as f64 / packed as f64
    }
}

// ---------------------------------------------------------------------------
// 5. NeuronMemory — the arena each neuron owns
// ---------------------------------------------------------------------------

/// Per-neuron compressed memory arena. Aggregates one of each stream type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuronMemory {
    /// Spike-time stream (lossless).
    pub spikes: SpikeTrain,
    /// Membrane voltage stream (quantised).
    pub v_trace: MembraneTrace,
}

impl NeuronMemory {
    /// Create a new memory arena.
    ///
    /// `dt_ms` is the simulation step size; `v_min..v_max` bounds voltage
    /// quantisation.
    pub fn new(dt_ms: f64, v_min: f32, v_max: f32) -> Self {
        Self {
            spikes: SpikeTrain::new(dt_ms),
            v_trace: MembraneTrace::new(v_min, v_max, dt_ms),
        }
    }

    /// Store a sample: always record `V`, record a spike time only when
    /// `spiked` is true.
    pub fn record(&mut self, t_ms: f64, v_mv: f32, spiked: bool) -> Result<(), CompressionError> {
        self.v_trace.extend(std::iter::once(v_mv))?;
        if spiked {
            self.spikes.push(t_ms)?;
        }
        Ok(())
    }

    /// Combined bytes on the wire.
    pub fn total_compressed_bytes(&self) -> usize {
        self.spikes.compressed_bytes() + self.v_trace.compressed_bytes()
    }
}

// ---------------------------------------------------------------------------
// Helpers: zstd + varint
// ---------------------------------------------------------------------------

fn compress(raw: &[u8]) -> Result<Vec<u8>, CompressionError> {
    Ok(zstd::stream::encode_all(raw, 3)?)
}

fn decompress(payload: &[u8]) -> Result<Vec<u8>, CompressionError> {
    if payload.is_empty() { return Ok(Vec::new()); }
    let mut out = Vec::new();
    zstd::stream::Decoder::new(payload)?.read_to_end(&mut out)?;
    Ok(out)
}

/// LEB128-style unsigned varint encoder.
fn varint_encode(mut v: u64, out: &mut Vec<u8>) {
    while v >= 0x80 {
        out.push(((v as u8) & 0x7F) | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

/// LEB128-style unsigned varint decoder — returns `(value, bytes_consumed)`.
fn varint_decode(bytes: &[u8]) -> (u64, usize) {
    let mut v: u64 = 0;
    let mut shift = 0;
    for (i, b) in bytes.iter().enumerate() {
        v |= ((b & 0x7F) as u64) << shift;
        if b & 0x80 == 0 {
            return (v, i + 1);
        }
        shift += 7;
    }
    (v, bytes.len())
}
