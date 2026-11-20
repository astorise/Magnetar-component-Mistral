//! Mistral dense decoder graph construction for Magnetar.
//!
//! The host-independent graph plan is testable without a WebAssembly runtime.
//! The WIT adapter is enabled for WebAssembly or the `bindings` feature.

pub mod plan;

#[cfg(any(target_arch = "wasm32", feature = "bindings"))]
mod component;
