# magnetar-component-mistral

## Purpose

A WebAssembly Component implementing the Mistral AI model families for the
[Magnetar](https://github.com/astorise/Magnetar) local AI Runtime. It will
supply prefill/decode graph semantics for Mistral-compatible Model Artifacts
through the Runtime's Model Component contract, without ever selecting a
Provider, Device, or receiving native Runtime handles -- per Magnetar's
architectural invariants (Components request Capabilities, not Providers
or Devices).

## Status

The initial dense Mistral graph builder covers prefill, cached decode and
layer segments, including sliding-window or full attention. It implements the
pre-norm attention and pre-norm SiLU-gated MLP ordering. Graph tests exercise
the pinned Mistral 7B v0.1 configuration and reject unsupported inputs.

The WIT entrypoints require the Runtime to supply a verified `silu` activation
and preserve the configured per-layer attention windows. Missing or different
activations are refused. The Magnetar repository includes tiny numerical
fixtures for full and sliding attention; real Mistral checkpoints and GPU
execution have not yet been qualified with this Component. Mixtral and later
variants are not implemented.

## Mistral 7B today

Mistral 7B v0.2/v0.3 has no sliding window and is exactly the Llama
decoder block, and llama.cpp's GGUF exports of it declare
`general.architecture = llama`. Such a file is therefore served by
`components/llama`, through `loaders/gguf` (which restores llama.cpp's
query/key row permutation and rebuilds its SentencePiece tokenizer):
Mistral-7B-Instruct-v0.3 `Q8_0` is qualified that way on CUDA. This
Component is for what the `mistral` family (Hugging Face `model_type`)
adds beyond that block.

## What this family needs

What sets this family apart from the generic decoder block
`components/qwen` and `components/llama` already implement, each to be
checked against the exact pinned checkpoint's `config.json` before it is
implemented:

- Mistral 7B is the same pre-norm, RoPE, grouped-query-attention, SwiGLU decoder block as Llama, plus sliding-window attention in its first release: the attention Kernels, step capture bounds and KV store sizing must honour a window.
- Mixtral replaces the MLP with a sparse Mixture-of-Experts (top-2 of 8 experts per token), which needs data-dependent routing on the Device (`execute-and-cache-moe-experts`).
- Later releases (Mistral NeMo, Mistral Small) keep the dense block with a larger, different tokenizer (Tekken), which the Hugging Face and GGUF loaders must ingest.

## Built on magnetar-component-sdk

The graph IR (`Phase`, `KvRole`, `Value`, `Attr`, `Kv`, `Node`), the
`GraphSink` the builder writes to, and the adapter to the contract's
`graph-builder` interface come from
[`magnetar-component-sdk`](https://github.com/astorise/magnetar-component-sdk),
pinned by commit while the SDK is in development (the crate is published once
it is stable). `src/plan.rs` re-exports the SDK types and keeps only what is
specific to Mistral: the configuration, its validation and the graph itself.

`tests/golden/` holds the graphs the builder emits, recorded before it moved
onto the SDK; `tests/golden_graphs.rs` compares every call against them.
Regenerate them deliberately with `UPDATE_GOLDEN=1 cargo test --test
golden_graphs`.

## Governing contract

The graph adapter targets `magnetar:model-component-graph@1.8.0` and requires
the `hidden-activation` and `layer-layout` capabilities where applicable.

## Relationship to magnetar-runtime

This Component will be loaded and validated by the Runtime's Component
Engine (Wasmtime-backed on native targets, browser-hosted on `wasm32`) and
never run with direct hardware, filesystem, or network access. It is
pinned into the main [Magnetar](https://github.com/astorise/Magnetar)
repository as a git submodule at `components/mistral`.
