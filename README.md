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

**Empty template.** This crate is a scaffold: no Mistral graph logic, no WIT
bindings and no Component Model exports exist here yet, and the Runtime has
no Mistral implementation of its own either. It is pinned into Magnetar as a
placeholder so the work has a home; it is not yet scheduled against an
OpenSpec change of its own.

## What this family needs

What sets this family apart from the generic decoder block
`components/qwen` and `components/llama` already implement, each to be
checked against the exact pinned checkpoint's `config.json` before it is
implemented:

- Mistral 7B is the same pre-norm, RoPE, grouped-query-attention, SwiGLU decoder block as Llama, plus sliding-window attention in its first release: the attention Kernels, step capture bounds and KV store sizing must honour a window.
- Mixtral replaces the MLP with a sparse Mixture-of-Experts (top-2 of 8 experts per token), which needs data-dependent routing on the Device (`execute-and-cache-moe-experts`).
- Later releases (Mistral NeMo, Mistral Small) keep the dense block with a larger, different tokenizer (Tekken), which the Hugging Face and GGUF loaders must ingest.

## Governing contract

None dedicated yet. A real implementation will target the same
`model-component-graph-contract` WIT interface (`magnetar:model-component-graph`)
`components/qwen` and `components/llama` implement, extended additively
where this family needs Operators or state the contract does not model yet.
Family-specific requirements will be scoped in the main repository's
`openspec/` (the `qualify-additional-model-and-state-families` change tracks
qualifying new families).

## Relationship to magnetar-runtime

This Component will be loaded and validated by the Runtime's Component
Engine (Wasmtime-backed on native targets, browser-hosted on `wasm32`) and
never run with direct hardware, filesystem, or network access. It is
pinned into the main [Magnetar](https://github.com/astorise/Magnetar)
repository as a git submodule at `components/mistral`.
