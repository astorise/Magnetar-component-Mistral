//! Adapt the host-independent Mistral graph builder to Magnetar's WIT capability.

use crate::plan::{self, Config, Layer, Phase, Segment};
use magnetar_component_api::v1_8::{
    Guest, export,
    magnetar::model_component_graph::{graph_builder as gb, model_config as mc},
};

struct Host;

// The graph-builder adapter is the SDK's, written once for every Component.
magnetar_component_sdk::wit_graph_sink!(Host, gb);

fn config() -> Result<Config, String> {
    let a = mc::model_config();
    let hidden_activation = validate_activation(mc::hidden_activation())?;
    if a.num_hidden_layers == 0 || a.num_hidden_layers > 4096 {
        return Err("invalid or excessive decoder layer count".into());
    }
    if a.attention_bias || a.rope_scaling_factor.is_some() {
        return Err("attention bias and linear RoPE scaling are not supported".into());
    }
    // Query the additive 1.7 capabilities even for a dense model so that no
    // model capability is silently discarded.
    let moe = mc::mixture_of_experts();
    let moe_layers = mc::mixture_of_experts_layers();
    let mla = mc::multi_head_latent_attention();
    let routing = mc::expert_routing();
    if moe.is_some() || moe_layers.is_some() || mla.is_some() || routing.is_some() {
        return Err("expert or latent-attention configuration is not supported".into());
    }
    let options = mc::options();
    if options.final_logit_softcapping.is_some() || options.per_layer_input_size.is_some() {
        return Err("logit soft-capping and per-layer input embeddings are not supported".into());
    }
    if mc::rope_scaling().is_some() {
        return Err("frequency-scaled RoPE is not supported".into());
    }
    let layers = mc::layer_layout().map_or_else(
        || {
            vec![
                Layer {
                    window: None,
                    head_dimension: a.head_dimension,
                    kv_heads: a.num_key_value_heads,
                    rope_theta: a.rope_theta,
                    intermediate_size: a.intermediate_size,
                    partial_rotary: None,
                    kv_source: None,
                    value_from_key: false
                };
                a.num_hidden_layers as usize
            ]
        },
        |layers| {
            layers
                .into_iter()
                .map(|l| Layer {
                    window: l.sliding_window,
                    head_dimension: l.head_dimension,
                    kv_heads: l.num_key_value_heads,
                    rope_theta: l.rope_theta,
                    intermediate_size: l.intermediate_size,
                    partial_rotary: l.partial_rotary_factor,
                    kv_source: l.kv_source_layer,
                    value_from_key: l.value_from_key,
                })
                .collect()
        },
    );
    if layers.len() != a.num_hidden_layers as usize {
        return Err("per-layer layout length does not match the architecture".into());
    }
    Ok(Config {
        hidden: a.hidden_size,
        heads: a.num_attention_heads,
        vocabulary: a.vocabulary_size,
        epsilon: a.rms_norm_eps,
        hidden_activation,
        tie_embeddings: a.tie_word_embeddings,
        layers,
    })
}

fn build(
    phase: Phase,
    sequence_length: u64,
    position_offset: u64,
    range: Option<(u32, u32)>,
) -> String {
    let config = config().unwrap_or_else(|reason| panic!("mistral: {reason}"));
    let (start_layer, end_layer) = range.unwrap_or((0, config.layers.len() as u32));
    plan::build_graph(
        &mut Host,
        &config,
        Segment {
            phase,
            sequence_length,
            position_offset,
            start_layer,
            end_layer,
        },
    )
    .unwrap_or_else(|reason| panic!("mistral: {reason}"))
}

/// Refuse a graph unless ingestion established the activation it uses.
fn validate_activation(value: Option<String>) -> Result<String, String> {
    match value.as_deref() {
        Some("silu") => Ok("silu".into()),
        Some(other) => Err(format!("unsupported checkpoint activation {other:?}")),
        None => Err("checkpoint activation is unknown".into()),
    }
}

struct Component;

impl Guest for Component {
    fn build_prefill_graph(sequence_length: u64) -> String {
        build(Phase::Prefill, sequence_length, 0, None)
    }
    fn build_decode_graph(cached_token_count: u64) -> String {
        build(Phase::Decode, 1, cached_token_count, None)
    }
    fn build_prefill_graph_segment(
        sequence_length: u64,
        start_layer: u32,
        end_layer: u32,
    ) -> String {
        build(
            Phase::Prefill,
            sequence_length,
            0,
            Some((start_layer, end_layer)),
        )
    }
    fn build_decode_graph_segment(
        cached_token_count: u64,
        start_layer: u32,
        end_layer: u32,
    ) -> String {
        build(
            Phase::Decode,
            1,
            cached_token_count,
            Some((start_layer, end_layer)),
        )
    }
}

export!(Component);

#[cfg(test)]
mod tests;
