//! Mistral dense decoder graph construction, independent of the WIT host.
//!
//! Mistral 7B applies RMSNorm before attention and before the SiLU-gated MLP.
//! Its earlier checkpoints use causal sliding-window attention.

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Phase {
    Prefill,
    Decode,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KvRole {
    K,
    V,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Bool(bool),
    Int(i64),
    Float(f64),
    Text(String),
}

#[derive(Clone, Debug, PartialEq)]
pub struct Attr {
    pub name: &'static str,
    pub value: Value,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Kv {
    pub resource: String,
    pub role: KvRole,
    pub append: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Node {
    pub id: String,
    pub operator: &'static str,
    pub family: &'static str,
    pub attrs: Vec<Attr>,
    pub inputs: Vec<String>,
    pub shape: Vec<u64>,
    pub kv: Option<Kv>,
}

/// Subset of the graph-builder capability needed by Mistral.
pub trait GraphSink {
    fn begin_graph(&mut self, phase: Phase, sequence_length: u64, position_offset: u64);
    fn declare_input(&mut self, name: &str, shape: &[u64]) -> String;
    fn weight_edge(&mut self, name: &str) -> String;
    fn alias_weight_edge(&mut self, edge: &str, target: &str);
    fn kv_resource(&mut self, layer: u64, role: KvRole) -> String;
    fn add_node(&mut self, node: Node) -> String;
    fn finish_graph(&mut self, output: &str) -> String;
}

#[derive(Clone, Debug, PartialEq)]
pub struct Layer {
    pub window: Option<u32>,
    pub head_dimension: u64,
    pub kv_heads: u32,
    pub rope_theta: f64,
    pub intermediate_size: u64,
    pub partial_rotary: Option<f64>,
    pub kv_source: Option<u32>,
    pub value_from_key: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Config {
    pub hidden: u64,
    pub heads: u32,
    pub vocabulary: u64,
    pub epsilon: f64,
    /// Checkpoint-declared MLP activation. Only `silu` is supported.
    pub hidden_activation: String,
    pub tie_embeddings: bool,
    pub layers: Vec<Layer>,
}

impl Config {
    /// Fail before building any graph when a geometry is not supported.
    pub fn validate(&self) -> Result<(), String> {
        if self.hidden == 0 || self.vocabulary == 0 || self.heads == 0 || self.layers.is_empty() {
            return Err("zero model dimension or empty layer layout".into());
        }
        if self.hidden_activation != "silu" {
            return Err(format!(
                "unsupported hidden activation {:?}",
                self.hidden_activation
            ));
        }
        if !self.epsilon.is_finite() || self.epsilon <= 0.0 {
            return Err("RMSNorm epsilon must be finite and positive".into());
        }
        for (index, layer) in self.layers.iter().enumerate() {
            if layer.head_dimension == 0
                || layer.head_dimension % 2 != 0
                || layer.intermediate_size == 0
            {
                return Err(format!("layer {index}: invalid head or MLP dimension"));
            }
            if layer.kv_heads == 0 || !self.heads.is_multiple_of(layer.kv_heads) {
                return Err(format!("layer {index}: KV heads must divide query heads"));
            }
            if !layer.rope_theta.is_finite() || layer.rope_theta <= 0.0 || layer.window == Some(0) {
                return Err(format!(
                    "layer {index}: invalid RoPE base or attention window"
                ));
            }
            if layer.partial_rotary.is_some() || layer.kv_source.is_some() || layer.value_from_key {
                return Err(format!(
                    "layer {index}: partial rotation or shared/derived KV is not supported"
                ));
            }
            if u64::from(self.heads)
                .checked_mul(layer.head_dimension)
                .is_none()
                || u64::from(layer.kv_heads)
                    .checked_mul(layer.head_dimension)
                    .is_none()
                || layer.head_dimension > i64::MAX as u64
            {
                return Err(format!("layer {index}: attention dimensions overflow"));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Segment {
    pub phase: Phase,
    pub sequence_length: u64,
    pub position_offset: u64,
    pub start_layer: u32,
    pub end_layer: u32,
}

fn attr(name: &'static str, value: Value) -> Attr {
    Attr { name, value }
}

struct Builder<'a, S> {
    sink: &'a mut S,
    config: &'a Config,
    sequence: u64,
    append: bool,
    position_offset: i64,
}

impl<S: GraphSink> Builder<'_, S> {
    #[allow(clippy::too_many_arguments)]
    fn node(
        &mut self,
        id: String,
        operator: &'static str,
        family: &'static str,
        attrs: Vec<Attr>,
        inputs: Vec<String>,
        width: u64,
        kv: Option<Kv>,
    ) -> String {
        self.sink.add_node(Node {
            id,
            operator,
            family,
            attrs,
            inputs,
            shape: vec![self.sequence, width],
            kv,
        })
    }

    fn matmul(
        &mut self,
        prefix: &str,
        suffix: &str,
        input: &str,
        weight: &str,
        width: u64,
        kv: Option<Kv>,
    ) -> String {
        let weight = self.sink.weight_edge(weight);
        self.node(
            format!("{prefix}.{suffix}"),
            "matmul",
            "LinearAlgebra",
            vec![],
            vec![input.into(), weight],
            width,
            kv,
        )
    }

    fn norm(
        &mut self,
        prefix: &str,
        suffix: &str,
        input: &str,
        weight: &str,
        width: u64,
    ) -> String {
        let weight = self.sink.weight_edge(weight);
        self.node(
            format!("{prefix}.{suffix}"),
            "rmsnorm",
            "Normalization",
            vec![attr("epsilon", Value::Float(self.config.epsilon))],
            vec![input.into(), weight],
            width,
            None,
        )
    }

    fn kv(&mut self, index: u64, role: KvRole) -> Kv {
        Kv {
            resource: self.sink.kv_resource(index, role),
            role,
            append: self.append,
        }
    }

    fn rope(
        &mut self,
        prefix: &str,
        suffix: &str,
        input: &str,
        layer: &Layer,
        heads: u64,
        kv: Option<Kv>,
    ) -> String {
        let attrs = vec![
            attr("base", Value::Float(layer.rope_theta)),
            attr("dimension", Value::Int(layer.head_dimension as i64)),
            attr("position_mode", Value::Text("sequential".into())),
            attr("rotation_pairing", Value::Text("half_split".into())),
            attr("position_offset", Value::Int(self.position_offset)),
            attr("head_count", Value::Int(heads as i64)),
        ];
        self.node(
            format!("{prefix}.{suffix}"),
            "rope",
            "PositionEncoding",
            attrs,
            vec![input.into()],
            heads * layer.head_dimension,
            kv,
        )
    }

    fn layer(&mut self, index: u64, layer: &Layer, input: &str) -> String {
        let p = format!("layer{index}");
        let w = |suffix: &str| format!("layers.{index}.{suffix}");
        let qdim = u64::from(self.config.heads) * layer.head_dimension;
        let kdim = u64::from(layer.kv_heads) * layer.head_dimension;
        let normalized = self.norm(
            &p,
            "input_norm",
            input,
            &w("input_norm"),
            self.config.hidden,
        );
        let q = self.matmul(
            &p,
            "q_proj",
            &normalized,
            &w("self_attn.q_proj"),
            qdim,
            None,
        );
        let k = self.matmul(
            &p,
            "k_proj",
            &normalized,
            &w("self_attn.k_proj"),
            kdim,
            None,
        );
        let vkv = self.kv(index, KvRole::V);
        let v = self.matmul(
            &p,
            "v_proj",
            &normalized,
            &w("self_attn.v_proj"),
            kdim,
            Some(vkv),
        );
        let q = self.rope(&p, "rope_q", &q, layer, self.config.heads.into(), None);
        let kkv = self.kv(index, KvRole::K);
        let k = self.rope(&p, "rope_k", &k, layer, layer.kv_heads.into(), Some(kkv));
        let mut attrs = vec![
            attr("causal", Value::Bool(true)),
            attr("head_count", Value::Int(self.config.heads.into())),
            attr("kv_head_count", Value::Int(layer.kv_heads.into())),
            attr("head_dimension", Value::Int(layer.head_dimension as i64)),
            attr("attention_mask_kind", Value::Text("causal".into())),
        ];
        if let Some(window) = layer.window {
            attrs.push(attr("window_size", Value::Int(window.into())));
        }
        let attention = self.node(
            format!("{p}.attention"),
            "attention",
            "Attention",
            attrs,
            vec![q, k, v],
            qdim,
            None,
        );
        let proj = self.matmul(
            &p,
            "o_proj",
            &attention,
            &w("self_attn.o_proj"),
            self.config.hidden,
            None,
        );
        let h = self.node(
            format!("{p}.residual1"),
            "residual-add",
            "Tensor",
            vec![],
            vec![input.into(), proj],
            self.config.hidden,
            None,
        );
        let mlp_input = self.norm(
            &p,
            "post_attn_norm",
            &h,
            &w("post_attn_norm"),
            self.config.hidden,
        );
        let gate = self.matmul(
            &p,
            "gate_proj",
            &mlp_input,
            &w("mlp.gate_proj"),
            layer.intermediate_size,
            None,
        );
        let up = self.matmul(
            &p,
            "up_proj",
            &mlp_input,
            &w("mlp.up_proj"),
            layer.intermediate_size,
            None,
        );
        let gate = self.node(
            format!("{p}.silu"),
            "silu",
            "Activation",
            vec![],
            vec![gate],
            layer.intermediate_size,
            None,
        );
        let mixed = self.node(
            format!("{p}.mul"),
            "mul",
            "Tensor",
            vec![],
            vec![gate, up],
            layer.intermediate_size,
            None,
        );
        let down = self.matmul(
            &p,
            "down_proj",
            &mixed,
            &w("mlp.down_proj"),
            self.config.hidden,
            None,
        );
        self.node(
            format!("{p}.residual2"),
            "residual-add",
            "Tensor",
            vec![],
            vec![h, down],
            self.config.hidden,
            None,
        )
    }
}

/// Build a prefill or decode graph, including partial layer segments.
pub fn build_graph<S: GraphSink>(
    sink: &mut S,
    config: &Config,
    segment: Segment,
) -> Result<String, String> {
    config.validate()?;
    if segment.sequence_length == 0
        || segment.position_offset > i64::MAX as u64
        || segment.start_layer > segment.end_layer
        || segment.end_layer as usize > config.layers.len()
        || (segment.phase == Phase::Decode && segment.sequence_length != 1)
    {
        return Err("invalid sequence length or layer segment".into());
    }
    sink.begin_graph(
        segment.phase,
        segment.sequence_length,
        segment.position_offset,
    );
    let mut b = Builder {
        sink,
        config,
        sequence: segment.sequence_length,
        append: segment.phase == Phase::Decode,
        position_offset: segment.position_offset as i64,
    };
    let mut hidden = if segment.start_layer == 0 {
        let ids = b
            .sink
            .declare_input("token_ids", &[segment.sequence_length]);
        let table = b.sink.weight_edge("token_embedding");
        b.node(
            "embedding".into(),
            "embedding",
            "Tensor",
            vec![],
            vec![ids, table],
            config.hidden,
            None,
        )
    } else {
        b.sink.declare_input(
            "hidden_states_in",
            &[segment.sequence_length, config.hidden],
        )
    };
    for index in segment.start_layer..segment.end_layer {
        hidden = b.layer(index.into(), &config.layers[index as usize], &hidden);
    }
    if segment.end_layer as usize == config.layers.len() {
        let weight = b.sink.weight_edge("final_norm");
        hidden = b.node(
            "final_norm".into(),
            "rmsnorm",
            "Normalization",
            vec![attr("epsilon", Value::Float(config.epsilon))],
            vec![hidden, weight],
            config.hidden,
            None,
        );
        let weight = b.sink.weight_edge("lm_head");
        if config.tie_embeddings {
            b.sink.weight_edge("token_embedding");
            b.sink.alias_weight_edge(&weight, "token_embedding");
        }
        hidden = b.node(
            "lm_head".into(),
            "matmul",
            "LinearAlgebra",
            vec![],
            vec![hidden, weight],
            config.vocabulary,
            None,
        );
    }
    Ok(b.sink.finish_graph(&hidden))
}
