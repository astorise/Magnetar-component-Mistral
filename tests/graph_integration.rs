use magnetar_component_mistral::plan::{
    Attr, Config, GraphSink, KvRole, Layer, Node, Phase, Segment, Value, build_graph,
};

#[derive(Default)]
struct RecordingSink {
    begin: Option<(Phase, u64, u64)>,
    nodes: Vec<Node>,
    inputs: Vec<String>,
    aliases: Vec<(String, String)>,
    output: Option<String>,
}

impl RecordingSink {
    fn node(&self, id: &str) -> &Node {
        self.nodes
            .iter()
            .find(|node| node.id == id)
            .unwrap_or_else(|| panic!("missing node {id}"))
    }
}

impl GraphSink for RecordingSink {
    fn begin_graph(&mut self, phase: Phase, length: u64, offset: u64) {
        self.begin = Some((phase, length, offset));
    }
    fn declare_input(&mut self, name: &str, _shape: &[u64]) -> String {
        self.inputs.push(name.into());
        format!("input:{name}")
    }
    fn weight_edge(&mut self, name: &str) -> String {
        format!("weight:{name}")
    }
    fn alias_weight_edge(&mut self, edge: &str, target: &str) {
        self.aliases.push((edge.into(), target.into()));
    }
    fn kv_resource(&mut self, layer: u64, role: KvRole) -> String {
        format!("{layer}:{role:?}")
    }
    fn add_node(&mut self, node: Node) -> String {
        let result = format!("out:{}", node.id);
        self.nodes.push(node);
        result
    }
    fn finish_graph(&mut self, output: &str) -> String {
        self.output = Some(output.into());
        output.into()
    }
}

fn attr<'a>(node: &'a Node, name: &str) -> Option<&'a Value> {
    node.attrs
        .iter()
        .find_map(|Attr { name: n, value }| (*n == name).then_some(value))
}

fn tiny() -> Config {
    Config {
        hidden: 8,
        heads: 4,
        vocabulary: 17,
        epsilon: 1e-6,
        hidden_activation: "silu".into(),
        tie_embeddings: false,
        layers: vec![Layer {
            window: Some(3),
            head_dimension: 4,
            kv_heads: 2,
            rope_theta: 10_000.0,
            intermediate_size: 12,
            partial_rotary: None,
            kv_source: None,
            value_from_key: false,
        }],
    }
}

fn segment(phase: Phase, start: u32, end: u32) -> Segment {
    Segment {
        phase,
        sequence_length: if phase == Phase::Decode { 1 } else { 3 },
        position_offset: if phase == Phase::Decode { 11 } else { 0 },
        start_layer: start,
        end_layer: end,
    }
}

#[test]
fn prefill_uses_mistral_pre_norm_attention_and_pre_norm_silu_mlp() {
    let mut sink = RecordingSink::default();
    build_graph(&mut sink, &tiny(), segment(Phase::Prefill, 0, 1)).unwrap();
    assert_eq!(sink.begin, Some((Phase::Prefill, 3, 0)));
    assert_eq!(
        sink.node("layer0.input_norm").inputs,
        ["out:embedding", "weight:layers.0.input_norm"]
    );
    for projection in ["q_proj", "k_proj", "v_proj"] {
        assert_eq!(
            sink.node(&format!("layer0.{projection}")).inputs[0],
            "out:layer0.input_norm"
        );
    }
    assert_eq!(
        sink.node("layer0.residual1").inputs,
        ["out:embedding", "out:layer0.o_proj"]
    );
    assert_eq!(
        sink.node("layer0.post_attn_norm").inputs[0],
        "out:layer0.residual1"
    );
    assert_eq!(
        sink.node("layer0.gate_proj").inputs[0],
        "out:layer0.post_attn_norm"
    );
    assert_eq!(
        sink.node("layer0.up_proj").inputs[0],
        "out:layer0.post_attn_norm"
    );
    assert_eq!(sink.node("layer0.silu").inputs[0], "out:layer0.gate_proj");
    assert_eq!(
        sink.node("layer0.residual2").inputs,
        ["out:layer0.residual1", "out:layer0.down_proj"]
    );
    assert!(
        !sink
            .nodes
            .iter()
            .any(|node| node.id == "layer0.q_norm" || node.id == "layer0.post_ffn_norm")
    );
    assert_eq!(sink.output.as_deref(), Some("out:lm_head"));
}

#[test]
fn published_mistral_7b_v01_uses_a_4096_token_window_on_every_layer() {
    let raw: serde_json::Value =
        serde_json::from_slice(include_bytes!("fixtures/mistral-7b-v01.config.json")).unwrap();
    assert_eq!(raw["model_type"], "mistral");
    assert_eq!(raw["hidden_act"], "silu");
    let heads = raw["num_attention_heads"].as_u64().unwrap() as u32;
    let hidden = raw["hidden_size"].as_u64().unwrap();
    let count = raw["num_hidden_layers"].as_u64().unwrap() as usize;
    let layer = Layer {
        window: Some(raw["sliding_window"].as_u64().unwrap() as u32),
        head_dimension: hidden / u64::from(heads),
        kv_heads: raw["num_key_value_heads"].as_u64().unwrap() as u32,
        rope_theta: raw["rope_theta"].as_f64().unwrap(),
        intermediate_size: raw["intermediate_size"].as_u64().unwrap(),
        partial_rotary: None,
        kv_source: None,
        value_from_key: false,
    };
    let config = Config {
        hidden,
        heads,
        vocabulary: raw["vocab_size"].as_u64().unwrap(),
        epsilon: raw["rms_norm_eps"].as_f64().unwrap(),
        hidden_activation: raw["hidden_act"].as_str().unwrap().into(),
        tie_embeddings: raw["tie_word_embeddings"].as_bool().unwrap(),
        layers: vec![layer; count],
    };
    let mut sink = RecordingSink::default();
    build_graph(&mut sink, &config, segment(Phase::Prefill, 0, count as u32)).unwrap();
    for index in 0..count {
        assert_eq!(
            attr(sink.node(&format!("layer{index}.attention")), "window_size"),
            Some(&Value::Int(4096))
        );
    }
}

#[test]
fn prefill_and_decode_record_kv_and_full_attention_has_no_window() {
    let mut cfg = tiny();
    cfg.layers[0].window = None;
    for phase in [Phase::Prefill, Phase::Decode] {
        let mut sink = RecordingSink::default();
        build_graph(&mut sink, &cfg, segment(phase, 0, 1)).unwrap();
        let append = phase == Phase::Decode;
        assert_eq!(
            sink.node("layer0.v_proj").kv.as_ref().unwrap().append,
            append
        );
        assert_eq!(
            sink.node("layer0.rope_k").kv.as_ref().unwrap().append,
            append
        );
        assert_eq!(
            sink.node("layer0.v_proj").kv.as_ref().unwrap().role,
            KvRole::V
        );
        assert_eq!(
            sink.node("layer0.rope_k").kv.as_ref().unwrap().role,
            KvRole::K
        );
        assert_eq!(attr(sink.node("layer0.attention"), "window_size"), None);
        assert_eq!(
            attr(sink.node("layer0.rope_q"), "rotation_pairing"),
            Some(&Value::Text("half_split".into()))
        );
    }
}

#[test]
fn middle_segments_use_hidden_state_and_tied_final_segment_aliases_embedding() {
    let mut cfg = tiny();
    cfg.layers.push(cfg.layers[0].clone());
    cfg.layers.push(cfg.layers[0].clone());
    cfg.tie_embeddings = true;
    let mut sink = RecordingSink::default();
    build_graph(&mut sink, &cfg, segment(Phase::Decode, 1, 2)).unwrap();
    assert_eq!(sink.begin, Some((Phase::Decode, 1, 11)));
    assert_eq!(sink.inputs, ["hidden_states_in"]);
    assert_eq!(sink.output.as_deref(), Some("out:layer1.residual2"));
    let mut last = RecordingSink::default();
    build_graph(&mut last, &cfg, segment(Phase::Decode, 2, 3)).unwrap();
    assert_eq!(
        last.aliases,
        [("weight:lm_head".into(), "token_embedding".into())]
    );
}

#[test]
fn unsupported_activation_and_invalid_geometry_are_rejected_before_host_calls() {
    for mutation in 0..5 {
        let mut cfg = tiny();
        match mutation {
            0 => cfg.layers[0].kv_heads = 3,
            1 => cfg.layers[0].window = Some(0),
            2 => cfg.layers[0].partial_rotary = Some(0.5),
            3 => cfg.layers[0].kv_source = Some(0),
            _ => cfg.hidden_activation = "gelu".into(),
        }
        let mut sink = RecordingSink::default();
        assert!(build_graph(&mut sink, &cfg, segment(Phase::Prefill, 0, 1)).is_err());
        assert!(sink.begin.is_none());
    }
}
