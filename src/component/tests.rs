use super::validate_activation;

#[test]
fn requires_the_supported_declared_activation() {
    assert_eq!(validate_activation(Some("silu".into())).unwrap(), "silu");
    assert!(validate_activation(None).is_err());
    assert!(validate_activation(Some("gelu".into())).is_err());
    assert!(validate_activation(Some("".into())).is_err());
}
