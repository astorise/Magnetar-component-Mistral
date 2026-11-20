use super::require_checkpoint_contracts;

#[test]
fn refuses_to_build_when_activation_and_uniform_window_are_not_exposed() {
    let error = require_checkpoint_contracts().unwrap_err();
    assert!(error.contains("#217"));
    assert!(error.contains("#218"));
}
