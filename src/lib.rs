//! Mistral AI Model Component for the Magnetar local AI Runtime.
//!
//! An empty template: no graph logic and no Component Model exports yet.
//! See the README for what this family needs from the Runtime before a
//! real implementation of `magnetar:model-component-graph` can be written.

/// The architecture family this Component will serve.
pub const ARCHITECTURE_FAMILY: &str = "mistral";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_its_architecture_family() {
        assert_eq!(ARCHITECTURE_FAMILY, "mistral");
    }
}
