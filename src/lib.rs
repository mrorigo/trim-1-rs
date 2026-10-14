// Rust guideline compliant 2026-10-14
//! Query-aware context compression using the embedded trim-1 line model.

mod bm25;
mod features;
mod model;
mod policy;

pub use features::{FEATURE_NAMES, line_features};
pub use model::Forest;
pub use policy::{DEFAULT_POINT, Options, POINTS, Policy, Result, compress, score};

/// Describes the embedded model for diagnostics.
pub fn model_info() -> std::result::Result<String, String> {
    let model = model::embedded().map_err(|e| e.to_string())?;
    Ok(format!(
        "{} trees, {} nodes, {} features, {} bytes",
        model.roots_len(),
        model.nodes_len(),
        features::N_FEATURES,
        include_bytes!("../model.bin").len()
    ))
}
