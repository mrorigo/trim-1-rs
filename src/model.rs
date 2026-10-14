// Rust guideline compliant 2026-10-14
//! Embedded LightGBM forest parsing and deterministic inference.

use std::error::Error;
use std::fmt;

use crate::features::{FEATURE_NAMES, N_FEATURES};

const MODEL: &[u8] = include_bytes!("../model.bin");

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelError(String);

impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl Error for ModelError {}

#[derive(Debug, Clone)]
struct Node {
    threshold: f32,
    feature: usize,
    left: i32,
    right: i32,
}

/// A validated forest used to score feature rows.
#[derive(Debug, Clone)]
pub struct Forest {
    roots: Vec<i32>,
    nodes: Vec<Node>,
    leaves: Vec<f64>,
}

impl Forest {
    /// Parses and validates a serialized `KEEP` model.
    ///
    /// # Errors
    ///
    /// Returns an error if the header, feature names, bounds, or body is invalid.
    pub fn parse(bytes: &[u8]) -> Result<Self, ModelError> {
        if bytes.len() < 24 {
            return Err(ModelError("trim model header: truncated".into()));
        }
        if &bytes[..4] != b"KEEP" || read_u32(bytes, 4)? != 1 {
            return Err(ModelError("trim model: bad magic or version".into()));
        }
        let features = read_u32(bytes, 8)? as usize;
        let trees = read_u32(bytes, 12)? as usize;
        let node_count = read_u32(bytes, 16)? as usize;
        let leaf_count = read_u32(bytes, 20)? as usize;
        if features != N_FEATURES || trees == 0 || node_count > 1 << 24 || leaf_count > 1 << 24 {
            return Err(ModelError(format!(
                "trim model: {features} features, {trees} trees, {node_count} nodes, {leaf_count} leaves (package computes {N_FEATURES} features)"
            )));
        }
        let mut expected = 24usize
            .checked_add(trees.checked_mul(4).ok_or_else(overflow_error)?)
            .and_then(|n| n.checked_add(node_count.checked_mul(20)?))
            .and_then(|n| n.checked_add(leaf_count.checked_mul(8)?))
            .ok_or_else(overflow_error)?;
        for name in FEATURE_NAMES {
            expected = expected
                .checked_add(2 + name.len())
                .ok_or_else(overflow_error)?;
        }
        if expected != bytes.len() {
            return Err(ModelError(format!(
                "trim model: header implies {expected} bytes, model has {}",
                bytes.len()
            )));
        }

        let mut cursor = 24;
        for (i, expected_name) in FEATURE_NAMES.iter().enumerate() {
            let len = read_u16(bytes, cursor)? as usize;
            cursor += 2;
            let end = cursor.checked_add(len).ok_or_else(overflow_error)?;
            let name = bytes
                .get(cursor..end)
                .ok_or_else(|| ModelError("trim model feature names: truncated".into()))?;
            if name != expected_name.as_bytes() {
                return Err(ModelError(format!(
                    "trim model feature {i} does not match package feature name"
                )));
            }
            cursor = end;
        }

        let mut roots = Vec::with_capacity(trees);
        for _ in 0..trees {
            roots.push(read_i32(bytes, cursor)?);
            cursor += 4;
        }
        let mut features = Vec::with_capacity(node_count);
        let mut thresholds = Vec::with_capacity(node_count);
        let mut left = Vec::with_capacity(node_count);
        let mut right = Vec::with_capacity(node_count);
        for _ in 0..node_count {
            features.push(read_i32(bytes, cursor)?);
            cursor += 4;
        }
        for _ in 0..node_count {
            thresholds.push(read_f64(bytes, cursor)?);
            cursor += 8;
        }
        for _ in 0..node_count {
            left.push(read_i32(bytes, cursor)?);
            cursor += 4;
        }
        for _ in 0..node_count {
            right.push(read_i32(bytes, cursor)?);
            cursor += 4;
        }
        let mut leaves = Vec::with_capacity(leaf_count);
        for _ in 0..leaf_count {
            leaves.push(read_f64(bytes, cursor)?);
            cursor += 8;
        }
        if cursor != bytes.len() {
            return Err(ModelError(format!(
                "trim model: {} trailing bytes",
                bytes.len() - cursor
            )));
        }

        let node_bound = i32::try_from(node_count).map_err(|_| overflow_error())?;
        let leaf_bound = i32::try_from(leaf_count).map_err(|_| overflow_error())?;
        let valid = |child: i32| {
            (child >= 0 && child < node_bound)
                || (child < 0 && (!child) >= 0 && (!child) < leaf_bound)
        };
        for &root in &roots {
            if !valid(root) {
                return Err(ModelError("trim model: root out of range".into()));
            }
        }
        let mut nodes = Vec::with_capacity(node_count);
        for i in 0..node_count {
            let feature = features[i];
            let l = left[i];
            let r = right[i];
            let threshold = thresholds[i];
            if feature < 0
                || feature as usize >= N_FEATURES
                || !valid(l)
                || !valid(r)
                || l >= 0 && l <= i as i32
                || r >= 0 && r <= i as i32
                || threshold.is_nan()
            {
                return Err(ModelError(format!("trim model: node {i} out of range")));
            }
            let mut threshold32 = threshold as f32;
            if f64::from(threshold32) > threshold {
                threshold32 = next_down(threshold32);
            }
            nodes.push(Node {
                threshold: threshold32,
                feature: feature as usize,
                left: l,
                right: r,
            });
        }
        Ok(Self {
            roots,
            nodes,
            leaves,
        })
    }

    /// Returns the ordered sum of tree leaf values for a feature row.
    pub fn raw(&self, row: &[f32]) -> f64 {
        let mut sum = 0.0;
        for &root in &self.roots {
            let mut index = root;
            while index >= 0 {
                let node = &self.nodes[index as usize];
                index = if row[node.feature] <= node.threshold {
                    node.left
                } else {
                    node.right
                };
            }
            sum += self.leaves[(!index) as usize];
        }
        sum
    }

    /// Converts the raw forest score to the model's keep probability.
    pub fn probability(&self, row: &[f32]) -> f64 {
        1.0 / (1.0 + portable_exp(-self.raw(row)))
    }

    pub(crate) fn roots_len(&self) -> usize {
        self.roots.len()
    }
    pub(crate) fn nodes_len(&self) -> usize {
        self.nodes.len()
    }
}

pub(crate) fn embedded() -> Result<Forest, ModelError> {
    Forest::parse(MODEL)
}

fn overflow_error() -> ModelError {
    ModelError("trim model: size overflow".into())
}

fn read_u16(bytes: &[u8], offset: usize) -> Result<u16, ModelError> {
    let b = bytes
        .get(offset..offset + 2)
        .ok_or_else(|| ModelError("trim model body: truncated".into()))?;
    Ok(u16::from_le_bytes([b[0], b[1]]))
}

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32, ModelError> {
    let b = bytes
        .get(offset..offset + 4)
        .ok_or_else(|| ModelError("trim model body: truncated".into()))?;
    Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
}

fn read_i32(bytes: &[u8], offset: usize) -> Result<i32, ModelError> {
    Ok(read_u32(bytes, offset)? as i32)
}

fn read_f64(bytes: &[u8], offset: usize) -> Result<f64, ModelError> {
    Ok(f64::from_bits(u64::from_le_bytes(
        bytes
            .get(offset..offset + 8)
            .ok_or_else(|| ModelError("trim model body: truncated".into()))?
            .try_into()
            .map_err(|_| ModelError("trim model body: truncated".into()))?,
    )))
}

fn next_down(value: f32) -> f32 {
    if value.is_nan() || value == f32::NEG_INFINITY {
        return value;
    }
    if value == 0.0 {
        return -f32::from_bits(1);
    }
    f32::from_bits(if value > 0.0 {
        value.to_bits() - 1
    } else {
        value.to_bits() + 1
    })
}

#[allow(clippy::excessive_precision, clippy::approx_constant)]
fn portable_exp(x: f64) -> f64 {
    const LN2_HI: f64 = 6.93147180369123816490e-01;
    const LN2_LO: f64 = 1.90821492927058770002e-10;
    const LOG2_E: f64 = 1.44269504088896338700e+00;
    const OVERFLOW: f64 = 7.09782712893383973096e+02;
    const UNDERFLOW: f64 = -7.45133219101941108420e+02;
    const NEAR_ZERO: f64 = 1.0 / (1 << 28) as f64;

    if x.is_nan() {
        return x;
    }
    if x > OVERFLOW {
        return f64::INFINITY;
    }
    if x < UNDERFLOW {
        return 0.0;
    }
    if -NEAR_ZERO < x && x < NEAR_ZERO {
        return 1.0 + x;
    }
    let k = if x < 0.0 {
        (LOG2_E * x - 0.5) as i32
    } else if x > 0.0 {
        (LOG2_E * x + 0.5) as i32
    } else {
        0
    };
    let hi = x - f64::from(k) * LN2_HI;
    let lo = f64::from(k) * LN2_LO;
    expmulti(hi, lo, k)
}

#[allow(clippy::excessive_precision)]
fn expmulti(hi: f64, lo: f64, k: i32) -> f64 {
    const P1: f64 = 1.66666666666666657415e-01;
    const P2: f64 = -2.77777777770155933842e-03;
    const P3: f64 = 6.61375632143793436117e-05;
    const P4: f64 = -1.65339022054652515390e-06;
    const P5: f64 = 4.13813679705723846039e-08;
    let r = hi - lo;
    let t = r * r;
    let c = r - t * (P1 + t * (P2 + t * (P3 + t * (P4 + t * P5))));
    let y = 1.0 - ((lo - r * c / (2.0 - c)) - hi);
    y * 2.0_f64.powi(k)
}
