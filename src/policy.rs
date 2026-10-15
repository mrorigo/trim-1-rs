// Rust guideline compliant 2026-10-14
//! Compression policies, line selection, and the public scoring API.

use crate::features::{block_headers, indent_of, line_features};
use crate::model::{Forest, TrimError, embedded};

/// Default operating point used when no point is named.
pub const DEFAULT_POINT: &str = "balanced";
/// Maximum UTF-8 context size accepted by compression.
pub const MAX_CONTEXT_BYTES: usize = 8 << 20;
/// Maximum line count accepted by compression.
pub const MAX_CONTEXT_LINES: usize = 200_000;

/// Converts line scores into a selection.
///
/// The thresholds control which lines are selected, how much neighboring
/// context is retained, and when a result is considered useful enough to use.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Policy {
    /// Keep lines scoring at least this share of the best score.
    pub floor: f64,
    /// Always keep this many top-scoring nonblank lines.
    pub top_k: usize,
    /// Keep this many adjacent lines on each side of a selected line.
    pub context: usize,
    /// Keep enclosing headers and the current section opener.
    pub headers: usize,
    /// Fill removed gaps no longer than this many lines.
    pub gap: usize,
    /// Return contexts shorter than this many bytes unchanged.
    pub min_chars: usize,
    /// Return unchanged unless at least this share of bytes is removed.
    pub min_cut: f64,
}

/// Named operating points.
pub const POINTS: [(&str, Policy); 3] = [
    (
        "conservative",
        Policy {
            floor: 0.0035,
            top_k: 3,
            context: 1,
            headers: 1,
            gap: 2,
            min_chars: 2000,
            min_cut: 0.1,
        },
    ),
    (
        "balanced",
        Policy {
            floor: 0.008,
            top_k: 3,
            context: 1,
            headers: 1,
            gap: 2,
            min_chars: 2000,
            min_cut: 0.1,
        },
    ),
    (
        "aggressive",
        Policy {
            floor: 0.035,
            top_k: 3,
            context: 1,
            headers: 1,
            gap: 2,
            min_chars: 2000,
            min_cut: 0.1,
        },
    ),
];

/// Selects an operating point or a custom policy.
///
/// An empty `point` selects the balanced operating point. When `policy` is
/// present, it takes precedence over `point`.
#[derive(Debug, Clone, Copy, Default)]
pub struct Options<'a> {
    /// Point name; an empty name selects `balanced`.
    pub point: &'a str,
    /// Optional custom policy that replaces the named point.
    pub policy: Option<Policy>,
}

/// One compression outcome.
///
/// The output is either compressed text with elision markers or the original
/// context with a reason that compression was skipped.
#[derive(Debug, Clone, PartialEq)]
pub struct Result {
    /// Compressed text, or the original context when unchanged.
    pub output: String,
    /// Number of kept lines.
    pub kept_lines: usize,
    /// Number of removed lines.
    pub removed_lines: usize,
    /// Operating point name, or `custom`.
    pub point: String,
    /// Policy used for the call.
    pub policy: Policy,
    /// Whether the original context was returned unchanged.
    pub unchanged: bool,
    /// Reason for returning unchanged text.
    pub reason: String,
}

/// Scores every context line for the query.
///
/// # Arguments
///
/// * `context` — text whose lines will be scored.
/// * `query` — the request used to compute relevance features.
///
/// # Returns
///
/// Returns each input line and its keep probability in matching order.
///
/// # Errors
///
/// Returns an error if the embedded model cannot be loaded or scored.
pub fn score(
    context: &str,
    query: &str,
) -> std::result::Result<(Vec<String>, Vec<f64>), TrimError> {
    let model = embedded().map_err(Clone::clone)?;
    score_with_model(context, query, model)
}

fn score_with_model(
    context: &str,
    query: &str,
    model: &Forest,
) -> std::result::Result<(Vec<String>, Vec<f64>), TrimError> {
    let (lines, rows) = line_features(context, query);
    let scores = lines
        .iter()
        .zip(rows)
        .map(|(line, row)| {
            if line.trim().is_empty() {
                Ok(0.0)
            } else {
                model.probability(&row)
            }
        })
        .collect::<std::result::Result<Vec<_>, _>>()?;
    Ok((lines, scores))
}

/// Compresses context to lines relevant to the query.
///
/// # Arguments
///
/// * `context` — text to compress.
/// * `query` — the request used to select relevant lines.
/// * `opts` — named operating point or custom policy.
///
/// # Returns
///
/// Returns the compressed text and selection metadata. Ineligible inputs are
/// returned unchanged with a reason.
pub fn compress(context: &str, query: &str, opts: Options<'_>) -> Result {
    let (name, policy) = if let Some(policy) = opts.policy {
        ("custom".to_owned(), policy)
    } else {
        let name = if opts.point.is_empty() {
            DEFAULT_POINT
        } else {
            opts.point
        };
        match POINTS.iter().find(|(point, _)| *point == name) {
            Some((_, policy)) => (name.to_owned(), *policy),
            None => {
                return unchanged(
                    context,
                    name.to_owned(),
                    Policy {
                        floor: 0.0,
                        top_k: 0,
                        context: 0,
                        headers: 0,
                        gap: 0,
                        min_chars: 0,
                        min_cut: 0.0,
                    },
                    "unknown operating point",
                );
            }
        }
    };
    let line_count = count_lines(context);
    let refusal = if query.trim().is_empty() {
        Some("no query")
    } else if context.len() < policy.min_chars {
        Some("context below the minimum size")
    } else if context.len() > MAX_CONTEXT_BYTES {
        Some("context over 8 MiB")
    } else if line_count > MAX_CONTEXT_LINES {
        Some("context over 200,000 lines")
    } else if has_marker_line(context) {
        Some("context has a line that reads as an elision marker")
    } else {
        None
    };
    if let Some(reason) = refusal {
        return unchanged(context, name, policy, reason);
    }
    let model = match embedded() {
        Ok(model) => model,
        Err(err) => return unchanged(context, name, policy, &err.to_string()),
    };
    let (lines, scores) = match score_with_model(context, query, model) {
        Ok(scored) => scored,
        Err(reason) => return unchanged(context, name, policy, &reason.to_string()),
    };
    let (output, kept_lines) = match policy.apply(&lines, &scores) {
        Ok(result) => result,
        Err(reason) => return unchanged(context, name, policy, &reason.to_string()),
    };
    if output.len() as f64 > context.len() as f64 * (1.0 - policy.min_cut) {
        return unchanged(context, name, policy, "cut below the minimum");
    }
    Result {
        output,
        kept_lines,
        removed_lines: line_count - kept_lines,
        point: name,
        policy,
        unchanged: false,
        reason: String::new(),
    }
}

impl Policy {
    /// Applies this policy to lines and scores, returning rendered text and kept line count.
    ///
    /// The score slice must contain one value per line. A trailing empty line
    /// produced by a final newline is preserved and excluded from selection.
    ///
    /// # Errors
    ///
    /// Returns an error when the number of scores does not match the number of
    /// lines.
    pub fn apply(
        &self,
        lines: &[String],
        scores: &[f64],
    ) -> std::result::Result<(String, usize), TrimError> {
        if lines.len() != scores.len() {
            return Err(TrimError::invalid_input(format!(
                "policy: expected {} scores, got {}",
                lines.len(),
                scores.len()
            )));
        }
        let final_newline = lines.len() > 1 && lines.last().is_some_and(String::is_empty);
        let n = lines.len() - usize::from(final_newline);
        let lines = &lines[..n];
        let scores = &scores[..n];
        let blank = lines
            .iter()
            .map(|line| line.trim().is_empty())
            .collect::<Vec<_>>();
        let indent = lines.iter().map(|line| indent_of(line)).collect::<Vec<_>>();
        let mut section = vec![-1isize; n];
        let mut open = -1isize;
        for (i, line) in lines.iter().enumerate() {
            section[i] = open;
            let trimmed = line.trim();
            if section_prefix(trimmed)
                || trimmed.len() < 120 && trimmed.starts_with('[') && trimmed.ends_with(']')
            {
                open = i as isize;
            }
        }
        let keep = select_lines(scores, &blank, &indent, &section, *self);
        let (mut output, kept) = render(lines, &keep);
        if final_newline {
            output.push('\n');
        }
        Ok((output, kept))
    }
}

fn unchanged(context: &str, name: String, policy: Policy, reason: &str) -> Result {
    let count = count_lines(context);
    Result {
        output: context.to_owned(),
        kept_lines: count,
        removed_lines: 0,
        point: name,
        policy,
        unchanged: true,
        reason: reason.to_owned(),
    }
}
fn count_lines(context: &str) -> usize {
    context
        .strip_suffix('\n')
        .unwrap_or(context)
        .matches('\n')
        .count()
        + 1
}
fn has_marker_line(context: &str) -> bool {
    context.lines().any(|line| {
        let line = line.trim_matches([' ', '\t', '\r']);
        let Some(rest) = line.strip_prefix("… ") else {
            return false;
        };
        [" lines …", " line …"].iter().any(|suffix| {
            rest.strip_suffix(suffix)
                .is_some_and(|count| !count.is_empty() && count.chars().all(|c| c.is_ascii_digit()))
        })
    })
}
fn section_prefix(s: &str) -> bool {
    [
        "# ",
        "## ",
        "### ",
        "#### ",
        "[start of ",
        "==> ",
        "diff --git ",
        "File: ",
    ]
    .iter()
    .any(|p| s.starts_with(p))
}
fn select_lines(
    scores: &[f64],
    blank: &[bool],
    indent: &[usize],
    section: &[isize],
    p: Policy,
) -> Vec<bool> {
    let n = scores.len();
    let mut order = (0..n).filter(|&i| !blank[i]).collect::<Vec<_>>();
    order.sort_by(|&a, &b| {
        if scores[a] > scores[b] {
            std::cmp::Ordering::Less
        } else if scores[a] < scores[b] {
            std::cmp::Ordering::Greater
        } else {
            std::cmp::Ordering::Equal
        }
    });
    let mut seed = vec![false; n];
    if let Some(&best) = order.first() {
        for (k, &i) in order.iter().enumerate() {
            seed[i] = k < p.top_k || scores[i] >= p.floor * scores[best];
        }
    }
    let mut keep = seed.clone();
    let headers = block_headers(indent, blank);
    for i in 0..n {
        if !seed[i] {
            continue;
        }
        let lo = i.saturating_sub(p.context);
        let hi = (n - 1).min(i.saturating_add(p.context));
        for v in &mut keep[lo..=hi] {
            *v = true;
        }
        let mut h = headers[i];
        let mut k = 0;
        while h >= 0 && (k as usize) < p.headers {
            keep[h as usize] = true;
            h = headers[h as usize];
            k += 1;
        }
        if p.headers > 0 && section[i] >= 0 {
            keep[section[i] as usize] = true;
        }
    }
    let mut last = None;
    for i in 0..n {
        if !keep[i] {
            continue;
        }
        if let Some(prev) = last {
            let gap = &keep[prev + 1..i];
            if !gap.is_empty() && gap.len() <= p.gap {
                for v in &mut keep[prev + 1..i] {
                    *v = true;
                }
            }
        }
        last = Some(i);
    }
    keep
}
fn render(lines: &[String], keep: &[bool]) -> (String, usize) {
    let mut out = String::new();
    let mut kept = 0;
    let mut i = 0;
    while i < lines.len() {
        if i > 0 {
            out.push('\n');
        }
        if keep[i] {
            out.push_str(&lines[i]);
            kept += 1;
            i += 1;
        } else {
            let mut j = i;
            while j < lines.len() && !keep[j] {
                j += 1;
            }
            out.push_str(&if j - i == 1 {
                "… 1 line …".to_owned()
            } else {
                format!("… {} lines …", j - i)
            });
            i = j;
        }
    }
    (out, kept)
}
