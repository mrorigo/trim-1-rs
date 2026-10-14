// Rust guideline compliant 2026-10-14
//! BM25 scoring with the token rules used to train trim-1.

use std::collections::{BTreeMap, HashMap};

const K1: f64 = 1.5;
const B: f64 = 0.75;

/// Scores every document against a query using BM25.
pub(crate) fn score(query: &str, docs: &[&str]) -> Vec<f64> {
    let query_terms = term_freq(query);
    let mut out = vec![0.0; docs.len()];
    if query_terms.is_empty() || docs.is_empty() {
        return out;
    }
    let mut tfs = Vec::with_capacity(docs.len());
    let mut doc_len = Vec::with_capacity(docs.len());
    let mut df = HashMap::<String, usize>::new();
    let mut total_len = 0usize;
    for doc in docs {
        let tf = term_freq(doc);
        let len = tf.values().sum::<usize>();
        total_len += len;
        for term in tf.keys() {
            *df.entry(term.clone()).or_default() += 1;
        }
        tfs.push(tf);
        doc_len.push(len);
    }
    let avgdl = if total_len > 0 {
        total_len as f64 / docs.len() as f64
    } else {
        1.0
    };
    let terms = query_terms
        .keys()
        .collect::<std::collections::BTreeSet<_>>();
    for i in 0..docs.len() {
        for term in &terms {
            let tf = *tfs[i].get(*term).unwrap_or(&0) as f64;
            if tf == 0.0 {
                continue;
            }
            let document_frequency = df[*term] as f64;
            let idf = (1.0
                + (docs.len() as f64 - document_frequency + 0.5) / (document_frequency + 0.5))
                .ln();
            let denominator = tf + K1 * (1.0 - B + B * doc_len[i] as f64 / avgdl);
            out[i] += idf * (tf * (K1 + 1.0)) / denominator;
        }
    }
    out
}

fn term_freq(text: &str) -> BTreeMap<String, usize> {
    let mut freq = BTreeMap::new();
    let mut token = String::new();
    let flush = |token: &mut String, freq: &mut BTreeMap<String, usize>| {
        if token.len() >= 2 && !is_stopword(token) {
            *freq.entry(std::mem::take(token)).or_default() += 1;
        } else {
            token.clear();
        }
    };
    for ch in text.chars().flat_map(char::to_lowercase) {
        if ch.is_alphanumeric() || ch == '_' {
            token.push(ch);
        } else {
            flush(&mut token, &mut freq);
        }
    }
    flush(&mut token, &mut freq);
    freq
}

pub(crate) fn is_stopword(s: &str) -> bool {
    matches!(
        s,
        "a" | "an"
            | "and"
            | "are"
            | "as"
            | "at"
            | "be"
            | "by"
            | "for"
            | "from"
            | "how"
            | "in"
            | "is"
            | "it"
            | "of"
            | "on"
            | "or"
            | "that"
            | "the"
            | "this"
            | "to"
            | "was"
            | "what"
            | "when"
            | "where"
            | "which"
            | "who"
            | "why"
            | "with"
    )
}
