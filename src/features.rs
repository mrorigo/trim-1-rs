// Rust guideline compliant 2026-10-14
//! Deterministic line features matching the embedded trim-1 model.

use std::collections::{HashMap, HashSet};

use crate::bm25::{self, is_stopword};

pub const FEATURE_NAMES: &[&str] = &[
    "bm25",
    "bm25_norm",
    "bm25_rank",
    "bm25_win_sum",
    "cover_win_sum",
    "bm25_tail_norm",
    "cover",
    "cover_idf",
    "cover_n",
    "cover_maxidf",
    "cover_rank",
    "cover_tail",
    "compound_hits",
    "compound_maxlen",
    "bigram_hits",
    "trigram_hits",
    "idf_mean",
    "idf_max",
    "uniq_frac",
    "k_error",
    "k_test",
    "k_path",
    "k_pathline",
    "k_heading",
    "k_diff",
    "k_section",
    "k_signature",
    "k_jsonkey",
    "k_separator",
    "k_frame",
    "k_kv",
    "k_bullet",
    "dup",
    "tmpl",
    "tmpl_first",
    "pos_rel",
    "pos_head",
    "pos_tail",
    "n_lines",
    "ctx_chars",
    "len_chars",
    "len_words",
    "alnum_frac",
    "upper_frac",
    "digit_frac",
    "indent",
    "indent_delta",
    "block_start",
    "header_dist",
    "header_bm25",
    "header_cover",
    "nb_bm25_max",
    "nb_bm25_mean",
    "nb_cover_max",
    "nb_compound_max",
    "prev_bm25",
    "next_bm25",
    "prev_blank",
    "next_blank",
    "q_words",
    "q_terms",
    "q_found",
    "ctx_bm25_max",
    "para_cover_max",
    "sec_cover",
    "prf_cover",
    "prf_compound",
];
pub(crate) const N_FEATURES: usize = 67;
const NEIGHBOUR_SPAN: usize = 3;
const COMPOUND_SPAN: usize = 5;
const MAX_COMPOUNDS: usize = 64;
const MAX_QUERY_BYTES: usize = 8 << 10;

// Feature indexes are part of the serialized model contract.
const F_BM25: usize = 0;
const F_BM25_NORM: usize = 1;
const F_BM25_RANK: usize = 2;
const F_BM25_WIN_SUM: usize = 3;
const F_COVER_WIN_SUM: usize = 4;
const F_BM25_TAIL_NORM: usize = 5;
const F_COVER: usize = 6;
const F_COVER_IDF: usize = 7;
const F_COVER_N: usize = 8;
const F_COVER_MAXIDF: usize = 9;
const F_COVER_RANK: usize = 10;
const F_COVER_TAIL: usize = 11;
const F_COMPOUND_HITS: usize = 12;
const F_COMPOUND_MAXLEN: usize = 13;
const F_BIGRAM_HITS: usize = 14;
const F_TRIGRAM_HITS: usize = 15;
const F_IDF_MEAN: usize = 16;
const F_IDF_MAX: usize = 17;
const F_UNIQ_FRAC: usize = 18;
const F_K_HEADING: usize = 23;
const F_DUP: usize = 32;
const F_TMPL: usize = 33;
const F_TMPL_FIRST: usize = 34;
const F_POS_REL: usize = 35;
const F_POS_HEAD: usize = 36;
const F_POS_TAIL: usize = 37;
const F_N_LINES: usize = 38;
const F_CTX_CHARS: usize = 39;
const F_LEN_CHARS: usize = 40;
const F_LEN_WORDS: usize = 41;
const F_ALNUM_FRAC: usize = 42;
const F_UPPER_FRAC: usize = 43;
const F_DIGIT_FRAC: usize = 44;
const F_INDENT: usize = 45;
const F_INDENT_DELTA: usize = 46;
const F_BLOCK_START: usize = 47;
const F_HEADER_DIST: usize = 48;
const F_HEADER_BM25: usize = 49;
const F_HEADER_COVER: usize = 50;
const F_NB_BM25_MAX: usize = 51;
const F_NB_BM25_MEAN: usize = 52;
const F_NB_COVER_MAX: usize = 53;
const F_NB_COMPOUND_MAX: usize = 54;
const F_PREV_BM25: usize = 55;
const F_NEXT_BM25: usize = 56;
const F_PREV_BLANK: usize = 57;
const F_NEXT_BLANK: usize = 58;
const F_Q_WORDS: usize = 59;
const F_Q_TERMS: usize = 60;
const F_Q_FOUND: usize = 61;
const F_CTX_BM25_MAX: usize = 62;
const F_PARA_COVER_MAX: usize = 63;
const F_SEC_COVER: usize = 64;
const F_PRF_COVER: usize = 65;
const F_PRF_COMPOUND: usize = 66;

/// Splits context into lines and computes one model feature row per line.
pub fn line_features(context: &str, query: &str) -> (Vec<String>, Vec<Vec<f32>>) {
    let query = cap_query(query);
    let lines = context.split('\n').map(str::to_owned).collect::<Vec<_>>();
    let n = lines.len();
    let docs = lines.iter().map(String::as_str).collect::<Vec<_>>();
    let bm = bm25::score(&query, &docs);
    let tail = query_tail(&query);
    let bm_tail = if tail != query.trim() {
        bm25::score(&tail, &docs)
    } else {
        bm.clone()
    };
    let bm_max = max_of(&bm);
    let tail_max = max_of(&bm_tail);

    let mut ids = HashMap::<String, usize>::new();
    let mut df = Vec::<i32>::new();
    let mut line_terms = Vec::<Vec<usize>>::with_capacity(n);
    let mut line_words = Vec::<Vec<String>>::with_capacity(n);
    for line in &lines {
        let (words, terms) = tokenize(line);
        line_words.push(words);
        let mut set = Vec::with_capacity(terms.len());
        for term in terms {
            let id = *ids.entry(term).or_insert_with(|| {
                let id = df.len();
                df.push(0);
                id
            });
            set.push(id);
        }
        set.sort_unstable();
        set.dedup();
        for &id in &set {
            df[id] += 1;
        }
        line_terms.push(set);
    }
    let idf = df
        .iter()
        .map(|&d| (1.0 + (n as f64 - f64::from(d) + 0.5) / (f64::from(d) + 0.5)).ln())
        .collect::<Vec<_>>();
    let (q_words, q_terms) = tokenize(&query);
    let q_terms = dedupe(q_terms);
    let mut in_query = vec![false; df.len()];
    let mut q_idf_sum = 0.0;
    let mut found = 0usize;
    for term in &q_terms {
        if let Some(&id) = ids.get(term) {
            in_query[id] = true;
            q_idf_sum += idf[id];
            found += 1;
        }
    }
    let (_, tail_terms) = tokenize(&tail);
    let mut in_tail = vec![false; df.len()];
    let mut tail_idf_sum = 0.0;
    for term in dedupe(tail_terms) {
        if let Some(&id) = ids.get(&term) {
            in_tail[id] = true;
            tail_idf_sum += idf[id];
        }
    }
    let (bigrams, trigrams) = ngrams(&q_words);
    let compounds = compound_terms(&query);

    let trimmed = lines
        .iter()
        .map(|line| line.trim().to_owned())
        .collect::<Vec<_>>();
    let blank = trimmed.iter().map(String::is_empty).collect::<Vec<_>>();
    let indent = lines.iter().map(|line| indent_of(line)).collect::<Vec<_>>();
    let mut dup_count = HashMap::<String, usize>::new();
    let mut tmpl_count = HashMap::<String, usize>::new();
    let tmpl_keys = trimmed.iter().map(|s| template(s)).collect::<Vec<_>>();
    for s in &trimmed {
        *dup_count.entry(s.clone()).or_default() += 1;
    }
    for s in &tmpl_keys {
        *tmpl_count.entry(s.clone()).or_default() += 1;
    }
    let (prev_nb, next_nb) = non_blank_neighbours(&blank);
    let headers = block_headers(&indent, &blank);
    let mut tmpl_seen = HashSet::new();
    let mut rows = vec![vec![0.0f32; N_FEATURES]; n];
    let mut cover = vec![0.0; n];
    let mut compound = vec![0.0; n];

    for i in 0..n {
        let lower = trimmed[i].to_lowercase();
        let set = &line_terms[i];
        let (mut cov, mut cov_idf, mut cov_max, mut idf_sum, mut idf_max, mut uniq, mut tail_idf) =
            (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
        for &id in set {
            if in_tail[id] {
                tail_idf += idf[id];
            }
            idf_sum += idf[id];
            idf_max = idf_max.max(idf[id]);
            if df[id] == 1 {
                uniq += 1.0;
            }
            if in_query[id] {
                cov += 1.0;
                cov_idf += idf[id];
                cov_max = cov_max.max(idf[id]);
            }
        }
        let mut hits = 0.0;
        let mut longest: f64 = 0.0;
        for c in &compounds {
            if lower.contains(c) {
                hits += 1.0;
                longest = longest.max(c.len() as f64);
            }
        }
        let mut bi = 0.0;
        let mut tri = 0.0;
        let words = &line_words[i];
        for j in 0..words.len().saturating_sub(1) {
            if bigrams.contains(&format!("{} {}", words[j], words[j + 1])) {
                bi += 1.0;
            }
            if j + 2 < words.len()
                && trigrams.contains(&format!("{} {} {}", words[j], words[j + 1], words[j + 2]))
            {
                tri += 1.0;
            }
        }
        cover[i] = ratio(cov_idf, q_idf_sum);
        compound[i] = hits;
        let r = &mut rows[i];
        r[F_BM25] = bm[i] as f32;
        r[F_BM25_NORM] = ratio(bm[i], bm_max) as f32;
        r[F_COVER_TAIL] = ratio(tail_idf, tail_idf_sum) as f32;
        r[F_BM25_TAIL_NORM] = ratio(bm_tail[i], tail_max) as f32;
        r[F_COVER] = ratio(cov, found as f64) as f32;
        r[F_COVER_IDF] = cover[i] as f32;
        r[F_COVER_N] = cov as f32;
        r[F_COVER_MAXIDF] = cov_max as f32;
        r[F_COMPOUND_HITS] = hits as f32;
        r[F_COMPOUND_MAXLEN] = longest.ln_1p() as f32;
        r[F_BIGRAM_HITS] = bi as f32;
        r[F_TRIGRAM_HITS] = tri as f32;
        if !set.is_empty() {
            r[F_IDF_MEAN] = (idf_sum / set.len() as f64) as f32;
            r[F_UNIQ_FRAC] = (uniq / set.len() as f64) as f32;
        }
        r[F_IDF_MAX] = idf_max as f32;
        let kinds = line_kinds(&trimmed[i], &lower);
        r[19..=31].copy_from_slice(&kinds);
        r[F_DUP] = ((dup_count[&trimmed[i]] - 1) as f64).ln_1p() as f32;
        r[F_TMPL] = ((tmpl_count[&tmpl_keys[i]] - 1) as f64).ln_1p() as f32;
        if tmpl_seen.insert(&tmpl_keys[i]) {
            r[F_TMPL_FIRST] = 1.0;
        }
        if n > 1 {
            r[F_POS_REL] = i as f32 / (n - 1) as f32;
        }
        r[F_POS_HEAD] = (i as f64).ln_1p() as f32;
        r[F_POS_TAIL] = ((n - 1 - i) as f64).ln_1p() as f32;
        r[F_N_LINES] = (n as f64).ln_1p() as f32;
        r[F_CTX_CHARS] = (context.len() as f64).ln_1p() as f32;
        r[F_LEN_CHARS] = (lines[i].len() as f64).ln_1p() as f32;
        r[F_LEN_WORDS] = (lines[i].split_whitespace().count() as f64).ln_1p() as f32;
        let (alnum, upper, digit, total) = char_mix(&trimmed[i]);
        if total > 0.0 {
            r[F_ALNUM_FRAC] = (alnum / total) as f32;
            r[F_UPPER_FRAC] = (upper / total) as f32;
            r[F_DIGIT_FRAC] = (digit / total) as f32;
        }
        r[F_INDENT] = (indent[i] as f64).ln_1p() as f32;
        if prev_nb[i] >= 0 && !blank[i] {
            r[F_INDENT_DELTA] = (indent[i] as isize - indent[prev_nb[i] as usize] as isize) as f32;
        }
        if next_nb[i] >= 0 && !blank[i] && indent[next_nb[i] as usize] > indent[i] {
            r[F_BLOCK_START] = 1.0;
        }
        r[F_Q_WORDS] = (q_words.len() as f64).ln_1p() as f32;
        r[F_Q_TERMS] = (q_terms.len() as f64).ln_1p() as f32;
        r[F_Q_FOUND] = ratio(found as f64, q_terms.len() as f64) as f32;
        r[F_CTX_BM25_MAX] = bm_max as f32;
    }

    let mut para_max = vec![0.0; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        let mut best: f64 = 0.0;
        while j < n && !blank[j] {
            best = best.max(cover[j]);
            j += 1;
        }
        for value in &mut para_max[i..j] {
            *value = best;
        }
        i = j.max(i + 1);
    }
    let mut by_cover = (0..n).collect::<Vec<_>>();
    by_cover.sort_by(|&a, &b| {
        if cover[a] > cover[b] {
            std::cmp::Ordering::Less
        } else if cover[a] < cover[b] {
            std::cmp::Ordering::Greater
        } else {
            std::cmp::Ordering::Equal
        }
    });
    let mut feedback = vec![false; df.len()];
    let mut fb_sum = 0.0;
    for &line in by_cover.iter().take(3) {
        if cover[line] == 0.0 {
            break;
        }
        for &id in &line_terms[line] {
            if !in_query[id] && !feedback[id] && df[id] <= 2.max((n / 50) as i32) {
                feedback[id] = true;
                fb_sum += idf[id];
            }
        }
    }
    let mut in_compounds = compounds.iter().cloned().collect::<HashSet<_>>();
    let mut hops = Vec::new();
    for &line in by_cover.iter().take(3) {
        if cover[line] == 0.0 {
            break;
        }
        for c in compound_terms(&lines[line]) {
            if !in_compounds.contains(&c) && hops.len() < MAX_COMPOUNDS {
                in_compounds.insert(c.clone());
                hops.push(c);
            }
        }
    }
    let mut heading: Option<usize> = None;
    for i in 0..n {
        let r = &mut rows[i];
        r[F_PARA_COVER_MAX] = para_max[i] as f32;
        if let Some(h) = heading {
            r[F_SEC_COVER] = cover[h] as f32;
        }
        if r[F_K_HEADING] == 1.0 {
            heading = Some(i);
        }
        let mut fb = 0.0;
        for &id in &line_terms[i] {
            if feedback[id] {
                fb += idf[id];
            }
        }
        r[F_PRF_COVER] = ratio(fb, fb_sum) as f32;
        if !hops.is_empty() {
            let lower = lines[i].to_lowercase();
            r[F_PRF_COMPOUND] = hops.iter().filter(|c| lower.contains(c.as_str())).count() as f32;
        }
    }
    let bm_rank = rank_frac(&bm);
    let cov_rank = rank_frac(&cover);
    for i in 0..n {
        let r = &mut rows[i];
        r[F_BM25_RANK] = bm_rank[i] as f32;
        r[F_COVER_RANK] = cov_rank[i] as f32;
        if headers[i] >= 0 {
            let h = headers[i] as usize;
            r[F_HEADER_DIST] = ((i - h) as f64).ln_1p() as f32;
            r[F_HEADER_BM25] = ratio(bm[h], bm_max) as f32;
            r[F_HEADER_COVER] = cover[h] as f32;
        }
        let (mut win_bm, mut win_cov) = (0.0, 0.0);
        for j in i.saturating_sub(2)..=(n - 1).min(i + 2) {
            win_bm += ratio(bm[j], bm_max);
            win_cov += cover[j];
        }
        r[F_BM25_WIN_SUM] = win_bm as f32;
        r[F_COVER_WIN_SUM] = win_cov as f32;
        let (mut nb_max, mut nb_sum, mut cov_nb, mut comp_nb, mut cnt) =
            (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0usize);
        for j in i.saturating_sub(NEIGHBOUR_SPAN)..=(n - 1).min(i + NEIGHBOUR_SPAN) {
            if j != i {
                let v = ratio(bm[j], bm_max);
                nb_max = nb_max.max(v);
                nb_sum += v;
                cov_nb = cov_nb.max(cover[j]);
                cnt += 1;
            }
        }
        for (j, &value) in compound
            .iter()
            .enumerate()
            .take((n - 1).min(i + COMPOUND_SPAN) + 1)
            .skip(i.saturating_sub(COMPOUND_SPAN))
        {
            if j != i {
                comp_nb = comp_nb.max(value);
            }
        }
        r[F_NB_BM25_MAX] = nb_max as f32;
        if cnt > 0 {
            r[F_NB_BM25_MEAN] = (nb_sum / cnt as f64) as f32;
        }
        r[F_NB_COVER_MAX] = cov_nb as f32;
        r[F_NB_COMPOUND_MAX] = comp_nb as f32;
        if i > 0 {
            r[F_PREV_BM25] = ratio(bm[i - 1], bm_max) as f32;
            if blank[i - 1] {
                r[F_PREV_BLANK] = 1.0;
            }
        }
        if i + 1 < n {
            r[F_NEXT_BM25] = ratio(bm[i + 1], bm_max) as f32;
            if blank[i + 1] {
                r[F_NEXT_BLANK] = 1.0;
            }
        }
    }
    (lines, rows)
}

fn cap_query(query: &str) -> String {
    if query.len() <= MAX_QUERY_BYTES {
        return query.to_owned();
    }
    let mut start = query.len() - MAX_QUERY_BYTES;
    while !query.is_char_boundary(start) {
        start += 1;
    }
    query[start..].to_owned()
}
fn ratio(a: f64, b: f64) -> f64 {
    if b <= 0.0 { 0.0 } else { a / b }
}
fn max_of(values: &[f64]) -> f64 {
    values.iter().copied().fold(0.0, f64::max)
}
fn rank_frac(values: &[f64]) -> Vec<f64> {
    let mut order = (0..values.len()).collect::<Vec<_>>();
    order.sort_by(|&a, &b| values[b].total_cmp(&values[a]));
    let mut out = vec![0.0; values.len()];
    let mut greater = 0;
    for (k, &i) in order.iter().enumerate() {
        if k > 0 && values[i] < values[order[k - 1]] {
            greater = k;
        }
        out[i] = greater as f64 / values.len().max(1) as f64;
    }
    out
}
fn query_tail(s: &str) -> String {
    let mut tail = s
        .trim_matches(go_space)
        .split('\n')
        .rev()
        .filter_map(|line| {
            let t = line.trim_matches(go_space);
            (!t.is_empty()).then_some(t)
        })
        .take(3)
        .collect::<Vec<_>>();
    tail.reverse();
    tail.join("\n")
}
fn tokenize(s: &str) -> (Vec<String>, Vec<String>) {
    let mut words = Vec::new();
    let mut terms = Vec::new();
    let mut start = None;
    for (i, ch) in s.char_indices() {
        if ch.is_alphanumeric() || ch == '_' {
            start.get_or_insert(i);
        } else if let Some(begin) = start.take() {
            add_token(&s[begin..i], &mut words, &mut terms);
        }
    }
    if let Some(begin) = start {
        add_token(&s[begin..], &mut words, &mut terms);
    }
    (words, terms)
}
fn add_token(raw: &str, words: &mut Vec<String>, terms: &mut Vec<String>) {
    let word = raw.to_lowercase();
    if word.len() < 2 || is_stopword(&word) {
        return;
    }
    words.push(word.clone());
    terms.push(word);
    if may_have_parts(raw) {
        let parts = split_ident(raw);
        if parts.len() > 1 {
            for part in parts {
                if part.len() >= 2 && !is_stopword(&part) {
                    terms.push(part);
                }
            }
        }
    }
}
fn may_have_parts(word: &str) -> bool {
    let mut digits = false;
    let mut others = false;
    for b in word.bytes() {
        match b {
            0x80..=0xff | b'_' | b'A'..=b'Z' => return true,
            b'0'..=b'9' => digits = true,
            _ => others = true,
        }
    }
    digits && others
}
fn split_ident(word: &str) -> Vec<String> {
    let chars = word.chars().collect::<Vec<_>>();
    let mut parts = Vec::new();
    let mut start = 0;
    let push = |end: usize, start: &mut usize, parts: &mut Vec<String>| {
        if end > *start {
            parts.push(chars[*start..end].iter().collect::<String>().to_lowercase());
        }
        *start = end;
    };
    for i in 0..chars.len() {
        let ch = chars[i];
        if ch == '_' {
            push(i, &mut start, &mut parts);
            start = i + 1;
            continue;
        }
        if i == start || i == 0 {
            continue;
        }
        let prev = chars[i - 1];
        if prev.is_lowercase() && ch.is_uppercase()
            || prev.is_uppercase()
                && ch.is_uppercase()
                && i + 1 < chars.len()
                && chars[i + 1].is_lowercase()
            || prev.is_numeric() != ch.is_numeric() && prev != '_'
        {
            push(i, &mut start, &mut parts);
        }
    }
    push(chars.len(), &mut start, &mut parts);
    parts
}
fn dedupe(values: Vec<String>) -> Vec<String> {
    let mut seen = HashSet::new();
    values
        .into_iter()
        .filter(|v| seen.insert(v.clone()))
        .collect()
}
fn ngrams(words: &[String]) -> (HashSet<String>, HashSet<String>) {
    let mut bi = HashSet::new();
    let mut tri = HashSet::new();
    for j in 0..words.len().saturating_sub(1) {
        bi.insert(format!("{} {}", words[j], words[j + 1]));
        if j + 2 < words.len() {
            tri.insert(format!("{} {} {}", words[j], words[j + 1], words[j + 2]));
        }
    }
    (bi, tri)
}
fn compound_terms(query: &str) -> Vec<String> {
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for field in query.split(|c: char| c.is_whitespace() || "\"'`()[]{}<>,;=|".contains(c)) {
        let field = field.trim_matches(|c| ".:!?*#".contains(c));
        if field.len() < 4 || seen.contains(field) {
            continue;
        }
        let structured = field.contains(['.', '_', '/', ':', '-', '\\'])
            || field
                .as_bytes()
                .windows(2)
                .any(|w| w[0].is_ascii_lowercase() && w[1].is_ascii_uppercase());
        if structured {
            seen.insert(field.to_owned());
            out.push(field.to_lowercase());
            if out.len() == MAX_COMPOUNDS {
                break;
            }
        }
    }
    out
}
fn template(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = String::with_capacity(s.len());
    let mut i = 0;
    while i < b.len() {
        let c = b[i];
        if c.is_ascii_digit() {
            while i < b.len() && (is_hex(b[i]) || b[i] == b'.' || b[i] == b':') {
                i += 1;
            }
            out.push('0');
            continue;
        }
        if is_hex(c) {
            let mut j = i;
            while j < b.len() && is_hex(b[j]) {
                j += 1;
            }
            if j - i >= 8 {
                out.push('0');
                i = j;
                continue;
            }
        }
        out.push((if c.is_ascii_uppercase() { c + 32 } else { c }) as char);
        i += 1;
    }
    out
}
fn is_hex(c: u8) -> bool {
    c.is_ascii_hexdigit()
}
pub(crate) fn indent_of(line: &str) -> usize {
    line.chars()
        .take_while(|&c| c == ' ' || c == '\t')
        .map(|c| if c == '\t' { 4 } else { 1 })
        .sum()
}
fn non_blank_neighbours(blank: &[bool]) -> (Vec<isize>, Vec<isize>) {
    let mut prev = vec![-1; blank.len()];
    let mut next = vec![-1; blank.len()];
    let mut last = -1;
    for i in 0..blank.len() {
        prev[i] = last;
        if !blank[i] {
            last = i as isize;
        }
    }
    last = -1;
    for i in (0..blank.len()).rev() {
        next[i] = last;
        if !blank[i] {
            last = i as isize;
        }
    }
    (prev, next)
}
pub(crate) fn block_headers(indent: &[usize], blank: &[bool]) -> Vec<isize> {
    let mut out = vec![-1; indent.len()];
    let mut stack: Vec<usize> = Vec::new();
    for i in 0..indent.len() {
        if blank[i] {
            continue;
        }
        while stack.last().is_some_and(|&j| indent[j] >= indent[i]) {
            stack.pop();
        }
        if let Some(&h) = stack.last() {
            out[i] = h as isize;
        }
        stack.push(i);
    }
    out
}
fn char_mix(s: &str) -> (f64, f64, f64, f64) {
    let (mut alnum, mut upper, mut digit, mut total) = (0.0, 0.0, 0.0, 0.0);
    for c in s.chars() {
        if c.is_whitespace() {
            continue;
        }
        total += 1.0;
        if c.is_alphanumeric() {
            alnum += 1.0;
        }
        if c.is_uppercase() {
            upper += 1.0;
        }
        if c.is_numeric() {
            digit += 1.0;
        }
    }
    (alnum, upper, digit, total)
}

fn go_space(ch: char) -> bool {
    matches!(ch, '\t'..='\r' | ' ' | '\u{0085}' | '\u{00a0}' | '\u{1680}' | '\u{2000}'..='\u{200a}' | '\u{2028}' | '\u{2029}' | '\u{202f}' | '\u{205f}' | '\u{3000}')
}

fn line_kinds(trimmed: &str, lower: &str) -> [f32; 13] {
    let mut k = [0.0; 13];
    if trimmed.is_empty() {
        return k;
    }
    let body = strip_line_number(lower);
    k[0] = contains_any(
        lower,
        &[
            "error",
            "exception",
            "traceback",
            "fatal",
            "panic",
            "fail",
            "assert",
            "warning",
            "denied",
            "refused",
            "not found",
            "cannot",
            "unable to",
            "invalid",
            "segmentation",
            "undefined",
            "unexpected",
            "errno",
            "critical",
        ],
    ) as u8 as f32;
    k[1] = (contains_any(
        body,
        &[
            " passed",
            " failed",
            " skipped",
            "xfail",
            "test session",
            "test summary",
        ],
    ) || has_prefix(
        body,
        &[
            "ok ", "fail", "pass", "=== run", "--- pass", "--- fail", "ran ",
        ],
    )) as u8 as f32;
    k[2] = has_path(trimmed) as u8 as f32;
    k[3] = has_path_line(trimmed) as u8 as f32;
    k[4] = (body.starts_with('#') && body.trim_start_matches('#').starts_with(' ')) as u8 as f32;
    k[5] = has_prefix(body, &["diff --git", "@@", "+++ ", "--- a/", "index "]) as u8 as f32;
    k[6] = is_section(body, trimmed) as u8 as f32;
    k[7] = has_prefix(
        body,
        &[
            "def ",
            "async def ",
            "class ",
            "func ",
            "function ",
            "fn ",
            "pub ",
            "struct ",
            "interface ",
            "type ",
            "impl ",
            "public ",
            "private ",
            "protected ",
            "static ",
            "export ",
            "module ",
            "package ",
            "import ",
            "from ",
            "#include",
            "const ",
            "var ",
            "let ",
            "@",
        ],
    ) as u8 as f32;
    k[8] = (body.starts_with('"') && body.contains("\":")) as u8 as f32;
    k[9] = (body.len() >= 3 && !body.chars().any(|c| c.is_alphanumeric())) as u8 as f32;
    k[10] = ((lower.contains("file \"") && lower.contains(", line "))
        || (body.starts_with("at ") && body.contains('('))
        || body.starts_with("goroutine ")) as u8 as f32;
    k[11] = is_key_value(body) as u8 as f32;
    k[12] = (has_prefix(body, &["- ", "* ", "• ", "+ "]) || numbered(body)) as u8 as f32;
    k
}
fn strip_line_number(s: &str) -> &str {
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() && bytes[i].is_ascii_digit() {
        i += 1;
    }
    if i == 0 || i == bytes.len() || !b" \t:|".contains(&bytes[i]) {
        s
    } else {
        s[i + 1..].trim()
    }
}
fn contains_any(s: &str, words: &[&str]) -> bool {
    words.iter().any(|w| s.contains(w))
}
fn has_prefix(s: &str, prefixes: &[&str]) -> bool {
    prefixes.iter().any(|p| s.starts_with(p))
}
fn has_path(s: &str) -> bool {
    if !s.contains(['/', '\\']) {
        return false;
    }
    s.split_whitespace().any(|field| {
        let f = field.trim_matches(['"', '\'', '`', '(', ')', ',', ';', ':', '[', ']']);
        f.rfind(['/', '\\']).is_some_and(|slash| {
            slash < f.len() - 1 && f[slash..].contains('.') && f.chars().any(char::is_alphabetic)
        })
    })
}
fn has_path_line(s: &str) -> bool {
    let b = s.as_bytes();
    for i in 1..b.len().saturating_sub(1) {
        if b[i] != b':' || !b[i + 1].is_ascii_digit() {
            continue;
        }
        let mut j = i;
        while j > 0 && i - j < 6 && b[j - 1].is_ascii_alphabetic() {
            j -= 1;
        }
        if j < i && j >= 2 && b[j - 1] == b'.' && b[j - 2] != b'.' && b[j - 2] != b' ' {
            return true;
        }
    }
    false
}
fn is_section(body: &str, trimmed: &str) -> bool {
    if body.starts_with('[') && body.ends_with(']')
        || body.starts_with("==>")
        || body.starts_with("file:")
    {
        return true;
    }
    if body.len() < 80 && body.ends_with(':') && !body.contains(['(', ')', '{', '}', '=']) {
        return true;
    }
    let letters = trimmed
        .chars()
        .filter(|c| c.is_alphabetic())
        .collect::<Vec<_>>();
    letters.len() >= 4 && letters.iter().all(|c| c.is_uppercase()) && trimmed.len() < 60
}
fn is_key_value(body: &str) -> bool {
    let b = body.as_bytes();
    let mut i = 0;
    while i < b.len()
        && (b[i] == b'_'
            || b[i] == b'.'
            || b[i] == b'-'
            || b[i].is_ascii_lowercase()
            || b[i].is_ascii_digit())
    {
        i += 1;
    }
    if i == 0 {
        return false;
    }
    while i < b.len() && b[i] == b' ' {
        i += 1;
    }
    i + 1 < b.len() && (b[i] == b':' || b[i] == b'=') && b[i + 1] != b'='
}
fn numbered(body: &str) -> bool {
    let b = body.as_bytes();
    let mut i = 0;
    while i < b.len() && b[i].is_ascii_digit() {
        i += 1;
    }
    i > 0 && i + 1 < b.len() && (b[i] == b'.' || b[i] == b')') && b[i + 1] == b' '
}
