// Rust guideline compliant 2026-10-14
//! Compatibility checks for the embedded trim-1 model and policy.

use trim::{FEATURE_NAMES, Forest, Options, POINTS, Policy, compress, line_features, score};

#[test]
fn embedded_model_parses_and_feature_contract_matches() {
    let bytes = include_bytes!("../model.bin");
    let forest = Forest::parse(bytes).expect("embedded model should parse");
    assert_eq!(FEATURE_NAMES.len(), 67);
    assert!(
        forest
            .raw(&vec![0.0; FEATURE_NAMES.len()])
            .unwrap()
            .is_finite()
    );
}

#[test]
fn model_raw_predictions_match_go_parity_fixture_bit_for_bit() {
    let bytes = include_bytes!("../testdata/parity.bin");
    let forest = Forest::parse(include_bytes!("../model.bin")).unwrap();
    let n = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
    let width = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
    assert_eq!(width, FEATURE_NAMES.len());
    assert_eq!(bytes.len(), 8 + n * width * 4 + n * 8);
    let rows_start = 8;
    let scores_start = rows_start + n * width * 4;
    for row_index in 0..n {
        let mut row = Vec::with_capacity(width);
        for feature in 0..width {
            let offset = rows_start + (row_index * width + feature) * 4;
            row.push(f32::from_bits(u32::from_le_bytes(
                bytes[offset..offset + 4].try_into().unwrap(),
            )));
        }
        let score_offset = scores_start + row_index * 8;
        let expected =
            u64::from_le_bytes(bytes[score_offset..score_offset + 8].try_into().unwrap());
        assert_eq!(
            forest.raw(&row).unwrap().to_bits(),
            expected,
            "fixture row {row_index}"
        );
    }
}

#[test]
fn feature_rows_are_repeatable_and_query_cap_is_utf8_safe() {
    let context = "ledger-db timeout\ncheckout payment";
    let query_tail = format!("{}why payment failed", "é".repeat(5000));
    let (_, a) = line_features(context, &format!("old words {query_tail}"));
    let (_, b) = line_features(context, &format!("different old words {query_tail}"));
    assert_eq!(a, b);
}

#[test]
fn score_and_compression_fail_closed_for_short_context() {
    let (_, scores) = score("small context\n", "query").unwrap();
    assert_eq!(scores.len(), 2);
    let result = compress("small context\n", "query", Options::default());
    assert!(result.unchanged);
    assert_eq!(result.reason, "context below the minimum size");
}

#[test]
fn final_newline_is_preserved_and_empty_split_piece_is_not_counted() {
    let lines = vec![
        "a".to_owned(),
        "b".to_owned(),
        "c".to_owned(),
        "d".to_owned(),
        String::new(),
    ];
    let policy = Policy {
        floor: 1.0,
        top_k: 1,
        context: 0,
        headers: 0,
        gap: 0,
        min_chars: 0,
        min_cut: 0.0,
    };
    let (out, kept) = policy.apply(&lines, &[1.0, 0.0, 0.0, 0.0, 0.0]).unwrap();
    assert_eq!(out, "a\n… 3 lines …\n");
    assert_eq!(kept, 1);
}

#[test]
fn policy_gap_and_input_lengths_are_respected() {
    let lines = ["first", "drop one", "drop two", "last"].map(str::to_owned);
    let policy = Policy {
        floor: 1.0,
        top_k: 1,
        context: 0,
        headers: 0,
        gap: 0,
        min_chars: 0,
        min_cut: 0.0,
    };
    let (output, kept) = policy.apply(&lines, &[1.0, 0.0, 0.0, 1.0]).unwrap();
    assert_eq!(output, "first\n… 2 lines …\nlast");
    assert_eq!(kept, 2);
    assert!(policy.apply(&lines, &[1.0]).is_err());
}

#[test]
fn public_forest_scoring_rejects_rows_with_wrong_width() {
    let forest = Forest::parse(include_bytes!("../model.bin")).unwrap();
    assert!(forest.raw(&[]).is_err());
    assert!(forest.probability(&[]).is_err());
}

#[test]
fn singular_elision_marker_is_refused() {
    let policy = Policy {
        floor: 0.0,
        top_k: 0,
        context: 0,
        headers: 0,
        gap: 0,
        min_chars: 0,
        min_cut: 0.0,
    };
    let result = compress(
        "first\n… 1 line …\nlast",
        "first",
        Options {
            point: "",
            policy: Some(policy),
        },
    );
    assert!(result.unchanged);
    assert_eq!(
        result.reason,
        "context has a line that reads as an elision marker"
    );
}

#[test]
fn named_points_are_stable() {
    assert_eq!(POINTS.len(), 3);
    assert_eq!(POINTS[1].0, "balanced");
}

#[test]
fn compression_matches_go_golden_fixture() {
    let mut context = String::from("$ kubectl logs deploy/checkout --since=1h\n");
    for i in 0..400 {
        context.push_str(&format!(
            "2026-10-11T03:{:02}:{:02}Z INFO checkout request {} served in {}ms\n",
            i / 60 % 60,
            i % 60,
            1000 + i,
            10 + i % 17
        ));
        if i == 271 {
            context.push_str(
                "2026-10-11T03:04:31Z ERROR payment client: ledger-db timeout after 30s (order 1271)\n",
            );
            context.push_str("    at LedgerClient.charge (ledger/client.go:88)\n");
        }
    }
    context.push_str("exit status 1\n");
    let result = compress(
        &context,
        "Why did the checkout payment fail? Find the ledger error.",
        Options::default(),
    );
    assert_eq!(
        result.output.as_bytes(),
        include_bytes!("../testdata/golden.txt")
    );
}
