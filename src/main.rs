// Rust guideline compliant 2026-10-14
//! Command-line interface for trim-1 context compression.

use std::env;
use std::fs;
use std::io::{self, Read, Write};
use std::process::ExitCode;

use trim::{Options, POINTS, compress};

fn main() -> ExitCode {
    match run() {
        Ok(code) => code,
        Err((code, message)) => {
            let _ = writeln!(io::stderr().lock(), "{message}");
            code
        }
    }
}

fn run() -> std::result::Result<ExitCode, (ExitCode, String)> {
    let mut args = env::args().skip(1);
    let mut query = None;
    let mut point = "balanced".to_owned();
    let mut file = None;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--query" => query = args.next(),
            "--point" => point = args.next().ok_or_else(usage_error)?,
            "--help" | "-h" => {
                let _ = writeln!(
                    io::stderr().lock(),
                    "usage: trim --query QUERY [--point POINT] [FILE]   (reads stdin without FILE)"
                );
                return Ok(ExitCode::SUCCESS);
            }
            _ if arg.starts_with('-') => return Err(usage_error()),
            _ if file.is_none() => file = Some(arg),
            _ => return Err(usage_error()),
        }
    }
    let Some(query) = query.filter(|q| !q.is_empty()) else {
        return Err(usage_error());
    };
    if !POINTS.iter().any(|(name, _)| *name == point) {
        return Err((ExitCode::from(2), format!("trim: unknown point {point:?}")));
    }
    let context = match file {
        Some(path) => fs::read(&path).map_err(|e| (ExitCode::from(1), format!("trim: {e}")))?,
        None => {
            let mut bytes = Vec::new();
            io::stdin()
                .read_to_end(&mut bytes)
                .map_err(|e| (ExitCode::from(1), format!("trim: {e}")))?;
            bytes
        }
    };
    let context = String::from_utf8_lossy(&context);
    let result = compress(
        &context,
        &query,
        Options {
            point: &point,
            policy: None,
        },
    );
    io::stdout()
        .lock()
        .write_all(result.output.as_bytes())
        .map_err(|e| (ExitCode::from(1), format!("trim: {e}")))?;
    if result.unchanged {
        let _ = writeln!(io::stderr().lock(), "trim: unchanged ({})", result.reason);
    } else {
        let _ = writeln!(
            io::stderr().lock(),
            "trim: kept {}, removed {}, {} -> {} bytes",
            result.kept_lines,
            result.removed_lines,
            context.len(),
            result.output.len()
        );
    }
    Ok(ExitCode::SUCCESS)
}

fn usage_error() -> (ExitCode, String) {
    (ExitCode::from(2), "usage: trim --query QUERY [--point POINT] [FILE]   (reads stdin without FILE)\n  --query QUERY   the current ask (required)\n  --point POINT   operating point: conservative, balanced or aggressive".to_owned())
}
