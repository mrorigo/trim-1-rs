# trim-1

trim-1 keeps the lines of a context that are relevant to a query. It returns
kept lines in their original order and inserts an elision marker where lines
were removed. Kept lines are not rewritten.

The crate embeds a gradient-boosted tree model. It runs locally with no
network access, GPU, or third-party Rust dependencies. Inference is
deterministic.

## Build and install

Build the command-line program with Rust 2024 support:

```sh
cargo build --release
```

Install the `trim` executable from this checkout with:

```sh
cargo install --path .
```

## Use the command line

Pass a query and a file. The command writes compressed context to standard
output and a one-line summary to standard error.

```sh
trim --query "why does the build fail" build.log > build.trimmed.log
```

Read from standard input when you omit the file:

```sh
cat build.log | trim --query "why does the build fail" --point balanced
```

The `--point` option accepts `conservative`, `balanced`, or `aggressive`. It
defaults to `balanced`. More aggressive points keep fewer lines and may remove
useful context.

The command exits with status 2 when the query is missing, arguments are
invalid, or the point is unknown. It exits with status 1 when it cannot read
input or write output.

## Use the Rust library

The library crate is named `trim`:

```rust
use trim::{compress, Options};

let context = "error: ledger-db timeout\nrequest completed\n";
let result = compress(
    context,
    "Why did the payment fail?",
    Options {
        point: "balanced",
        policy: None,
    },
);

print!("{}", result.output);
```

An empty `Options.point` selects `balanced`. Set `Options.policy` to use a
custom `Policy` instead of a named point. `Result` reports the output, line
counts, selected point and policy, and whether the input was returned
unchanged.

Use `score` to get the input lines and their keep probabilities:

```rust
fn main() -> Result<(), String> {
    let context = "error: ledger-db timeout\nrequest completed\n";
    let (lines, probabilities) = trim::score(context, "Why did the payment fail?")?;
    println!("{} lines scored", lines.len());
    println!("highest keep probability: {}", probabilities.iter().copied().fold(0.0, f64::max));
    Ok(())
}
```

`score` returns a `Result` with an error string if the embedded model cannot be
loaded. `line_features` exposes the model feature rows, and `model_info` reports
the embedded model size and dimensions.

## When compression returns the input unchanged

Compression fails closed and reports a reason in `Result.reason` when:

- The query contains only whitespace.
- The context is shorter than the selected policy's minimum size (2,000 bytes
  for the named points).
- The context exceeds 8 MiB or 200,000 lines.
- The context contains a line that looks like an elision marker.
- The selected lines remove less than 10% of the context bytes.
- The named operating point is unknown or the model cannot load.

The command still writes the unchanged context to standard output. It reports
the reason on standard error.

The library accepts Rust `&str` input. The command reads bytes and decodes them
as UTF-8, replacing invalid byte sequences with the Unicode replacement
character.

## Compatibility checks

The tests compare raw model predictions with the upstream Go parity fixture,
bit for bit. They also compare compressed output with the upstream golden
fixture. Run the checks with:

```sh
cargo fmt --check
cargo clippy -- -D warnings
cargo test
```

The model and compatibility fixtures come from the upstream Go project at
[`caveman-ai/trim-1`](https://github.com/caveman-ai/trim-1).
