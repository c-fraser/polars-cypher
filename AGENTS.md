# Agents

This file provides guidance to coding agents collaborating on this repository.

## Architecture

- `polars-cypher/src/cypher.rs`: parses *Cypher* text into an AST, with a span on every node
- `polars-cypher/src/plan.rs`: binds the AST, plans the joins, and lowers the plan to a *Polars* `LazyFrame`
- `polars-cypher/src/layout.rs`: the on-disk graph layout, written by `GraphWriter` and read by `Graph`
- `polars-cypher/src/neo4j.rs`: `Export`, behind the `neo4j` feature
- `polars-cypher-cli/`: the `polars-cypher` command-line interface

## Project Style

- Use English in code, examples, and comments
- Prefer conciseness, but not at the expense of clarity
- Implement features concisely, efficiently, and maintainably
- Favor concrete and consolidated over abstract and separated
- Comments explain *why*, not *what*
- Tests cover behavior and regressions, not trivial code
- Only use standard ASCII characters (0-127). Avoid emojis, curly quotes, smart quotes, or other UTF-8 symbols
- Don't use banner/divider comments like `# ---- Section ----`
- Begin `//` comments with a lowercase letter, unless they begin with an identifier or proper
  noun, and don't end them with punctuation. Doc comments (`///`, `//!`) are full sentences
- Use contractions in documentation and comments where natural (e.g. "don't" not "do not")
- Italicize project and product names in prose, e.g. *Cypher*, *Polars*, *Neo4j*, and *Parquet*
- Public APIs should have clear and succinct documentation that links to relevant URLs, structs, and methods, but never
  to private items

## Behavior and Compatibility

- *Neo4j* is the reference for query semantics. Cover new query features with a comparison test in
  `polars-cypher/tests/neo4j.rs`
- Errors about a query should carry the span of the offending text
- Keep the supported *Cypher* list in `README.md` and the `polars-cypher/src/lib.rs` crate docs in sync
- Bump `LAYOUT_VERSION` in `layout.rs` whenever the on-disk layout changes
- Mark public enums and structs that may grow `#[non_exhaustive]`
- Don't use Rust features newer than the `rust-version` in `Cargo.toml`

## Development Commands

- Format code and add license headers: `make check`
- Run all tests: `make test`
- Run unit tests quickly: `cargo test -p polars-cypher --lib`
- Check the minimum supported Rust version: `make msrv`
