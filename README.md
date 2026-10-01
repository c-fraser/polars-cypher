# polars-cypher

[![test](https://github.com/c-fraser/polars-cypher/actions/workflows/test.yml/badge.svg)](https://github.com/c-fraser/polars-cypher/actions/workflows/test.yml)
[![Release](https://img.shields.io/github/v/release/c-fraser/polars-cypher?logo=github&sort=semver)](https://github.com/c-fraser/polars-cypher/releases)
[![Crates.io](https://img.shields.io/crates/v/polars-cypher.svg)](https://crates.io/crates/polars-cypher)
[![Documentation](https://docs.rs/polars-cypher/badge.svg)](https://docs.rs/polars-cypher)
[![Apache License 2.0](https://img.shields.io/badge/License-Apache2-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)

Run [Cypher](https://neo4j.com/docs/cypher-manual/current/) queries on property graphs stored as
[Parquet](https://parquet.apache.org/), using [Polars](https://docs.pola.rs/).

## Install

### Rust

```shell
cargo install polars-cypher-cli
```

> Requires [Rust](https://rust-lang.org/) *1.95+*.

Or add the library to a project:

```shell
cargo add polars-cypher --features neo4j
```

> The `neo4j` feature enables `Export`, which reads a graph from *Neo4j*
> using [neo4rs](https://github.com/neo4j-labs/neo4rs).

### Releases

Download a `polars-cypher` distribution for Linux (`x86_64` or `aarch64`) from a
[release](https://github.com/c-fraser/polars-cypher/releases).

## Usage

### CLI

```shell
polars-cypher export <bolt-uri> <graph-dir/> [--user neo4j] [--password ...] [--database ...]
polars-cypher schema <graph-dir/>
polars-cypher query  <graph-dir/> "<cypher>" [--param name=value]... [--explain]
```

### Library

```rust
use polars_cypher::Export;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let graph = Export::new("bolt://localhost:7687")
        .password("password")
        .write("graph/")?;
    let lf = graph
        .query("MATCH (p:Person) WHERE p.born < $year RETURN p.name")
        .parameter("year", 1970)
        .lazy()?;
    let df = lf.collect()?;
    Ok(())
}
```

### Cypher

`polars-cypher` implements a read-only subset of *Cypher*:

- [x] `MATCH` with one or more comma-separated patterns
- [x] Node labels (zero, one, or multiple) and inline property maps
- [x] Relationship direction, zero/one/alternative types (`:A|B`)
- [x] `WHERE`: comparisons, arithmetic, `AND`/`OR`/`NOT`/`XOR`
- [x] `IS [NOT] NULL`, `IN [...]`
- [x] `STARTS WITH` / `ENDS WITH` / `CONTAINS`
- [x] Parameters (`$x`)
- [x] `RETURN` aliases and `DISTINCT`, with unaliased columns named after their text
- [x] Aggregations: `count(*)`, `count`, `sum`, `avg`, `min`, `max`, `collect`, with implicit
  grouping, and `count`/`collect` of a whole node or relationship
- [x] `ORDER BY`, including expressions that aren't returned (except with `DISTINCT` or
  aggregations)
- [x] `SKIP` and `LIMIT`, as non-negative integer literals or parameters
- [x] Returning a whole entity (`RETURN p`) as a struct of `_id`, `_labels` (a list) or `_type`,
  and properties
- [x] Functions: `id`, `labels`, `type`, `coalesce`, `toUpper`, `toLower`, `size`, `abs`
- [x] Missing properties evaluate to `null`
- [ ] Variable-length relationships (`*1..3`)
- [ ] `OPTIONAL MATCH`, `WITH`, `UNWIND`, and `CASE`
- [ ] `IN` with a list parameter, rather than a list literal
- [ ] Comparing whole entities (`WHERE a <> b`)
- [ ] Aggregations nested in expressions (`count(*) + 1`)

## Demo

Refer to the *Docker Compose* [demo](demo/run.sh).

## License

    Copyright 2026 c-fraser

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        https://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
