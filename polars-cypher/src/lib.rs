// Copyright 2026 c-fraser
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! # polars-cypher
//!
//! Run [Cypher](https://neo4j.com/docs/cypher-manual/current/) queries on a property [`Graph`]
//! stored as [Parquet](https://parquet.apache.org/), using [Polars](https://docs.pola.rs/).
//!
//! A graph is written with a [`GraphWriter`], or exported from *Neo4j* with [`Export`] (behind
//! the `neo4j` feature). [`Graph::query`] then plans a query as a *Polars* `LazyFrame`, to
//! collect or compose further.
//!
//! ```rust,no_run
//! use polars_cypher::Graph;
//!
//! # fn main() -> Result<(), polars_cypher::Error> {
//! let graph = Graph::open("graph/")?;
//! let lf = graph
//!     .query("MATCH (p:Person) WHERE p.born < $year RETURN p.name")
//!     .parameter("year", 1970)
//!     .lazy()?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Cypher
//!
//! Queries are read-only, of the form `MATCH ... [WHERE ...] RETURN ...`, supporting:
//!
//! - One or more comma-separated patterns, with node labels, relationship directions and types
//!   (`:A|B`), and inline property maps
//! - Comparisons, arithmetic, `AND`/`OR`/`NOT`/`XOR`, `IS [NOT] NULL`, `IN [...]`, and
//!   `STARTS WITH`/`ENDS WITH`/`CONTAINS`
//! - Parameters (`$x`), bound with [`QueryBuilder::parameter`]
//! - `RETURN` with aliases, `DISTINCT`, `ORDER BY`, `SKIP`, and `LIMIT`
//! - Aggregations (`count`, `sum`, `avg`, `min`, `max`, `collect`) with implicit grouping
//! - Functions `id`, `labels`, `type`, `coalesce`, `toUpper`, `toLower`, `size`, and `abs`
//!
//! Variable-length relationships, `OPTIONAL MATCH`, `WITH`, `UNWIND`, and `CASE` aren't
//! supported yet.
//!
#![cfg_attr(feature = "neo4j", doc = "[`Export`]: Export")]
#![cfg_attr(
    not(feature = "neo4j"),
    doc = "[`Export`]: https://docs.rs/polars-cypher/latest/polars_cypher/struct.Export.html"
)]

use std::error::Error as StdError;
use std::fmt;

mod cypher;
mod layout;
#[cfg(feature = "neo4j")]
mod neo4j;
mod plan;
mod query;

pub use layout::{
    Graph, GraphWriter, Manifest, NodeId, NodeTable, Properties, PropertySchema, PropertyType,
    RelTable, Value,
};
#[cfg(feature = "neo4j")]
pub use neo4j::Export;
pub use query::{BoundParameter, QueryBuilder};

/// A failure writing, opening, or querying a graph, with the underlying error as its
/// [`source`](StdError::source) when there is one.
#[derive(Debug)]
pub struct Error {
    message: String,
    source: Option<Box<dyn StdError + Send + Sync>>,
}

impl Error {
    pub(crate) fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            source: None,
        }
    }

    pub(crate) fn wrap(
        message: impl Into<String>,
        source: impl StdError + Send + Sync + 'static,
    ) -> Self {
        Self {
            message: message.into(),
            source: Some(Box::new(source)),
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl StdError for Error {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        self.source
            .as_ref()
            .map(|e| e.as_ref() as &(dyn StdError + 'static))
    }
}
