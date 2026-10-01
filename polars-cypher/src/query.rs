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

//! Building and running *Cypher* queries for a [`Graph`].

use std::collections::HashMap;

use polars::prelude::LazyFrame;

use crate::Error;
use crate::layout::Graph;
use crate::plan;

impl Graph {
    /// Start building a *Cypher* query for this graph.
    pub fn query(&self, cypher: impl Into<String>) -> QueryBuilder<'_> {
        QueryBuilder {
            graph: self,
            cypher: cypher.into(),
            parameters: HashMap::new(),
        }
    }
}

/// A *Cypher* query and its bound parameters, built with [`Graph::query`].
pub struct QueryBuilder<'g> {
    graph: &'g Graph,
    cypher: String,
    parameters: HashMap<String, BoundParameter>,
}

impl<'g> QueryBuilder<'g> {
    /// Bind the value of the `$name` parameter, replacing any earlier value.
    pub fn parameter(mut self, name: impl Into<String>, value: impl Into<BoundParameter>) -> Self {
        self.parameters.insert(name.into(), value.into());
        self
    }

    /// Plan the query as a *Polars* [`LazyFrame`], to collect or compose further.
    ///
    /// Fails if the query is invalid, unsupported, or references an unbound parameter, with a
    /// message pointing at the offending text.
    pub fn lazy(self) -> Result<LazyFrame, Error> {
        plan::plan_query(self.graph, &self.cypher, &self.parameters).map(|(_, lf)| lf)
    }

    /// Plan the query without running it, and describe both its logical plan and the optimized
    /// *Polars* plan it lowers to.
    ///
    /// ```rust,no_run
    /// use polars_cypher::Graph;
    ///
    /// # fn main() -> Result<(), polars_cypher::Error> {
    /// let graph = Graph::open("graph/")?;
    /// let plan = graph.query("MATCH (p:Person) RETURN p.name").explain()?;
    /// println!("{plan}");
    /// # Ok(())
    /// # }
    /// ```
    pub fn explain(self) -> Result<String, Error> {
        let (logical, lf) = plan::plan_query(self.graph, &self.cypher, &self.parameters)?;
        let optimized = lf
            .explain(true)
            .map_err(|e| Error::wrap("failed to optimize the query", e))?;
        Ok(format!(
            "logical plan:\n{logical:#?}\n\npolars optimized plan:\n{optimized}"
        ))
    }
}

/// A query parameter value, bound with [`QueryBuilder::parameter`].
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum BoundParameter {
    /// A *Cypher* `INTEGER`.
    Int(i64),
    /// A *Cypher* `FLOAT`.
    Float(f64),
    /// A *Cypher* `STRING`.
    Str(String),
    /// A *Cypher* `BOOLEAN`.
    Bool(bool),
    /// `null`.
    Null,
}

/// Implement `From` for types that convert losslessly into a [`BoundParameter`] variant.
macro_rules! from_lossless {
    ($variant:ident($target:ty): $($source:ty),+) => {
        $(
            impl From<$source> for BoundParameter {
                fn from(v: $source) -> Self {
                    BoundParameter::$variant(<$target>::from(v))
                }
            }
        )+
    };
}

from_lossless!(Int(i64): i8, i16, i32, i64, u8, u16, u32);
from_lossless!(Float(f64): f32, f64);

impl From<&str> for BoundParameter {
    fn from(v: &str) -> Self {
        BoundParameter::Str(v.to_string())
    }
}

impl From<String> for BoundParameter {
    fn from(v: String) -> Self {
        BoundParameter::Str(v)
    }
}

impl From<bool> for BoundParameter {
    fn from(v: bool) -> Self {
        BoundParameter::Bool(v)
    }
}

/// Binds `None` as [`BoundParameter::Null`], and `Some(v)` as `v` itself.
impl<T: Into<BoundParameter>> From<Option<T>> for BoundParameter {
    fn from(v: Option<T>) -> Self {
        v.map_or(BoundParameter::Null, Into::into)
    }
}
