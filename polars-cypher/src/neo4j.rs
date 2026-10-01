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

//! Exporting a [`Graph`] from Neo4j with [`Export`].

use std::collections::HashMap;
use std::path::Path;
use std::sync::LazyLock;

use chrono::{DateTime, FixedOffset, NaiveDate, NaiveDateTime};
use neo4rs::{BoltType, ConfigBuilder, Row, query};

use crate::Error;
use crate::layout::{Graph, GraphWriter, NodeId, Properties, Value};

/// Exports every node and relationship from a Neo4j database into a [`Graph`].
///
/// ```rust,no_run
/// use polars_cypher::Export;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
/// let graph = Export::new("bolt://localhost:7687")
///     .user("neo4j")
///     .password("password")
///     .write("graph/")?;
/// println!("exported {} node tables", graph.manifest().nodes.len());
/// # Ok(())
/// # }
/// ```
#[derive(Clone)]
pub struct Export {
    uri: String,
    user: String,
    password: String,
    database: Option<String>,
}

impl Export {
    /// Export from the Neo4j instance at the Bolt `uri`, as the `neo4j` user with an empty
    /// password, from the server's default database. Use [`Export::user`],
    /// [`Export::password`], and [`Export::database`] to change these.
    pub fn new(uri: impl Into<String>) -> Self {
        Self {
            uri: uri.into(),
            user: "neo4j".to_string(),
            password: String::new(),
            database: None,
        }
    }

    /// Authenticate as `user`.
    pub fn user(mut self, user: impl Into<String>) -> Self {
        self.user = user.into();
        self
    }

    /// Authenticate with the `password`.
    pub fn password(mut self, password: impl Into<String>) -> Self {
        self.password = password.into();
        self
    }

    /// Export from `database`.
    pub fn database(mut self, database: impl Into<String>) -> Self {
        self.database = Some(database.into());
        self
    }

    /// Export the graph into `out_dir`, and return it.
    ///
    /// This blocks the calling thread on its own async runtime, so from async code call it
    /// through [`spawn_blocking`](https://docs.rs/tokio/latest/tokio/task/fn.spawn_blocking.html).
    pub fn write(&self, out_dir: impl AsRef<Path>) -> Result<Graph, Error> {
        // the Bolt driver is async, but the public API is synchronous
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| Error::wrap("failed to start the Bolt client runtime", e))?;
        let mut writer = GraphWriter::create(out_dir)?;
        runtime.block_on(self.read(&mut writer))?;
        writer.write()
    }

    async fn read(&self, writer: &mut GraphWriter) -> Result<(), Error> {
        let mut config = ConfigBuilder::default()
            .uri(self.uri.as_str())
            .user(self.user.as_str())
            .password(self.password.as_str());
        if let Some(database) = &self.database {
            config = config.db(database.as_str());
        }
        let config = config
            .build()
            .map_err(|e| Error::wrap("invalid Neo4j connection settings", e))?;
        let bolt = neo4rs::Graph::connect(config)
            .await
            .map_err(|e| Error::wrap(format!("failed to connect to {}", self.uri), e))?;
        let mut txn = bolt
            .start_txn()
            .await
            .map_err(|e| Error::wrap("failed to start a transaction", e))?;

        let mut node_ids: HashMap<String, NodeId> = HashMap::new();
        let mut rows = txn
            .execute(query(&NODES_QUERY))
            .await
            .map_err(|e| Error::wrap("failed to read nodes", e))?;
        while let Some(row) = rows
            .next(txn.handle())
            .await
            .map_err(|e| Error::wrap("failed to read nodes", e))?
        {
            let labels: Vec<String> = get(&row, "labels")?;
            let id = writer.add_node(labels, properties(&row)?)?;
            node_ids.insert(get(&row, "id")?, id);
        }

        let mut rows = txn
            .execute(query(&RELS_QUERY))
            .await
            .map_err(|e| Error::wrap("failed to read relationships", e))?;
        while let Some(row) = rows
            .next(txn.handle())
            .await
            .map_err(|e| Error::wrap("failed to read relationships", e))?
        {
            let endpoint = |key| -> Result<NodeId, Error> {
                let id: String = get(&row, key)?;
                node_ids.get(&id).copied().ok_or_else(|| {
                    Error::new(format!(
                        "relationship endpoint {id} was not among the read nodes"
                    ))
                })
            };
            let rel_type: String = get(&row, "type")?;
            writer.add_relationship(
                endpoint("src")?,
                endpoint("dst")?,
                &rel_type,
                properties(&row)?,
            )?;
        }

        txn.commit()
            .await
            .map_err(|e| Error::wrap("failed to close the read transaction", e))?;
        Ok(())
    }
}

static NODES_QUERY: LazyLock<String> = LazyLock::new(|| {
    format!(
        "MATCH (n) RETURN elementId(n) AS id, labels(n) AS labels, keys(n) AS keys, {} AS values",
        property_values("n")
    )
});

fn get<'r, T: serde::Deserialize<'r>>(row: &'r Row, key: &str) -> Result<T, Error> {
    row.get(key)
        .map_err(|e| Error::wrap(format!("unexpected `{key}` in Neo4j result"), e))
}

fn properties(row: &Row) -> Result<Properties, Error> {
    let keys: Vec<String> = get(row, "keys")?;
    let values: Vec<BoltType> = get(row, "values")?;
    let mut properties = Properties::new();
    for (key, value) in keys.into_iter().zip(values) {
        let value = decode(value).map_err(|e| Error::wrap(format!("property `{key}`"), e))?;
        if let Some(value) = value {
            properties.insert(key, value);
        }
    }
    Ok(properties)
}

static RELS_QUERY: LazyLock<String> = LazyLock::new(|| {
    format!(
        "MATCH (a)-[r]->(b) RETURN elementId(a) AS src, elementId(b) AS dst, type(r) AS type, \
         keys(r) AS keys, {} AS values",
        property_values("r")
    )
});

/// The values of `entity`'s properties, in `keys(entity)` order.
fn property_values(entity: &str) -> String {
    let value = format!("{entity}[k]");
    let any_of = |template: &dyn Fn(&str) -> String| {
        STRINGIFIED_TYPES
            .iter()
            .map(|ty| format!("{value} IS :: {}", template(ty)))
            .collect::<Vec<_>>()
            .join(" OR ")
    };
    format!(
        "[k IN keys({entity}) | CASE WHEN {} THEN toString({value}) \
         WHEN {} THEN [x IN {value} | toString(x)] ELSE {value} END]",
        any_of(&|ty| ty.to_string()),
        any_of(&|ty| format!("LIST<{ty}>")),
    )
}

/// Decode a property value, or `None` for `null`.
fn decode(value: BoltType) -> Result<Option<Value>, Error> {
    let conversion = |e| Error::wrap("failed to convert temporal value", e);
    Ok(Some(match value {
        BoltType::Null(_) => return Ok(None),
        BoltType::Boolean(b) => Value::Bool(b.value),
        BoltType::Integer(i) => Value::Int(i.value),
        BoltType::Float(f) => Value::Float(f.value),
        BoltType::String(s) => Value::String(s.value),
        BoltType::Bytes(b) => Value::List(b.value.iter().map(|&b| Value::Int(b.into())).collect()),
        BoltType::Date(d) => {
            let date = NaiveDate::try_from(&d).map_err(conversion)?;
            let epoch = NaiveDate::default(); // 1970-01-01
            Value::Date((date - epoch).num_days() as i32)
        }
        BoltType::DateTime(dt) => Value::Datetime {
            micros: DateTime::<FixedOffset>::try_from(&dt)
                .map_err(conversion)?
                .timestamp_micros(),
            utc: true,
        },
        BoltType::DateTimeZoneId(dt) => Value::Datetime {
            micros: DateTime::<FixedOffset>::try_from(&dt)
                .map_err(conversion)?
                .timestamp_micros(),
            utc: true,
        },
        BoltType::LocalDateTime(dt) => Value::Datetime {
            micros: NaiveDateTime::try_from(&dt)
                .map_err(conversion)?
                .and_utc()
                .timestamp_micros(),
            utc: false,
        },
        BoltType::List(items) => Value::List(
            items
                .value
                .into_iter()
                .filter_map(|item| decode(item).transpose())
                .collect::<Result<_, _>>()?,
        ),
        other => return Err(Error::new(format!("unsupported property value {other:?}"))),
    }))
}

/// Neo4j types without a Polars equivalent. They are stringified server-side with `toString`, which
/// yields their canonical *Cypher* text.
const STRINGIFIED_TYPES: [&str; 4] = ["DURATION", "POINT", "ZONED TIME", "LOCAL TIME"];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stringify_type_without_a_polars_equivalent() {
        let values = property_values("n");
        assert!(values.starts_with("[k IN keys(n) | CASE WHEN n[k] IS :: DURATION OR"));
        assert!(values.contains("n[k] IS :: LIST<LOCAL TIME> THEN [x IN n[k] | toString(x)]"));
    }

    #[test]
    fn decode_scalar() {
        assert_eq!(decode(BoltType::from(42)).unwrap(), Some(Value::Int(42)));
        assert_eq!(decode(BoltType::Null(Default::default())).unwrap(), None);
        let date = NaiveDate::from_ymd_opt(1970, 1, 11).unwrap();
        assert_eq!(decode(BoltType::from(date)).unwrap(), Some(Value::Date(10)));
    }

    #[test]
    fn normalize_zoned_datetime_to_utc() {
        let dt = DateTime::parse_from_rfc3339("1970-01-01T01:00:00+01:00").unwrap();
        assert_eq!(
            decode(BoltType::from(dt)).unwrap(),
            Some(Value::Datetime {
                micros: 0,
                utc: true
            })
        );
    }
}
