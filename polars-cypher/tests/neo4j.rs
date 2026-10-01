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

//! Compares query results of `polars-cypher` and *Neo4j*.

use std::collections::HashMap as StdHashMap;
use std::error::Error;
use std::path::Path;
use std::sync::{LazyLock, OnceLock};
use std::time::Duration;

use neo4rs::{ConfigBuilder, Graph as BoltGraph, Query as BoltQuery, Row};
use polars::prelude::*;
use polars_cypher::{BoundParameter, Export, Graph};
use testcontainers::runners::AsyncRunner;
use testcontainers::{ContainerAsync, ImageExt, ReuseDirective};
use testcontainers_modules::neo4j::{Neo4j, Neo4jImage};
use tokio::runtime::Runtime;
use tokio::sync::{Mutex as AsyncMutex, MutexGuard};

#[tokio::test(flavor = "multi_thread")]
async fn where_return_property() -> Result<(), BoxError> {
    let guard = movies_env().await?;
    let env = guard.as_ref().unwrap();
    let cypher = "MATCH (p:Person) WHERE p.born < 1965 RETURN p.name AS value";
    let live = bolt_strings(&env.bolt, cypher).await?;
    let local = env.run_local(cypher)?;
    assert_eq!(sorted(&live), sorted(&local_strings(&local, "value")));
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn relationship_pattern() -> Result<(), BoxError> {
    let guard = movies_env().await?;
    let env = guard.as_ref().unwrap();
    let cypher =
        "MATCH (p:Person)-[:ACTED_IN]->(m:Movie {title: 'The Matrix'}) RETURN p.name AS value";
    let live = bolt_strings(&env.bolt, cypher).await?;
    let local = env.run_local(cypher)?;
    assert_eq!(sorted(&live), sorted(&local_strings(&local, "value")));
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn multi_hop_relationship_pattern() -> Result<(), BoxError> {
    let guard = movies_env().await?;
    let env = guard.as_ref().unwrap();
    let cypher = "MATCH (a:Person)-[:ACTED_IN]->(m:Movie)<-[:DIRECTED]-(d:Person) \
                  WHERE a.name = 'Keanu Reeves' RETURN DISTINCT d.name AS value";
    let live = bolt_strings(&env.bolt, cypher).await?;
    let local = env.run_local(cypher)?;
    assert_eq!(sorted(&live), sorted(&local_strings(&local, "value")));
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn count_star() -> Result<(), BoxError> {
    let guard = movies_env().await?;
    let env = guard.as_ref().unwrap();
    let live = bolt_ints(
        &env.bolt,
        "MATCH (p:Person)-[:ACTED_IN]->(:Movie) RETURN count(*) AS value",
    )
    .await?;
    let local = env.run_local("MATCH (p:Person)-[:ACTED_IN]->(:Movie) RETURN count(*) AS value")?;
    assert_eq!(live, local_ints(&local, "value"));
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn aggregate_min_max() -> Result<(), BoxError> {
    let guard = movies_env().await?;
    let env = guard.as_ref().unwrap();
    for func in ["min", "max"] {
        let cypher = format!("MATCH (p:Person) RETURN {func}(p.born) AS value");
        let live = bolt_ints(&env.bolt, &cypher).await?;
        let local = env.run_local(&cypher)?;
        assert_eq!(live, local_ints(&local, "value"), "{func}");
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn order_by_skip_limit() -> Result<(), BoxError> {
    let guard = movies_env().await?;
    let env = guard.as_ref().unwrap();
    let cypher = "MATCH (p:Person) RETURN p.name AS value ORDER BY value SKIP 2 LIMIT 5";
    let live = bolt_strings(&env.bolt, cypher).await?;
    let local = env.run_local(cypher)?;
    // ORDER BY is present, so row order must match exactly
    assert_eq!(live, local_strings(&local, "value"));
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn in_list_and_starts_with() -> Result<(), BoxError> {
    let guard = movies_env().await?;
    let env = guard.as_ref().unwrap();
    let cypher = "MATCH (p:Person) WHERE p.born IN [1961, 1964, 1967] \
                  AND p.name STARTS WITH 'K' RETURN p.name AS value";
    let live = bolt_strings(&env.bolt, cypher).await?;
    let local = env.run_local(cypher)?;
    assert_eq!(sorted(&live), sorted(&local_strings(&local, "value")));
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn is_null() -> Result<(), BoxError> {
    let guard = movies_env().await?;
    let env = guard.as_ref().unwrap();
    let cypher = "MATCH (p:Person) WHERE p.born IS NULL RETURN count(*) AS value";
    let live = bolt_ints(&env.bolt, cypher).await?;
    let local = env.run_local(cypher)?;
    assert_eq!(live, local_ints(&local, "value"));
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn parameter() -> Result<(), BoxError> {
    let guard = movies_env().await?;
    let env = guard.as_ref().unwrap();
    let cypher = "MATCH (p:Person) WHERE p.born < $year RETURN count(*) AS value";
    let query = BoltQuery::new(cypher.to_string()).param("year", 1965);
    let live = bolt_rows(&env.bolt, query)
        .await?
        .iter()
        .map(|row| row.get::<i64>("value"))
        .collect::<Result<Vec<_>, _>>()?;
    let mut params = StdHashMap::new();
    params.insert("year".to_string(), BoundParameter::Int(1965));
    let local = env.run_local_with_params(cypher, params)?;
    assert_eq!(live, local_ints(&local, "value"));
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn functions() -> Result<(), BoxError> {
    let guard = movies_env().await?;
    let env = guard.as_ref().unwrap();
    let cypher = "MATCH (p:Person) WHERE p.name = 'Keanu Reeves' RETURN toUpper(p.name) AS value";
    let live = bolt_strings(&env.bolt, cypher).await?;
    let local = env.run_local(cypher)?;
    assert_eq!(live, local_strings(&local, "value"));
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn multi_label() -> Result<(), BoxError> {
    let guard = movies_env().await?;
    let env = guard.as_ref().unwrap();
    let cypher = "MATCH (a:Actor:Artist) RETURN a.name AS value";
    let live = bolt_strings(&env.bolt, cypher).await?;
    let local = env.run_local(cypher)?;
    assert_eq!(sorted(&live), sorted(&local_strings(&local, "value")));
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn self_loop() -> Result<(), BoxError> {
    let guard = movies_env().await?;
    let env = guard.as_ref().unwrap();
    let cypher = "MATCH (a:Actor)-[:KNOWS]->(a) RETURN count(*) AS value";
    let live = bolt_ints(&env.bolt, cypher).await?;
    let local = env.run_local(cypher)?;
    assert_eq!(live, local_ints(&local, "value"));
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn parallel_relationships() -> Result<(), BoxError> {
    let guard = movies_env().await?;
    let env = guard.as_ref().unwrap();
    let cypher = "MATCH (:Actor {name: 'Ada'})-[:WORKED_WITH]->(:Actor {name: 'Bo'}) \
                  RETURN count(*) AS value";
    let live = bolt_ints(&env.bolt, cypher).await?;
    let local = env.run_local(cypher)?;
    assert_eq!(live, local_ints(&local, "value"));
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn relationship_uniqueness() -> Result<(), BoxError> {
    // two relationship variables of the same type between the same fixed endpoints must never
    // bind to the same physical relationship, which Cypher enforces without `WHERE r1 <> r2`
    let guard = movies_env().await?;
    let env = guard.as_ref().unwrap();
    let cypher = "MATCH (a:Actor {name: 'Ada'})-[r1:WORKED_WITH]->(b:Actor {name: 'Bo'}), \
                  (a)-[r2:WORKED_WITH]->(b) RETURN count(*) AS value";
    let live = bolt_ints(&env.bolt, cypher).await?;
    let local = env.run_local(cypher)?;
    assert_eq!(live, local_ints(&local, "value"));
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn mixed_int_and_float() -> Result<(), BoxError> {
    let guard = movies_env().await?;
    let env = guard.as_ref().unwrap();
    let cypher = "MATCH (a:Artist) WHERE a.score > 2.9 RETURN a.name AS value";
    let live = bolt_strings(&env.bolt, cypher).await?;
    let local = env.run_local(cypher)?;
    assert_eq!(sorted(&live), sorted(&local_strings(&local, "value")));
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn typed_list() -> Result<(), BoxError> {
    let guard = movies_env().await?;
    let env = guard.as_ref().unwrap();
    let cypher = "MATCH (r:Release) RETURN size(r.ratings) AS value";
    let live = bolt_ints(&env.bolt, cypher).await?;
    let local = env.run_local(cypher)?;
    assert_eq!(live, local_ints(&local, "value"));
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn anonymous_relationship_uniqueness() -> Result<(), BoxError> {
    let guard = movies_env().await?;
    let env = guard.as_ref().unwrap();
    let cypher = "MATCH (:Person)-[:ACTED_IN]->(:Movie)<-[:ACTED_IN]-(:Person) \
                  RETURN count(*) AS value";
    let live = bolt_ints(&env.bolt, cypher).await?;
    let local = env.run_local(cypher)?;
    assert_eq!(live, local_ints(&local, "value"));
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn disconnected_patterns() -> Result<(), BoxError> {
    let guard = movies_env().await?;
    let env = guard.as_ref().unwrap();
    let cypher = "MATCH (p:Person), (m:Movie) WHERE p.born > 1975 RETURN count(*) AS value";
    let live = bolt_ints(&env.bolt, cypher).await?;
    let local = env.run_local(cypher)?;
    assert_eq!(live, local_ints(&local, "value"));
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn missing_property() -> Result<(), BoxError> {
    let guard = movies_env().await?;
    let env = guard.as_ref().unwrap();
    let cypher = "MATCH (m:Movie) WHERE m.born IS NULL RETURN count(*) AS value";
    let live = bolt_ints(&env.bolt, cypher).await?;
    let local = env.run_local(cypher)?;
    assert_eq!(live, local_ints(&local, "value"));
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn integer_arithmetic() -> Result<(), BoxError> {
    let guard = movies_env().await?;
    let env = guard.as_ref().unwrap();
    let cypher = "MATCH (p:Person {name: 'Keanu Reeves'}) \
                  RETURN (-p.born) / 7 * 10 + (-p.born) % 7 AS value";
    let live = bolt_ints(&env.bolt, cypher).await?;
    let local = env.run_local(cypher)?;
    assert_eq!(live, local_ints(&local, "value"));
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn undirected_self_loop() -> Result<(), BoxError> {
    let guard = movies_env().await?;
    let env = guard.as_ref().unwrap();
    let cypher = "MATCH (a:Actor)-[:KNOWS]-(a) RETURN count(*) AS value";
    let live = bolt_ints(&env.bolt, cypher).await?;
    let local = env.run_local(cypher)?;
    assert_eq!(live, local_ints(&local, "value"));
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn unlabeled_node() -> Result<(), BoxError> {
    let guard = movies_env().await?;
    let env = guard.as_ref().unwrap();
    let cypher = "MATCH (:Actor)-[:KNOWS]-(n) RETURN n.name AS value";
    let live = bolt_strings(&env.bolt, cypher).await?;
    let local = env.run_local(cypher)?;
    assert_eq!(sorted(&live), sorted(&local_strings(&local, "value")));
    Ok(())
}

type BoxError = Box<dyn Error + Send + Sync>;

async fn movies_env() -> Result<MutexGuard<'static, Option<TestEnv>>, BoxError> {
    let mutex = MOVIES.get_or_init(|| AsyncMutex::new(None));
    let mut guard = mutex.lock().await;
    if guard.is_none() {
        *guard = Some(RUNTIME.spawn(TestEnv::new()).await??);
    }
    Ok(guard)
}

/// Run `query` on the *Neo4j* instance, on the runtime that owns the pool's connections.
async fn bolt_rows(bolt: &BoltGraph, query: BoltQuery) -> Result<Vec<Row>, BoxError> {
    let bolt = bolt.clone();
    let rows = RUNTIME.spawn(async move {
        let mut stream = bolt.execute(query).await?;
        let mut rows = Vec::new();
        while let Some(row) = stream.next().await? {
            rows.push(row);
        }
        Ok::<_, neo4rs::Error>(rows)
    });
    Ok(rows.await??)
}

async fn bolt_strings(bolt: &BoltGraph, cypher: &str) -> Result<Vec<String>, BoxError> {
    let rows = bolt_rows(bolt, BoltQuery::new(cypher.to_string())).await?;
    Ok(rows
        .iter()
        .map(|row| row.get::<String>("value"))
        .collect::<Result<_, _>>()?)
}

fn sorted<T: Ord + Clone>(v: &[T]) -> Vec<T> {
    let mut v = v.to_vec();
    v.sort();
    v
}

fn local_strings(df: &DataFrame, col: &str) -> Vec<String> {
    df.column(col)
        .unwrap()
        .str()
        .unwrap()
        .iter()
        .flatten()
        .map(str::to_string)
        .collect()
}

async fn bolt_ints(bolt: &BoltGraph, cypher: &str) -> Result<Vec<i64>, BoxError> {
    let rows = bolt_rows(bolt, BoltQuery::new(cypher.to_string())).await?;
    Ok(rows
        .iter()
        .map(|row| row.get::<i64>("value"))
        .collect::<Result<_, _>>()?)
}

fn local_ints(df: &DataFrame, col: &str) -> Vec<i64> {
    df.column(col)
        .unwrap()
        .i64()
        .unwrap()
        .iter()
        .flatten()
        .collect()
}

struct TestEnv {
    // kept alive for the container's lifetime, but never read directly
    #[allow(dead_code)]
    container: ContainerAsync<Neo4jImage>,
    bolt: BoltGraph,
    graph: Graph,
}

impl TestEnv {
    async fn new() -> Result<Self, BoxError> {
        let container = Neo4j::default()
            .with_version("2026-community")
            .with_container_name("polars-cypher-test-neo4j")
            .with_reuse(ReuseDirective::Always)
            .start()
            .await?;
        let port = container.get_host_port_ipv4(7687).await?;
        let uri = format!("bolt://127.0.0.1:{port}");

        let config = ConfigBuilder::default()
            .uri(&uri)
            .user("neo4j")
            .password("password")
            .build()?;
        // a reused container may still be booting after a restart, and holds the prior fixture
        let deadline = tokio::time::Instant::now() + Duration::from_secs(60);
        let bolt = loop {
            let ready = match BoltGraph::connect(config.clone()).await {
                Ok(bolt) => bolt
                    .run(BoltQuery::new("RETURN 1".to_string()))
                    .await
                    .map(|_| bolt),
                Err(e) => Err(e),
            };
            match ready {
                Ok(bolt) => break bolt,
                Err(e) if tokio::time::Instant::now() >= deadline => return Err(e.into()),
                Err(_) => tokio::time::sleep(Duration::from_millis(500)).await,
            }
        };
        bolt.run(BoltQuery::new("MATCH (n) DETACH DELETE n".to_string()))
            .await?;

        let script = std::fs::read_to_string(
            Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/movies.cypher"),
        )?;
        for statement in script.split(';') {
            let statement = statement.trim();
            if statement.is_empty() {
                continue;
            }
            bolt.run(BoltQuery::new(statement.to_string())).await?;
        }

        let graph_dir =
            std::env::temp_dir().join(format!("polars-cypher-neo4j-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&graph_dir);
        // `Export::write` blocks on its own runtime, so it can't run on this one's worker thread
        let export = Export::new(uri).password("password");
        let graph = tokio::task::spawn_blocking(move || export.write(graph_dir)).await??;

        Ok(Self {
            container,
            bolt,
            graph,
        })
    }

    /// Run `cypher` through `polars-cypher` and collect the result.
    fn run_local(&self, cypher: &str) -> Result<DataFrame, BoxError> {
        self.run_local_with_params(cypher, StdHashMap::new())
    }

    fn run_local_with_params(
        &self,
        cypher: &str,
        params: StdHashMap<String, BoundParameter>,
    ) -> Result<DataFrame, BoxError> {
        let mut builder = self.graph.query(cypher);
        for (name, value) in params {
            builder = builder.parameter(name, value);
        }
        let lf = builder.lazy()?;
        Ok(lf.collect()?)
    }
}

static MOVIES: OnceLock<AsyncMutex<Option<TestEnv>>> = OnceLock::new();

// each test has its own runtime, which shuts down when the test ends, so the shared container
// and connection pool live on this one instead
static RUNTIME: LazyLock<Runtime> = LazyLock::new(|| Runtime::new().unwrap());
