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

use std::collections::BTreeSet;
use std::error::Error as StdError;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{Parser, Subcommand};
use polars_cypher::{BoundParameter, Export, Graph, PropertySchema};

#[derive(Parser)]
#[command(
    name = "polars-cypher",
    version,
    about = "Cypher graph query engine built on Polars."
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Export a graph from a running Neo4j instance into the polars-cypher graph layout.
    Export {
        /// Bolt URI of the Neo4j instance, e.g. bolt://localhost:7687.
        uri: String,
        /// Directory to write the exported graph into.
        out_dir: PathBuf,
        /// User to authenticate as.
        #[arg(long, env = "NEO4J_USER", default_value = "neo4j")]
        user: String,
        /// Password to authenticate with.
        #[arg(
            long,
            env = "NEO4J_PASSWORD",
            default_value = "",
            hide_env_values = true
        )]
        password: String,
        /// Database to export, instead of the server's default database.
        #[arg(long, env = "NEO4J_DATABASE")]
        database: Option<String>,
    },
    /// Print an exported graph's manifest.
    Schema {
        /// Path to an exported graph directory.
        graph_dir: PathBuf,
    },
    /// Run a Cypher query against an exported graph.
    Query {
        /// Path to an exported graph directory.
        graph_dir: PathBuf,
        /// The Cypher query text.
        cypher: String,
        /// A query parameter, e.g. --param year=1970.
        #[arg(long = "param", value_parser = parse_param)]
        param: Vec<(String, BoundParameter)>,
        /// Print the logical plan and the optimized Polars plan.
        #[arg(long)]
        explain: bool,
    },
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let result = match cli.command {
        Command::Export {
            uri,
            out_dir,
            user,
            password,
            database,
        } => {
            let mut export = Export::new(uri).user(user).password(password);
            if let Some(database) = database {
                export = export.database(database);
            }
            run_export(&export, &out_dir)
        }
        Command::Schema { graph_dir } => schema(&graph_dir),
        Command::Query {
            graph_dir,
            cypher,
            param,
            explain,
        } => query(&graph_dir, &cypher, param, explain),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn parse_param(s: &str) -> Result<(String, BoundParameter), String> {
    let (key, value) = s
        .split_once('=')
        .ok_or_else(|| format!("expected `key=value`, got `{s}`"))?;
    let quoted = ['\'', '"']
        .iter()
        .find_map(|&q| value.strip_prefix(q)?.strip_suffix(q));
    // only digits, signs, points, and exponents, so words like `nan` and `inf` stay strings
    let numeric = value
        .bytes()
        .all(|b| b.is_ascii_digit() || b"+-.eE".contains(&b));
    let value = if let Some(text) = quoted {
        BoundParameter::Str(text.to_string())
    } else if let Ok(n) = value.parse::<i64>() {
        BoundParameter::Int(n)
    } else if let Ok(n) = value.parse::<f64>()
        && numeric
    {
        BoundParameter::Float(n)
    } else if value.eq_ignore_ascii_case("true") {
        BoundParameter::Bool(true)
    } else if value.eq_ignore_ascii_case("false") {
        BoundParameter::Bool(false)
    } else if value.eq_ignore_ascii_case("null") {
        BoundParameter::Null
    } else {
        BoundParameter::Str(value.to_string())
    };
    Ok((key.to_string(), value))
}

fn run_export(export: &Export, out_dir: &Path) -> Result<(), Box<dyn StdError + Send + Sync>> {
    let graph = export.write(out_dir)?;
    let manifest = graph.manifest();
    println!(
        "exported {} node table(s), {} relationship table(s):",
        manifest.nodes.len(),
        manifest.relationships.len()
    );
    for t in &manifest.nodes {
        println!("  {}: {} rows", t.file, t.row_count);
    }
    for t in &manifest.relationships {
        println!("  {}: {} rows", t.file, t.row_count);
    }
    let coerced: BTreeSet<&str> = manifest
        .nodes
        .iter()
        .flat_map(|t| &t.schema)
        .chain(manifest.relationships.iter().flat_map(|t| &t.schema))
        .filter(|p| p.coerced)
        .map(|p| p.name.as_str())
        .collect();
    for name in coerced {
        eprintln!("warning: property `{name}` has mixed types and was widened");
    }
    Ok(())
}

fn schema(graph_dir: &Path) -> Result<(), Box<dyn StdError + Send + Sync>> {
    let graph = Graph::open(graph_dir)?;
    let manifest = graph.manifest();
    println!("layout version {}", manifest.version);
    println!("\nnode tables:");
    for t in &manifest.nodes {
        let label = t.label.as_deref().unwrap_or("(unlabeled)");
        println!("  {label} ({} rows)", t.row_count);
        print_properties(&t.schema);
    }
    println!("\nrelationship tables:");
    for t in &manifest.relationships {
        println!(
            "  {} ({} rows, {:?} -> {:?})",
            t.rel_type, t.row_count, t.src_labels, t.dst_labels
        );
        print_properties(&t.schema);
    }
    Ok(())
}

fn print_properties(schema: &[PropertySchema]) {
    for p in schema {
        let coerced = if p.coerced { " (coerced)" } else { "" };
        println!("    {}: {:?}{coerced}", p.name, p.dtype);
    }
}

fn query(
    graph_dir: &Path,
    cypher: &str,
    params: Vec<(String, BoundParameter)>,
    explain: bool,
) -> Result<(), Box<dyn StdError + Send + Sync>> {
    let graph = Graph::open(graph_dir)?;
    let query = params
        .into_iter()
        .fold(graph.query(cypher), |q, (name, value)| {
            q.parameter(name, value)
        });
    if explain {
        println!("{}", query.explain()?);
    } else {
        println!("{}", query.lazy()?.collect()?);
    }
    Ok(())
}
