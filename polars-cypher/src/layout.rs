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

//! Writing and reading the on-disk [`Graph`] layout.

use std::collections::btree_map::Entry;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fmt;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

use chrono::{DateTime, NaiveDate, SecondsFormat};
use polars::prelude::*;
use serde::{Deserialize, Serialize};

use crate::Error;

/// A property [`Graph`] stored as a directory of *Parquet* tables, written by a [`GraphWriter`].
///
/// ```text
/// graph/
///   manifest.json                  # layout version, tables, files, schemas, row counts
///   nodes/<Label>.parquet          # _id: u64, then one column per property
///   nodes/_unlabeled.parquet       # nodes without labels, if any
///   relationships/<TYPE>.parquet   # _id, _src, _dst: u64, then properties
/// ```
#[derive(Debug, Clone)]
pub struct Graph {
    root: PathBuf,
    manifest: Manifest,
}

impl Graph {
    /// Open the graph directory at `path`.
    ///
    /// Fails if the graph was written with a different layout version, in which case it must be
    /// rewritten (using the current [`GraphWriter`]).
    pub fn open(path: impl AsRef<Path>) -> Result<Self, Error> {
        let root = path.as_ref().to_path_buf();
        let manifest = Manifest::load(&manifest_path(&root))?;
        if manifest.version != LAYOUT_VERSION {
            return Err(Error::new(format!(
                "{} has layout version {}, but only version {LAYOUT_VERSION} is supported",
                root.display(),
                manifest.version
            )));
        }
        Ok(Self::new(root, manifest))
    }

    /// Initialize a [`Graph`].
    pub(crate) fn new(root: PathBuf, manifest: Manifest) -> Self {
        Self { root, manifest }
    }

    /// The graph's tables and their schemas.
    pub fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    /// Scan the [`LazyFrame`] for nodes with the `label`.
    pub(crate) fn scan_node_table(&self, label: &str) -> Result<LazyFrame, Error> {
        let table = self.manifest.node_table(label);
        let table = table.ok_or_else(|| Error::new(format!("unknown label `{label}`")))?;
        self.scan(&table.file)
    }

    /// Scan the [`LazyFrame`] for relationships with the `type`.
    pub(crate) fn scan_rel_table(&self, rel_type: &str) -> Result<LazyFrame, Error> {
        let table = self.manifest.rel_table(rel_type);
        let table = table.ok_or_else(|| Error::new(format!("unknown type `{rel_type}`")))?;
        self.scan(&table.file)
    }

    /// Scan the [`LazyFrame`] for a table's `file`, relative to the graph directory.
    pub(crate) fn scan(&self, file: &str) -> Result<LazyFrame, Error> {
        scan_parquet(&self.root.join(file))
    }
}

/// Writes a [`Graph`] incrementally, so it needn't fit in memory.
///
/// Rows are staged on disk until [`GraphWriter::write`] converts them into *Parquet* tables. A
/// property's type is unified across the whole graph, so a property with mixed types is widened,
/// from integers to floats or otherwise to strings, and marked [`PropertySchema::coerced`].
///
/// The property names `_id`, `_src`, `_dst`, `_type`, and `_labels` are reserved for the columns
/// every table or query has.
///
/// ```rust,no_run
/// use polars_cypher::{GraphWriter, Value};
///
/// # fn main() -> Result<(), polars_cypher::Error> {
/// let mut writer = GraphWriter::create("graph/")?;
/// let name = |name: &str| [("name".to_string(), Value::String(name.to_string()))].into();
/// let ada = writer.add_node(["Person"], name("Ada"))?;
/// let cy = writer.add_node(["Person"], name("Cy"))?;
/// writer.add_relationship(ada, cy, "KNOWS", Default::default())?;
/// let graph = writer.write()?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct GraphWriter {
    root: PathBuf,
    next_id: u64,
    types: ObservedTypes,
    /// Staged node tables by label, with `None` for nodes without labels.
    nodes: BTreeMap<Option<String>, StagedTable>,
    relationships: BTreeMap<String, StagedTable>,
}

/// The id of a node added to a [`GraphWriter`], to add relationships between nodes with.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(u64);

impl GraphWriter {
    /// Start writing the graph directory at `path`, staging rows in `path/.staging`.
    pub fn create(path: impl AsRef<Path>) -> Result<Self, Error> {
        let root = path.as_ref().to_path_buf();
        // a previous writer that never finished may have left rows behind
        let _ = fs::remove_dir_all(staging_dir(&root));
        create_dir(&staging_dir(&root))?;
        Ok(Self {
            root,
            next_id: 0,
            types: ObservedTypes::new(),
            nodes: BTreeMap::new(),
            relationships: BTreeMap::new(),
        })
    }

    /// Add a node, with a row in the node table of each of its `labels`, or in the unlabeled
    /// node table if it has none.
    pub fn add_node<L: AsRef<str>>(
        &mut self,
        labels: impl IntoIterator<Item = L>,
        properties: Properties,
    ) -> Result<NodeId, Error> {
        check_reserved(&properties)?;
        let mut labels: Vec<Option<String>> = labels
            .into_iter()
            .map(|l| Some(l.as_ref().to_string()))
            .collect();
        if labels.is_empty() {
            labels.push(None);
        }
        let row = StagedRow {
            id: self.next_id(),
            endpoints: None,
            properties,
        };
        observe(&mut self.types, &row.properties);
        let dir = staging_dir(&self.root).join("nodes");
        for label in labels {
            stage(&mut self.nodes, &dir, label, false)?.append(&row)?;
        }
        Ok(NodeId(row.id))
    }

    /// Add a relationship of `rel_type` from `src` to `dst`.
    pub fn add_relationship(
        &mut self,
        src: NodeId,
        dst: NodeId,
        rel_type: &str,
        properties: Properties,
    ) -> Result<(), Error> {
        check_reserved(&properties)?;
        let row = StagedRow {
            id: self.next_id(),
            endpoints: Some((src.0, dst.0)),
            properties,
        };
        observe(&mut self.types, &row.properties);
        let dir = staging_dir(&self.root).join("relationships");
        let table = stage(&mut self.relationships, &dir, rel_type.to_string(), true)?;
        table.append(&row)
    }

    /// Write every table's *Parquet* file and the `manifest.json`, remove the staged rows, and
    /// return the [`Graph`].
    pub fn write(self) -> Result<Graph, Error> {
        let types = resolve(&self.types);
        create_dir(&self.root.join(NODES_DIR))?;
        create_dir(&self.root.join(RELATIONSHIPS_DIR))?;
        let mut files = FileNames::default();

        // `None` sorts first, so the unlabeled table claims its file name before any label
        let mut node_tables = Vec::with_capacity(self.nodes.len());
        for (label, table) in self.nodes {
            let name = label.as_deref().unwrap_or("_unlabeled");
            let file = files.assign(NODES_DIR, name);
            let (row_count, schema) = table.write(&self.root.join(&file), &types)?;
            node_tables.push(NodeTable {
                label,
                file,
                row_count,
                schema,
            });
        }

        let mut rel_tables = Vec::with_capacity(self.relationships.len());
        for (rel_type, table) in self.relationships {
            let file = files.assign(RELATIONSHIPS_DIR, &rel_type);
            let (row_count, schema) = table.write(&self.root.join(&file), &types)?;
            let endpoint_labels =
                |endpoint| endpoint_labels(&self.root, &file, endpoint, &node_tables);
            rel_tables.push(RelTable {
                src_labels: endpoint_labels(SRC_COL)?,
                dst_labels: endpoint_labels(DST_COL)?,
                rel_type,
                file,
                row_count,
                schema,
            });
        }

        let manifest = Manifest {
            version: LAYOUT_VERSION,
            nodes: node_tables,
            relationships: rel_tables,
        };
        manifest.save(&manifest_path(&self.root))?;
        let _ = fs::remove_dir_all(staging_dir(&self.root));
        Ok(Graph::new(self.root, manifest))
    }

    /// Node and relationship ids share one sequence, so they're unique across the whole graph.
    fn next_id(&mut self) -> u64 {
        self.next_id += 1;
        self.next_id - 1
    }
}

/// A node or relationship row, as staged in bincode.
#[derive(Debug, Serialize, Deserialize)]
struct StagedRow {
    id: u64,
    /// The source and target node ids of a relationship.
    endpoints: Option<(u64, u64)>,
    properties: Properties,
}

/// The staged bincode file of a table.
#[derive(Debug)]
struct StagedTable {
    path: PathBuf,
    file: BufWriter<File>,
    is_rel: bool,
    /// Every property name on any of the table's rows.
    names: BTreeSet<String>,
    row_count: u64,
}

/// The staged table `key` in `tables`, created in `dir`.
fn stage<'a, K: Ord>(
    tables: &'a mut BTreeMap<K, StagedTable>,
    dir: &Path,
    key: K,
    is_rel: bool,
) -> Result<&'a mut StagedTable, Error> {
    // numbered, since labels and types aren't necessarily valid file names
    let path = dir.join(format!("{}.bin", tables.len()));
    match tables.entry(key) {
        Entry::Occupied(entry) => Ok(entry.into_mut()),
        Entry::Vacant(entry) => {
            create_dir(dir)?;
            let file = File::create(&path)
                .map_err(|e| Error::wrap(format!("failed to create {}", path.display()), e))?;
            Ok(entry.insert(StagedTable {
                path,
                file: BufWriter::new(file),
                is_rel,
                names: BTreeSet::new(),
                row_count: 0,
            }))
        }
    }
}

/// Fail if `properties` has a name reserved for the columns every table or query has.
fn check_reserved(properties: &Properties) -> Result<(), Error> {
    let reserved = [ID_COL, SRC_COL, DST_COL, TYPE_COL, LABELS_COL];
    match properties
        .keys()
        .find(|name| reserved.contains(&name.as_str()))
    {
        Some(name) => Err(Error::new(format!("property name `{name}` is reserved"))),
        None => Ok(()),
    }
}

/// Assigns each table a unique file, named after its label or type.
#[derive(Default)]
struct FileNames(HashSet<String>);

impl FileNames {
    /// A file in `dir` for the table `name`, unique even on case-insensitive file systems.
    fn assign(&mut self, dir: &str, name: &str) -> String {
        let stem: String = name
            .chars()
            .map(|c| match c {
                'a'..='z' | 'A'..='Z' | '0'..='9' | '_' | '-' => c,
                _ => '_',
            })
            .collect();
        let mut file = format!("{dir}/{stem}.parquet");
        let mut n = 0;
        while !self.0.insert(file.to_lowercase()) {
            n += 1;
            file = format!("{dir}/{stem}-{n}.parquet");
        }
        file
    }
}

impl StagedTable {
    fn append(&mut self, row: &StagedRow) -> Result<(), Error> {
        for name in row.properties.keys() {
            if !self.names.contains(name) {
                self.names.insert(name.clone());
            }
        }
        bincode::serde::encode_into_std_write(row, &mut self.file, bincode::config::standard())
            .map_err(|e| {
                Error::wrap(
                    format!("failed to stage a row in {}", self.path.display()),
                    e,
                )
            })?;
        self.row_count += 1;
        Ok(())
    }

    /// Write the staged rows into the *Parquet* table at `path`, [`CHUNK_ROWS`] rows at a time.
    /// Returns the table's row count and schema.
    fn write(
        mut self,
        path: &Path,
        types: &PropertyTypes,
    ) -> Result<(u64, Vec<PropertySchema>), Error> {
        let failed = |e| Error::wrap(format!("failed to write {}", path.display()), e);
        self.file.flush().map_err(failed)?;
        let schema: Vec<PropertySchema> = self
            .names
            .iter()
            .map(|name| {
                let (dtype, coerced) = types[name];
                PropertySchema {
                    name: name.clone(),
                    dtype,
                    coerced,
                }
            })
            .collect();
        let id_cols: &[&str] = if self.is_rel {
            &[ID_COL, SRC_COL, DST_COL]
        } else {
            &[ID_COL]
        };
        let polars_schema: Schema = id_cols
            .iter()
            .map(|&name| Field::new(name.into(), DataType::UInt64))
            .chain(
                schema
                    .iter()
                    .map(|p| Field::new(p.name.as_str().into(), p.dtype.to_polars())),
            )
            .collect();

        let file = File::create(path).map_err(failed)?;
        let mut writer = ParquetWriter::new(file)
            .batched(&polars_schema)
            .map_err(|e| failed(e.into()))?;
        let mut staged = BufReader::new(File::open(&self.path).map_err(failed)?);
        let mut remaining = self.row_count;
        while remaining > 0 {
            let rows = (0..remaining.min(CHUNK_ROWS as u64))
                .map(|_| {
                    bincode::serde::decode_from_std_read(&mut staged, bincode::config::standard())
                })
                .collect::<Result<Vec<StagedRow>, _>>()
                .map_err(|e| Error::wrap(format!("failed to read {}", self.path.display()), e))?;
            remaining -= rows.len() as u64;
            let df = build_chunk(&rows, &schema)?;
            writer.write_batch(&df).map_err(|e| failed(e.into()))?;
        }
        writer.finish().map_err(|e| failed(e.into()))?;
        Ok((self.row_count, schema))
    }
}

/// The number of staged rows converted into a *Parquet* table at a time.
const CHUNK_ROWS: usize = 50_000;

fn staging_dir(root: &Path) -> PathBuf {
    root.join(".staging")
}

fn create_dir(path: &Path) -> Result<(), Error> {
    fs::create_dir_all(path)
        .map_err(|e| Error::wrap(format!("failed to create {}", path.display()), e))
}

/// Build a table chunk from `rows` using the `schema`.
fn build_chunk(rows: &[StagedRow], schema: &[PropertySchema]) -> Result<DataFrame, Error> {
    let u64_column = |name: &str, value: fn(&StagedRow) -> Option<u64>| {
        UInt64Chunked::from_iter_options(name.into(), rows.iter().map(value)).into_column()
    };
    let mut columns = vec![u64_column(ID_COL, |row| Some(row.id))];
    if rows.iter().any(|row| row.endpoints.is_some()) {
        columns.push(u64_column(SRC_COL, |row| row.endpoints.map(|(src, _)| src)));
        columns.push(u64_column(DST_COL, |row| row.endpoints.map(|(_, dst)| dst)));
    }
    for p in schema {
        let values: Vec<Option<&Value>> =
            rows.iter().map(|row| row.properties.get(&p.name)).collect();
        columns.push(build_column(&p.name, p.dtype, &values)?);
    }
    DataFrame::new(rows.len(), columns).map_err(|e| Error::wrap("failed to build table", e))
}

/// The labels of the nodes that `relationships`'s `endpoint` column refers to, among the
/// `node_tables`.
fn endpoint_labels(
    root: &Path,
    relationships: &str,
    endpoint: &str,
    node_tables: &[NodeTable],
) -> Result<Vec<String>, Error> {
    let relationships = root.join(relationships);
    let mut labels = Vec::new();
    for table in node_tables {
        let Some(label) = &table.label else {
            continue;
        };
        let ids = scan_parquet(&relationships)?.select([col(endpoint).alias(ID_COL)]);
        let nodes = scan_parquet(&root.join(&table.file))?.select([col(ID_COL)]);
        let matched = ids
            .join(
                nodes,
                [col(ID_COL)],
                [col(ID_COL)],
                JoinArgs::new(JoinType::Semi),
            )
            .limit(1)
            .collect()
            .map_err(|e| Error::wrap(format!("failed to read {}", relationships.display()), e))?;
        if matched.height() > 0 {
            labels.push(label.clone());
        }
    }
    Ok(labels)
}

/// The properties of a node or relationship, by name.
pub type Properties = BTreeMap<String, Value>;

/// A property value. Values without an equivalent variant should be converted to
/// [`Value::String`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum Value {
    /// A *Cypher* `BOOLEAN`.
    Bool(bool),
    /// A *Cypher* `INTEGER`.
    Int(i64),
    /// A *Cypher* `FLOAT`.
    Float(f64),
    /// A *Cypher* `STRING`.
    String(String),
    /// Days since the epoch.
    Date(i32),
    /// Microseconds since the epoch, in UTC when `utc` is set, and local otherwise.
    Datetime { micros: i64, utc: bool },
    /// A list, whose elements are widened to a common type when written.
    List(Vec<Value>),
}

impl Value {
    /// This value's column type, or `None` for an empty list (which carries no element type).
    fn property_type(&self) -> Option<PropertyType> {
        Some(match self {
            Value::Bool(_) => PropertyType::Boolean,
            Value::Int(_) => PropertyType::Int64,
            Value::Float(_) => PropertyType::Float64,
            Value::String(_) => PropertyType::String,
            Value::Date(_) => PropertyType::Date,
            Value::Datetime { utc, .. } => PropertyType::Datetime { utc: *utc },
            Value::List(items) => items
                .iter()
                .filter_map(Value::property_type)
                .reduce(unify)?
                .list_of(),
        })
    }
}

/// *Cypher* text of the [`Value`], used when a property is widened to (or stored as) a string.
impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Bool(b) => write!(f, "{b}"),
            Value::Int(i) => write!(f, "{i}"),
            Value::Float(x) if x.is_finite() && x.fract() == 0.0 => write!(f, "{x:.1}"),
            Value::Float(x) => write!(f, "{x}"),
            Value::String(s) => f.write_str(s),
            Value::Date(days) => match date_from_days(*days) {
                Some(date) => write!(f, "{}", date.format("%Y-%m-%d")),
                None => write!(f, "{days}"),
            },
            Value::Datetime { micros, utc } => match DateTime::from_timestamp_micros(*micros) {
                Some(dt) if *utc => f.write_str(&dt.to_rfc3339_opts(SecondsFormat::AutoSi, true)),
                Some(dt) => write!(f, "{}", dt.naive_utc().format("%Y-%m-%dT%H:%M:%S%.f")),
                None => write!(f, "{micros}"),
            },
            Value::List(items) => {
                f.write_str("[")?;
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        f.write_str(", ")?;
                    }
                    write!(f, "{item}")?;
                }
                f.write_str("]")
            }
        }
    }
}

/// The tables of a [`Graph`], as recorded in its `manifest.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Manifest {
    /// The on-disk layout version the graph was written with.
    pub version: u32,
    /// A table per node label, and one for nodes without labels if there are any.
    pub nodes: Vec<NodeTable>,
    /// A table per relationship type.
    pub relationships: Vec<RelTable>,
}

impl Manifest {
    fn load(path: &Path) -> Result<Self, Error> {
        let text = fs::read_to_string(path).map_err(|e| {
            Error::wrap(format!("failed to read manifest at {}", path.display()), e)
        })?;
        serde_json::from_str(&text)
            .map_err(|e| Error::wrap(format!("failed to parse manifest at {}", path.display()), e))
    }

    fn save(&self, path: &Path) -> Result<(), Error> {
        let text = serde_json::to_string_pretty(self)
            .map_err(|e| Error::wrap("failed to serialize manifest", e))?;
        fs::write(path, text)
            .map_err(|e| Error::wrap(format!("failed to write manifest to {}", path.display()), e))
    }

    pub(crate) fn node_table(&self, label: &str) -> Option<&NodeTable> {
        self.nodes
            .iter()
            .find(|t| t.label.as_deref() == Some(label))
    }

    pub(crate) fn rel_table(&self, rel_type: &str) -> Option<&RelTable> {
        self.relationships.iter().find(|t| t.rel_type == rel_type)
    }

    /// All relationship types present in the graph.
    pub(crate) fn rel_types(&self) -> impl Iterator<Item = &str> {
        self.relationships.iter().map(|t| t.rel_type.as_str())
    }
}

/// A table of the nodes with a label.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct NodeTable {
    /// The label, or `None` for the table of nodes without labels.
    pub label: Option<String>,
    /// The *Parquet* file, relative to the graph directory.
    pub file: String,
    /// The number of nodes with the label.
    pub row_count: u64,
    /// The property columns, after the `_id` column.
    pub schema: Vec<PropertySchema>,
}

/// A table of the relationships of a type.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct RelTable {
    /// The relationship type.
    #[serde(rename = "type")]
    pub rel_type: String,
    /// The *Parquet* file, relative to the graph directory.
    pub file: String,
    /// The number of relationships of the type.
    pub row_count: u64,
    /// The labels of the relationships' source nodes.
    pub src_labels: Vec<String>,
    /// The labels of the relationships' target nodes.
    pub dst_labels: Vec<String>,
    /// The property columns, after the `_id`, `_src`, and `_dst` columns.
    pub schema: Vec<PropertySchema>,
}

/// A property column in a [`NodeTable`] or [`RelTable`].
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct PropertySchema {
    /// The property and column name.
    pub name: String,
    /// The column type, shared by every table with the property.
    #[serde(rename = "type")]
    pub dtype: PropertyType,
    /// Whether the property held values of more than one type, and was widened.
    #[serde(default)]
    pub coerced: bool,
}

/// The *Polars* column type of a property.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum PropertyType {
    Boolean,
    Int64,
    Float64,
    String,
    Date,
    /// A microsecond datetime, in UTC when `utc` is set, and local otherwise.
    Datetime {
        utc: bool,
    },
    ListBoolean,
    ListInt64,
    ListFloat64,
    ListString,
}

impl PropertyType {
    pub(crate) fn to_polars(self) -> DataType {
        match self {
            PropertyType::Boolean => DataType::Boolean,
            PropertyType::Int64 => DataType::Int64,
            PropertyType::Float64 => DataType::Float64,
            PropertyType::String => DataType::String,
            PropertyType::Date => DataType::Date,
            PropertyType::Datetime { utc } => {
                DataType::Datetime(TimeUnit::Microseconds, utc.then_some(TimeZone::UTC))
            }
            PropertyType::ListBoolean => DataType::List(Box::new(DataType::Boolean)),
            PropertyType::ListInt64 => DataType::List(Box::new(DataType::Int64)),
            PropertyType::ListFloat64 => DataType::List(Box::new(DataType::Float64)),
            PropertyType::ListString => DataType::List(Box::new(DataType::String)),
        }
    }

    /// The inner type for a list type, or `None` for a scalar type.
    pub(crate) fn inner(self) -> Option<PropertyType> {
        match self {
            PropertyType::ListBoolean => Some(PropertyType::Boolean),
            PropertyType::ListInt64 => Some(PropertyType::Int64),
            PropertyType::ListFloat64 => Some(PropertyType::Float64),
            PropertyType::ListString => Some(PropertyType::String),
            _ => None,
        }
    }

    /// The list type whose inner elements are `self`.
    fn list_of(self) -> PropertyType {
        match self {
            PropertyType::Boolean => PropertyType::ListBoolean,
            PropertyType::Int64 => PropertyType::ListInt64,
            PropertyType::Float64 => PropertyType::ListFloat64,
            _ => PropertyType::ListString,
        }
    }
}

fn manifest_path(root: &Path) -> PathBuf {
    root.join("manifest.json")
}

/// The current on-disk layout version.
const LAYOUT_VERSION: u32 = 0;

fn scan_parquet(path: &Path) -> Result<LazyFrame, Error> {
    let path_str = path
        .to_str()
        .ok_or_else(|| Error::new(format!("path {} is not valid UTF-8", path.display())))?;
    LazyFrame::scan_parquet(PlRefPath::new(path_str.to_string()), Default::default())
        .map_err(|e| Error::wrap(format!("failed to scan {}", path.display()), e))
}

const NODES_DIR: &str = "nodes";

const RELATIONSHIPS_DIR: &str = "relationships";

/// The internal identity column name on every node and relationship table.
pub(crate) const ID_COL: &str = "_id";

/// The relationship source-endpoint column name.
pub(crate) const SRC_COL: &str = "_src";

/// The relationship target-endpoint column name.
pub(crate) const DST_COL: &str = "_dst";

/// The column of a relationship's type, added by queries.
pub(crate) const TYPE_COL: &str = "_type";

/// The column of a node's sorted labels, added by queries.
pub(crate) const LABELS_COL: &str = "_labels";

/// The type observed so far per property name, and whether it has been widened. The type is
/// `None` while a property has only been observed as empty lists.
type ObservedTypes = HashMap<String, (Option<PropertyType>, bool)>;

fn observe(types: &mut ObservedTypes, properties: &Properties) {
    for (name, value) in properties {
        let (resolved, coerced) = types.entry(name.clone()).or_insert((None, false));
        if let Some(ty) = value.property_type() {
            *coerced |= resolved.is_some_and(|r| r != ty);
            *resolved = Some(resolved.map_or(ty, |r| unify(r, ty)));
        }
    }
}

fn resolve(types: &ObservedTypes) -> PropertyTypes {
    types
        .iter()
        // a property only ever observed as an empty list has no element type to go on
        .map(|(name, &(ty, coerced))| {
            (
                name.clone(),
                (ty.unwrap_or(PropertyType::ListString), coerced),
            )
        })
        .collect()
}

/// Resolved `(type, coerced)` per property name. Types are resolved across the whole graph, not
/// per table, so that a multi-label node's property has the same type in each of its label
/// tables, and unlabeled scans can concatenate tables without type conflicts.
type PropertyTypes = HashMap<String, (PropertyType, bool)>;

/// The narrowest type that can represent values of both `a` and `b`. Integers widen to floats,
/// and any other combination falls back to strings.
fn unify(a: PropertyType, b: PropertyType) -> PropertyType {
    use PropertyType::*;
    match (a, b) {
        _ if a == b => a,
        (Int64, Float64) | (Float64, Int64) => Float64,
        (ListInt64, ListFloat64) | (ListFloat64, ListInt64) => ListFloat64,
        _ if a.inner().is_some() && b.inner().is_some() => ListString,
        _ => String,
    }
}

/// Build a column of `dtype` from `values`.
///
/// `dtype` was resolved by [`unify`] over every value, so each value is either already of that
/// type or widens to it.
fn build_column(
    name: &str,
    dtype: PropertyType,
    values: &[Option<&Value>],
) -> Result<Column, Error> {
    let name = PlSmallStr::from(name);
    let column = match dtype {
        PropertyType::Boolean => BooleanChunked::from_iter_options(
            name,
            values.iter().map(|v| match v {
                Some(Value::Bool(b)) => Some(*b),
                _ => None,
            }),
        )
        .into_column(),
        PropertyType::Int64 => Int64Chunked::from_iter_options(
            name,
            values.iter().map(|v| match v {
                Some(Value::Int(i)) => Some(*i),
                _ => None,
            }),
        )
        .into_column(),
        PropertyType::Float64 => Float64Chunked::from_iter_options(
            name,
            values.iter().map(|v| match v {
                Some(Value::Float(x)) => Some(*x),
                Some(Value::Int(i)) => Some(*i as f64),
                _ => None,
            }),
        )
        .into_column(),
        PropertyType::String => {
            StringChunked::from_iter_options(name, values.iter().map(|v| v.map(Value::to_string)))
                .into_column()
        }
        PropertyType::Date => Int32Chunked::from_iter_options(
            name,
            values.iter().map(|v| match v {
                Some(Value::Date(days)) => Some(*days),
                _ => None,
            }),
        )
        .into_date()
        .into_column(),
        PropertyType::Datetime { utc } => Int64Chunked::from_iter_options(
            name,
            values.iter().map(|v| match v {
                Some(Value::Datetime { micros, .. }) => Some(*micros),
                _ => None,
            }),
        )
        .into_datetime(TimeUnit::Microseconds, utc.then_some(TimeZone::UTC))
        .into_column(),
        list => {
            let element = list.inner().unwrap_or(PropertyType::String);
            let rows = values
                .iter()
                .map(|v| match v {
                    Some(Value::List(items)) => {
                        let items: Vec<Option<&Value>> = items.iter().map(Some).collect();
                        let inner = build_column("", element, &items)?;
                        Ok(Some(inner.as_materialized_series().clone()))
                    }
                    _ => Ok(None),
                })
                .collect::<Result<Vec<_>, Error>>()?;
            let mut chunked: ListChunked = rows.into_iter().collect();
            chunked.rename(name);
            // an all-null column is collected with a `Null` inner type
            chunked
                .into_column()
                .cast(&list.to_polars())
                .map_err(|e| Error::wrap("failed to build list column", e))?
        }
    };
    Ok(column)
}

fn date_from_days(days: i32) -> Option<NaiveDate> {
    NaiveDate::from_ymd_opt(1970, 1, 1)?.checked_add_signed(chrono::TimeDelta::days(days as i64))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multi_label_nodes() {
        let graph = write_graph("multi-label", |w| {
            w.add_node(
                ["Actor", "Person"],
                properties(&[("name", Value::String("Ada".into()))]),
            )?;
            w.add_node(
                ["Person"],
                properties(&[("name", Value::String("Cy".into()))]),
            )?;
            Ok(())
        });
        let manifest = graph.manifest();
        assert_eq!(manifest.node_table("Actor").unwrap().row_count, 1);
        assert_eq!(manifest.node_table("Person").unwrap().row_count, 2);

        let actor = graph.scan_node_table("Actor").unwrap().collect().unwrap();
        assert_eq!(u64s(&actor, ID_COL), vec![0]);
        let _ = fs::remove_dir_all(&graph.root);
    }

    #[test]
    fn widen_types() {
        let graph = write_graph("widen", |w| {
            w.add_node(
                ["A"],
                properties(&[("n", Value::Int(1)), ("s", Value::Int(1))]),
            )?;
            w.add_node(
                ["B"],
                properties(&[("n", Value::Float(2.5)), ("s", Value::String("x".into()))]),
            )?;
            w.add_node(["C"], properties(&[("k", Value::Int(3))]))?;
            Ok(())
        });
        let manifest = graph.manifest();

        // `n` is only an integer on `A`, but widens there too, matching `B`
        let n = property(&manifest.node_table("A").unwrap().schema, "n");
        assert_eq!((n.dtype, n.coerced), (PropertyType::Float64, true));
        let s = property(&manifest.node_table("A").unwrap().schema, "s");
        assert_eq!((s.dtype, s.coerced), (PropertyType::String, true));
        let k = property(&manifest.node_table("C").unwrap().schema, "k");
        assert_eq!((k.dtype, k.coerced), (PropertyType::Int64, false));

        let a = graph.scan_node_table("A").unwrap().collect().unwrap();
        assert_eq!(a.column("n").unwrap().f64().unwrap().get(0), Some(1.0));
        assert_eq!(a.column("s").unwrap().str().unwrap().get(0), Some("1"));
        let _ = fs::remove_dir_all(&graph.root);
    }

    #[test]
    fn preserve_temporal_and_list_types() {
        let graph = write_graph("temporal-list", |w| {
            let at = Value::Datetime {
                micros: 1_000,
                utc: true,
            };
            w.add_node(
                ["T"],
                properties(&[
                    ("day", Value::Date(1)),
                    ("at", at),
                    ("ints", Value::List(vec![Value::Int(1), Value::Int(2)])),
                    ("empty", Value::List(vec![])),
                ]),
            )?;
            w.add_node(["T"], properties(&[("ints", Value::List(vec![]))]))?;
            Ok(())
        });
        let manifest = graph.manifest();
        let schema = &manifest.node_table("T").unwrap().schema;
        assert_eq!(property(schema, "day").dtype, PropertyType::Date);
        assert_eq!(
            property(schema, "at").dtype,
            PropertyType::Datetime { utc: true }
        );
        assert_eq!(property(schema, "ints").dtype, PropertyType::ListInt64);
        assert_eq!(property(schema, "empty").dtype, PropertyType::ListString);

        let t = graph.scan_node_table("T").unwrap().collect().unwrap();
        for p in schema {
            assert_eq!(t.column(&p.name).unwrap().dtype(), &p.dtype.to_polars());
        }
        let _ = fs::remove_dir_all(&graph.root);
    }

    #[test]
    fn keep_self_loops_and_parallel_edges() {
        let graph = write_graph("relationships", |w| {
            let a = w.add_node(["A"], Properties::new())?;
            let b = w.add_node(["B"], Properties::new())?;
            for (src, dst) in [(b, a), (a, a), (a, b), (a, b)] {
                w.add_relationship(src, dst, "R", Properties::new())?;
            }
            Ok(())
        });
        let manifest = graph.manifest();
        let table = manifest.rel_table("R").unwrap();
        assert_eq!(table.row_count, 4);
        assert_eq!(table.src_labels, vec!["A", "B"]);
        assert_eq!(table.dst_labels, vec!["A", "B"]);

        let df = graph.scan_rel_table("R").unwrap().collect().unwrap();
        assert_eq!(u64s(&df, SRC_COL), vec![1, 0, 0, 0]);
        assert_eq!(u64s(&df, DST_COL), vec![0, 0, 1, 1]);
        // relationship ids follow node ids and stay distinct for parallel edges
        assert_eq!(u64s(&df, ID_COL), vec![2, 3, 4, 5]);
        let _ = fs::remove_dir_all(&graph.root);
    }

    #[test]
    fn write_tables() {
        let graph = write_graph("chunks", |w| {
            for _ in 0..CHUNK_ROWS {
                w.add_node(["A"], properties(&[("n", Value::Int(1))]))?;
            }
            // `late` is only in the second chunk, and widens `n` after the first was staged
            w.add_node(
                ["A"],
                properties(&[("n", Value::Float(2.5)), ("late", Value::Bool(true))]),
            )?;
            Ok(())
        });
        let table = graph.manifest().node_table("A").unwrap();
        assert_eq!(table.row_count, CHUNK_ROWS as u64 + 1);
        assert_eq!(property(&table.schema, "n").dtype, PropertyType::Float64);

        let df = graph.scan_node_table("A").unwrap().collect().unwrap();
        let n = df.column("n").unwrap().f64().unwrap();
        assert_eq!((n.get(0), n.get(CHUNK_ROWS)), (Some(1.0), Some(2.5)));
        let late = df.column("late").unwrap().bool().unwrap();
        assert_eq!(
            (late.null_count(), late.get(CHUNK_ROWS)),
            (CHUNK_ROWS, Some(true))
        );
        let _ = fs::remove_dir_all(&graph.root);
    }

    #[test]
    fn stage_floats() {
        let floats = [f64::NAN, f64::INFINITY, f64::NEG_INFINITY];
        let graph = write_graph("floats", |w| {
            for x in floats {
                let list = Value::List(vec![Value::Float(x)]);
                w.add_node(["A"], properties(&[("x", Value::Float(x)), ("xs", list)]))?;
            }
            Ok(())
        });
        let df = graph.scan_node_table("A").unwrap().collect().unwrap();
        let x: Vec<f64> = df
            .column("x")
            .unwrap()
            .f64()
            .unwrap()
            .iter()
            .flatten()
            .collect();
        assert!(x[0].is_nan());
        assert_eq!(&x[1..], &floats[1..]);
        let xs = df.column("xs").unwrap().list().unwrap();
        let first = xs.get_as_series(0).unwrap();
        assert!(first.f64().unwrap().get(0).unwrap().is_nan());
        let _ = fs::remove_dir_all(&graph.root);
    }

    #[test]
    fn name_files_safely() {
        let graph = write_graph("files", |w| {
            for labels in [
                vec!["Person"],
                vec!["PERSON"],
                vec!["a/../b"],
                vec!["_unlabeled"],
            ] {
                w.add_node(labels, Properties::new())?;
            }
            w.add_node(Vec::<&str>::new(), Properties::new())?;
            Ok(())
        });
        let files: Vec<(Option<&str>, &str)> = graph
            .manifest()
            .nodes
            .iter()
            .map(|t| (t.label.as_deref(), t.file.as_str()))
            .collect();
        assert_eq!(
            files,
            vec![
                (None, "nodes/_unlabeled.parquet"),
                (Some("PERSON"), "nodes/PERSON.parquet"),
                (Some("Person"), "nodes/Person-1.parquet"),
                (Some("_unlabeled"), "nodes/_unlabeled-1.parquet"),
                (Some("a/../b"), "nodes/a____b.parquet"),
            ]
        );
        let traversal = graph.scan_node_table("a/../b").unwrap().collect().unwrap();
        assert_eq!(traversal.height(), 1);
        let _ = fs::remove_dir_all(&graph.root);
    }

    #[test]
    fn reject_reserved_property_names() {
        let out_dir = std::env::temp_dir().join(format!(
            "polars-cypher-layout-test-{}-reserved",
            std::process::id()
        ));
        let mut writer = GraphWriter::create(&out_dir).unwrap();
        let err = writer
            .add_node(["A"], properties(&[("_id", Value::Int(1))]))
            .unwrap_err();
        assert_eq!(err.to_string(), "property name `_id` is reserved");
        let _ = fs::remove_dir_all(&out_dir);
    }

    #[test]
    fn reject_other_layout_versions() {
        let graph = write_graph("version", |w| {
            w.add_node(["A"], Properties::new()).map(drop)
        });
        let mut manifest = graph.manifest().clone();
        manifest.version = LAYOUT_VERSION + 1;
        manifest.save(&manifest_path(&graph.root)).unwrap();
        let err = Graph::open(&graph.root).unwrap_err();
        assert!(
            err.to_string().contains("only version 0 is supported"),
            "{err}"
        );
        let _ = fs::remove_dir_all(&graph.root);
    }

    #[test]
    fn format_widened_values() {
        assert_eq!(Value::Float(3.0).to_string(), "3.0");
        assert_eq!(Value::Float(2.5).to_string(), "2.5");
        assert_eq!(Value::Date(0).to_string(), "1970-01-01");
        let utc = Value::Datetime {
            micros: 0,
            utc: true,
        };
        assert_eq!(utc.to_string(), "1970-01-01T00:00:00Z");
    }

    /// Write a graph with the `build` function.
    fn write_graph(name: &str, build: impl FnOnce(&mut GraphWriter) -> Result<(), Error>) -> Graph {
        let out_dir = std::env::temp_dir().join(format!(
            "polars-cypher-layout-test-{}-{name}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&out_dir);
        let mut writer = GraphWriter::create(&out_dir).unwrap();
        build(&mut writer).unwrap_or_else(|e| panic!("add failed: {e}"));
        let graph = writer
            .write()
            .unwrap_or_else(|e| panic!("finish failed: {e}"));
        assert!(!staging_dir(&out_dir).exists());
        graph
    }

    fn properties(props: &[(&str, Value)]) -> Properties {
        props
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect()
    }

    fn property<'a>(schema: &'a [PropertySchema], name: &str) -> &'a PropertySchema {
        schema.iter().find(|p| p.name == name).expect(name)
    }

    fn u64s(df: &DataFrame, name: &str) -> Vec<u64> {
        df.column(name)
            .unwrap()
            .u64()
            .unwrap()
            .iter()
            .flatten()
            .collect()
    }
}
