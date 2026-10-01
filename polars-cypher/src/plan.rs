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

//! Binder, logical plan, join-order heuristic, and lowering to a *Polars* `LazyFrame`.

use std::collections::{HashMap, HashSet, VecDeque};
use std::error::Error as StdError;
use std::fmt;

use polars::prelude::*;

use crate::Error;
use crate::cypher::{
    BinaryOp, Direction, Expr, Literal, MapLiteral, Pattern, PatternElement, Query, ReturnClause,
    Span, StringMatchOp, parse, render,
};
use crate::layout::{
    DST_COL, Graph, ID_COL, LABELS_COL, PropertySchema, PropertyType, SRC_COL, TYPE_COL,
};
use crate::query::BoundParameter;

/// Parse, bind, plan, then lower `cypher` for the [`Graph`].
///
/// Returns the [`LogicalPlan`] and the [`LazyFrame`].
pub fn plan_query(
    graph: &Graph,
    cypher: &str,
    params: &HashMap<String, BoundParameter>,
) -> Result<(LogicalPlan, LazyFrame), Error> {
    let query = parse(cypher).map_err(|e| Error::new(render(&e.message, e.span, cypher)))?;
    let logical = bind(graph, &query, cypher).map_err(|e| match e.span {
        Some(span) => Error::new(render(&e.message, span, cypher)),
        None => Error::new(e.message),
    })?;
    let lf = lower(&logical, graph, params, cypher)?;
    Ok((logical, lf))
}

/// The operations between the bound query and its lowering to a *Polars* [`LazyFrame`].
#[derive(Debug, Clone)]
pub enum LogicalPlan {
    /// The starting scan for the plan.
    NodeScan {
        var: String,
        labels: Vec<String>,
        filter: Option<Expr>,
    },
    /// A join from `from_var` across relationships of `rel_types` (any type if empty) to `to`.
    Expand {
        input: Box<LogicalPlan>,
        from_var: String,
        rel_var: String,
        rel_types: Vec<String>,
        direction: Direction,
        rel_filter: Option<Expr>,
        to: ExpandTarget,
    },
    /// A cross (Cartesian) join with a disconnected pattern component.
    CrossJoin {
        left: Box<LogicalPlan>,
        right: Box<LogicalPlan>,
    },
    /// A filter by a predicate spanning several variables.
    Filter {
        input: Box<LogicalPlan>,
        predicate: Expr,
    },
    /// The `RETURN` items of a query without aggregates.
    Project {
        input: Box<LogicalPlan>,
        items: Vec<ProjectItem>,
        distinct: bool,
    },
    /// `items` includes both the implicit `GROUP BY` keys and the aggregate outputs, which is
    /// determined by each item's `ProjectExpr` variant.
    Aggregate {
        input: Box<LogicalPlan>,
        items: Vec<ProjectItem>,
        distinct: bool,
    },
    /// `ORDER BY` the `(column, ascending)` keys, then drop the `hidden` columns, which were
    /// projected only to sort by expressions that aren't returned.
    Sort {
        input: Box<LogicalPlan>,
        keys: Vec<(String, bool)>,
        hidden: Vec<String>,
    },
    /// `SKIP` and `LIMIT`, each a non-negative integer literal or parameter.
    Slice {
        input: Box<LogicalPlan>,
        skip: Option<Expr>,
        limit: Option<Expr>,
    },
}

/// The node an [`LogicalPlan::Expand`] reaches.
#[derive(Debug, Clone)]
pub enum ExpandTarget {
    /// The relationship reaches a node variable not seen before in the plan.
    New(NewNode),
    /// The relationship reaches a node variable already bound earlier (a cycle in the pattern),
    /// lowered as an equality filter instead of a fresh scan.
    Existing { var: String },
}

/// A projected output column, named by its `AS` alias, or else by its expression's text.
#[derive(Debug, Clone)]
pub struct ProjectItem {
    pub name: String,
    pub expr: ProjectExpr,
}

/// Where a new node variable's scan comes from when reached by an [`LogicalPlan::Expand`].
#[derive(Debug, Clone)]
pub struct NewNode {
    pub var: String,
    pub labels: Vec<String>,
    pub filter: Option<Expr>,
}

/// What a projected column computes.
#[derive(Debug, Clone)]
pub enum ProjectExpr {
    /// An ordinary scalar expression.
    Scalar(Expr),
    /// The node or relationship entity as a struct column.
    Entity { var: String, kind: EntityKind },
    /// An aggregate function applied to `arg` (absent for `count(*)`).
    Aggregate {
        func: AggFunc,
        arg: Option<Expr>,
        distinct: bool,
    },
}

/// Whether a whole-entity [`ProjectExpr::Entity`] is a node or a relationship.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntityKind {
    Node,
    Relationship,
}

/// An aggregate function in a [`ProjectExpr::Aggregate`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggFunc {
    CountStar,
    Count,
    Sum,
    Avg,
    Min,
    Max,
    Collect,
}

/// Bind and plan the parsed `query` for the `graph`.
///
/// Resolves variables, labels, and relationship types per the [`Graph::manifest`], and names
/// unaliased `RETURN` items after their text in `source`. Splits `WHERE` into
/// conjuncts and orders joins starting from the most constrained node (see [`choose_start`]),
/// extending outward across the [`PatternGraph`]. A re-visited variable is treated as a cycle
/// (equality filter) rather than a new join.
fn bind(graph: &Graph, query: &Query, source: &str) -> Result<LogicalPlan, BindError> {
    let mut pg = PatternGraph::default();
    // inline property maps are conjuncts of `WHERE`, so they're checked and routed the same way
    let mut conjuncts: Vec<Expr> = Vec::new();
    for pattern in &query.patterns {
        collect_pattern(&mut pg, &mut conjuncts, pattern)?;
    }
    conjuncts.extend(query.where_clause.clone());
    let where_clause = conjuncts.into_iter().reduce(and);
    for (var, decl) in &pg.nodes {
        for label in &decl.labels {
            if graph.manifest().node_table(label).is_none() {
                return Err(BindError::new(
                    format!("unknown label `{label}` (on `{var}`)"),
                    decl.span,
                ));
            }
        }
    }
    for rel in &pg.rels {
        for ty in &rel.types {
            if graph.manifest().rel_table(ty).is_none() {
                return Err(BindError::new(
                    format!("unknown relationship type `{ty}`"),
                    rel.span,
                ));
            }
        }
    }

    let rel_vars: HashSet<String> = pg.rels.iter().map(|r| r.rel_var.clone()).collect();
    let node_vars: HashSet<String> = pg.nodes.iter().map(|(v, _)| v.clone()).collect();
    let items = query.return_clause.items.iter().map(|item| &item.expr);
    for expr in where_clause.iter().chain(items) {
        check_variables(expr, &pg)?;
    }

    let (mut single_var_filters, rel_filters, leftover) =
        split_where(where_clause.as_ref(), &node_vars, &rel_vars)?;
    let mut plan = build_join_tree(&pg, &mut single_var_filters, &rel_filters, graph)?;

    if let Some(extra) = leftover {
        plan = LogicalPlan::Filter {
            input: Box::new(plan),
            predicate: extra,
        };
    }

    plan = add_relationship_uniqueness_filters(plan, &pg);
    plan = apply_return(plan, &query.return_clause, &pg, source)?;
    Ok(plan)
}

/// Lower a [`LogicalPlan`] to a *Polars* [`LazyFrame`].
///
/// Internal columns are named `var.prop`, with `var.`[`ID_COL`] for identity. Relationship
/// endpoints are joined through temporary [`SRC_COL`] and [`DST_COL`] columns that never survive
/// past the [`LogicalPlan::Expand`] that consumes them.
fn lower(
    plan: &LogicalPlan,
    graph: &Graph,
    params: &HashMap<String, BoundParameter>,
    source: &str,
) -> Result<LazyFrame, Error> {
    let mut ctx = LowerCtx {
        graph,
        params,
        source,
        var_labels: HashMap::new(),
        var_rel_types: HashMap::new(),
        label_vars: HashSet::new(),
    };
    collect_vars(plan, &mut ctx);
    lower_plan(plan, &ctx)
}

/// A binding or planning failure, pointing at the offending construct in the query text.
#[derive(Debug, Clone)]
struct BindError {
    pub message: String,
    pub span: Option<Span>,
}

impl BindError {
    fn new(message: impl Into<String>, span: Span) -> Self {
        Self {
            message: message.into(),
            span: Some(span),
        }
    }

    fn unsupported(what: impl Into<String>, span: Span) -> Self {
        Self {
            message: format!("unsupported: {}", what.into()),
            span: Some(span),
        }
    }
}

impl fmt::Display for BindError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl StdError for BindError {}

/// The connectivity graph for a query's `MATCH` patterns. Specifies node variables (merged across
/// repeated occurrences) and the relationships connecting them.
#[derive(Debug, Default)]
struct PatternGraph {
    /// Insertion-ordered for deterministic planning.
    pub nodes: Vec<(String, NodeDecl)>,
    pub rels: Vec<RelDecl>,
}

impl PatternGraph {
    pub fn node(&self, var: &str) -> Option<&NodeDecl> {
        self.nodes.iter().find(|(v, _)| v == var).map(|(_, d)| d)
    }

    pub fn node_mut(&mut self, var: &str) -> Option<&mut NodeDecl> {
        self.nodes
            .iter_mut()
            .find(|(v, _)| v == var)
            .map(|(_, d)| d)
    }

    /// Whether `var` is a node or relationship variable of the pattern.
    pub fn binds(&self, var: &str) -> bool {
        self.node(var).is_some() || self.rels.iter().any(|r| r.rel_var == var)
    }

    pub fn adjacency(&self) -> HashMap<&str, Vec<usize>> {
        let mut adj: HashMap<&str, Vec<usize>> = HashMap::new();
        for (i, rel) in self.rels.iter().enumerate() {
            adj.entry(rel.left.as_str()).or_default().push(i);
            if rel.right != rel.left {
                adj.entry(rel.right.as_str()).or_default().push(i);
            }
        }
        adj
    }
}

fn collect_pattern(
    pg: &mut PatternGraph,
    conjuncts: &mut Vec<Expr>,
    pattern: &Pattern,
) -> Result<(), BindError> {
    let mut prev_var: Option<String> = None;
    let mut pending_rel: Option<PendingRel> = None;
    let mut anon = pg.nodes.len() + pg.rels.len();

    for element in &pattern.elements {
        match element {
            PatternElement::Node(np) => {
                let var = match &np.variable {
                    Some(v) => v.clone(),
                    None => {
                        anon += 1;
                        format!("_anon{anon}")
                    }
                };
                let pending_var = pending_rel.as_ref().map(|(v, ..)| v.as_str());
                if pending_var == Some(var.as_str()) || pg.rels.iter().any(|r| r.rel_var == var) {
                    return Err(BindError::new(
                        format!("`{var}` is already bound to a relationship"),
                        np.span,
                    ));
                }
                if let Some(map) = &np.properties {
                    conjuncts.extend(map_conjuncts(&var, map));
                }
                upsert_node(pg, &var, &np.labels, np.span);

                if let Some((rel_var, types, direction, span)) = pending_rel.take() {
                    pg.rels.push(RelDecl {
                        rel_var,
                        types,
                        direction,
                        left: prev_var
                            .clone()
                            .expect("relationship always follows a node"),
                        right: var.clone(),
                        span,
                    });
                }
                prev_var = Some(var);
            }
            PatternElement::Relationship(rp) => {
                // anonymous relationships are named too, since uniqueness filters need their ids
                let var = match &rp.variable {
                    Some(v) => v.clone(),
                    None => {
                        anon += 1;
                        format!("_anon{anon}")
                    }
                };
                // a relationship can't be traversed twice, so its variable can only be bound once
                if pg.node(&var).is_some() || pg.rels.iter().any(|r| r.rel_var == var) {
                    return Err(BindError::new(format!("`{var}` is already bound"), rp.span));
                }
                if let Some(map) = &rp.properties {
                    conjuncts.extend(map_conjuncts(&var, map));
                }
                pending_rel = Some((var, rp.types.clone(), rp.direction, rp.span));
            }
        }
    }
    Ok(())
}

/// Split `WHERE` into conjuncts, organizing each by the set of pattern variables it references,
/// single-node-variable predicates, single-relationship-variable predicates, and everything else
/// (kept as one leftover filter applied after the whole pattern is bound).
fn split_where(
    where_clause: Option<&Expr>,
    node_vars: &HashSet<String>,
    rel_vars: &HashSet<String>,
) -> Result<(VarFilters, VarFilters, Option<Expr>), BindError> {
    let mut node_filters: HashMap<String, Expr> = HashMap::new();
    let mut rel_filters: HashMap<String, Expr> = HashMap::new();
    let mut leftover: Option<Expr> = None;

    let Some(expr) = where_clause else {
        return Ok((node_filters, rel_filters, leftover));
    };

    for conjunct in flatten_and(expr) {
        let mut refs = HashSet::new();
        collect_variables(&conjunct, &mut refs);
        let refs: Vec<&String> = refs.iter().collect();
        match refs.as_slice() {
            [v] if node_vars.contains(*v) => {
                let entry = node_filters.remove(*v);
                let merged = match entry {
                    Some(existing) => and(existing, conjunct),
                    None => conjunct,
                };
                node_filters.insert((*v).clone(), merged);
            }
            [v] if rel_vars.contains(*v) => {
                let entry = rel_filters.remove(*v);
                let merged = match entry {
                    Some(existing) => and(existing, conjunct),
                    None => conjunct,
                };
                rel_filters.insert((*v).clone(), merged);
            }
            _ => {
                leftover = Some(match leftover {
                    Some(existing) => and(existing, conjunct),
                    None => conjunct,
                });
            }
        }
    }

    Ok((node_filters, rel_filters, leftover))
}

fn and(a: Expr, b: Expr) -> Expr {
    let span = a.span().to(b.span());
    Expr::Binary {
        op: BinaryOp::And,
        lhs: Box::new(a),
        rhs: Box::new(b),
        span,
    }
}

fn build_join_tree(
    pg: &PatternGraph,
    single_var_filters: &mut HashMap<String, Expr>,
    rel_filters: &HashMap<String, Expr>,
    graph: &Graph,
) -> Result<LogicalPlan, BindError> {
    if pg.nodes.is_empty() {
        return Err(BindError {
            message: "MATCH has no patterns".to_string(),
            span: None,
        });
    }

    let adjacency = pg.adjacency();
    let mut visited: HashSet<String> = HashSet::new();
    let mut remaining: Vec<&str> = pg.nodes.iter().map(|(v, _)| v.as_str()).collect();
    let mut plan: Option<LogicalPlan> = None;

    while !remaining.is_empty() {
        let start = choose_start(pg, &remaining, single_var_filters, graph.manifest());
        let start = start.to_string();
        visited.insert(start.clone());
        remaining.retain(|v| *v != start);

        let start_decl = pg.node(&start).expect("declared");
        let mut component = LogicalPlan::NodeScan {
            var: start.clone(),
            labels: start_decl.labels.clone(),
            filter: single_var_filters.remove(&start),
        };

        let mut used_rels: HashSet<usize> = HashSet::new();
        let mut queue: VecDeque<String> = VecDeque::new();
        queue.push_back(start.clone());

        while let Some(cur) = queue.pop_front() {
            let Some(rel_ids) = adjacency.get(cur.as_str()) else {
                continue;
            };
            for &ri in rel_ids {
                if used_rels.contains(&ri) {
                    continue;
                }
                let rel = &pg.rels[ri];
                let (from, to) = if rel.left == cur {
                    (rel.left.clone(), rel.right.clone())
                } else {
                    (rel.right.clone(), rel.left.clone())
                };
                used_rels.insert(ri);

                let is_new = !visited.contains(&to);
                let target = if is_new {
                    visited.insert(to.clone());
                    remaining.retain(|v| *v != to);
                    queue.push_back(to.clone());
                    let to_decl = pg.node(&to).expect("declared");
                    ExpandTarget::New(NewNode {
                        var: to.clone(),
                        labels: to_decl.labels.clone(),
                        filter: single_var_filters.remove(&to),
                    })
                } else {
                    ExpandTarget::Existing { var: to.clone() }
                };

                let effective_direction = if from == rel.left {
                    rel.direction
                } else {
                    flip(rel.direction)
                };

                let rel_filter = rel_filters.get(&rel.rel_var).cloned();

                component = LogicalPlan::Expand {
                    input: Box::new(component),
                    from_var: from,
                    rel_var: rel.rel_var.clone(),
                    rel_types: rel.types.clone(),
                    direction: effective_direction,
                    rel_filter,
                    to: target,
                };
            }
        }

        plan = Some(match plan {
            None => component,
            Some(existing) => LogicalPlan::CrossJoin {
                left: Box::new(existing),
                right: Box::new(component),
            },
        });
    }

    Ok(plan.expect("at least one component"))
}

/// Add [`ID_COL`] inequality filters between every pair of relationship variables that could
/// bind to the same underlying relationship (their type sets overlap, or either is unrestricted).
fn add_relationship_uniqueness_filters(plan: LogicalPlan, pg: &PatternGraph) -> LogicalPlan {
    let rels = &pg.rels;
    let mut extra: Option<Expr> = None;
    for i in 0..rels.len() {
        for j in (i + 1)..rels.len() {
            let (a, b) = (&rels[i], &rels[j]);
            let overlap = a.types.is_empty()
                || b.types.is_empty()
                || a.types.iter().any(|t| b.types.contains(t));
            if !overlap {
                continue;
            }
            let (va, vb) = (a.rel_var.clone(), b.rel_var.clone());
            let span = a.span.to(b.span);
            let ne = Expr::Binary {
                op: BinaryOp::Ne,
                lhs: Box::new(Expr::Property {
                    base: Box::new(Expr::Variable(va, span)),
                    name: ID_COL.to_string(),
                    span,
                }),
                rhs: Box::new(Expr::Property {
                    base: Box::new(Expr::Variable(vb, span)),
                    name: ID_COL.to_string(),
                    span,
                }),
                span,
            };
            extra = Some(match extra {
                Some(e) => and(e, ne),
                None => ne,
            });
        }
    }
    match extra {
        Some(predicate) => LogicalPlan::Filter {
            input: Box::new(plan),
            predicate,
        },
        None => plan,
    }
}

fn apply_return(
    plan: LogicalPlan,
    ret: &ReturnClause,
    pg: &PatternGraph,
    source: &str,
) -> Result<LogicalPlan, BindError> {
    let mut items = Vec::new();
    for item in &ret.items {
        // as in Neo4j, an unaliased column is named after the item's text
        let name = match &item.alias {
            Some(alias) => alias.clone(),
            None => source[item.span.start..item.span.end].to_string(),
        };
        let expr = classify_return_expr(&item.expr, pg)?;
        items.push(ProjectItem { name, expr });
    }
    let is_aggregate = items
        .iter()
        .any(|item| matches!(item.expr, ProjectExpr::Aggregate { .. }));
    if is_aggregate {
        for item in &items {
            if let ProjectExpr::Scalar(e) = &item.expr
                && contains_aggregate_call(e)
            {
                return Err(BindError::unsupported(
                    "aggregate functions nested inside another expression",
                    e.span(),
                ));
            }
        }
    }

    let mut keys = Vec::new();
    let mut hidden = Vec::new();
    for (i, order_item) in ret.order_by.iter().enumerate() {
        let name = match resolve_order_target(&order_item.expr, &items, pg)? {
            Some(name) => name,
            // project an expression that isn't returned as a hidden column to sort by
            None if !is_aggregate && !ret.distinct => {
                check_variables(&order_item.expr, pg)?;
                let expr = classify_return_expr(&order_item.expr, pg)?;
                if matches!(expr, ProjectExpr::Aggregate { .. }) {
                    return Err(BindError::unsupported(
                        "ORDER BY an aggregate that isn't returned",
                        order_item.span,
                    ));
                }
                let name = format!("_order{i}");
                items.push(ProjectItem {
                    name: name.clone(),
                    expr,
                });
                hidden.push(name.clone());
                name
            }
            None => {
                return Err(BindError::unsupported(
                    "ORDER BY an expression that isn't returned, with DISTINCT or aggregates",
                    order_item.span,
                ));
            }
        };
        keys.push((name, order_item.ascending));
    }

    let input = Box::new(plan);
    let distinct = ret.distinct;
    let mut plan = if is_aggregate {
        LogicalPlan::Aggregate {
            input,
            items,
            distinct,
        }
    } else {
        LogicalPlan::Project {
            input,
            items,
            distinct,
        }
    };
    if !keys.is_empty() {
        plan = LogicalPlan::Sort {
            input: Box::new(plan),
            keys,
            hidden,
        };
    }
    if ret.skip.is_some() || ret.limit.is_some() {
        plan = LogicalPlan::Slice {
            input: Box::new(plan),
            skip: ret.skip.clone(),
            limit: ret.limit.clone(),
        };
    }
    Ok(plan)
}

/// Fail on the first variable in `expr` that the pattern doesn't bind.
fn check_variables(expr: &Expr, pg: &PatternGraph) -> Result<(), BindError> {
    let mut unknown = None;
    expr.visit(&mut |e| {
        if let Expr::Variable(name, span) = e
            && !pg.binds(name)
        {
            unknown.get_or_insert_with(|| (name.clone(), *span));
        }
    });
    match unknown {
        Some((name, span)) => Err(BindError::new(format!("unknown variable `{name}`"), span)),
        None => Ok(()),
    }
}

/// Fill in the variable bookkeeping of `ctx` from `plan`.
fn collect_vars(plan: &LogicalPlan, ctx: &mut LowerCtx) {
    let mut exprs: Vec<&Expr> = Vec::new();
    match plan {
        LogicalPlan::NodeScan {
            var,
            labels,
            filter,
        } => {
            ctx.var_labels.insert(var.clone(), labels.clone());
            exprs.extend(filter);
        }
        LogicalPlan::Expand {
            input,
            rel_var,
            rel_types,
            rel_filter,
            to,
            ..
        } => {
            collect_vars(input, ctx);
            ctx.var_rel_types.insert(rel_var.clone(), rel_types.clone());
            exprs.extend(rel_filter);
            if let ExpandTarget::New(n) = to {
                ctx.var_labels.insert(n.var.clone(), n.labels.clone());
                exprs.extend(&n.filter);
            }
        }
        LogicalPlan::CrossJoin { left, right } => {
            collect_vars(left, ctx);
            collect_vars(right, ctx);
        }
        LogicalPlan::Filter { input, predicate } => {
            collect_vars(input, ctx);
            exprs.push(predicate);
        }
        LogicalPlan::Project { input, items, .. } | LogicalPlan::Aggregate { input, items, .. } => {
            collect_vars(input, ctx);
            for item in items {
                match &item.expr {
                    ProjectExpr::Scalar(e) => exprs.push(e),
                    ProjectExpr::Entity {
                        var,
                        kind: EntityKind::Node,
                    } => {
                        ctx.label_vars.insert(var.clone());
                    }
                    ProjectExpr::Entity { .. } => {}
                    ProjectExpr::Aggregate {
                        func: AggFunc::Collect,
                        arg: Some(Expr::Variable(var, _)),
                        ..
                    } => {
                        ctx.label_vars.insert(var.clone());
                    }
                    ProjectExpr::Aggregate { arg, .. } => exprs.extend(arg),
                }
            }
        }
        LogicalPlan::Sort { input, .. } | LogicalPlan::Slice { input, .. } => {
            collect_vars(input, ctx)
        }
    }
    for expr in exprs {
        expr.visit(&mut |e| {
            if let Expr::FunctionCall { name, args, .. } = e
                && name.eq_ignore_ascii_case("labels")
                && let [Expr::Variable(var, _)] = args.as_slice()
            {
                ctx.label_vars.insert(var.clone());
            }
        });
    }
}

struct LowerCtx<'a> {
    graph: &'a Graph,
    params: &'a HashMap<String, BoundParameter>,
    /// The query text, to render errors with.
    source: &'a str,
    /// Node variable to the labels it was scanned with (empty means unlabeled).
    var_labels: HashMap<String, Vec<String>>,
    /// Relationship variable to the types it was scanned with (empty means unrestricted).
    var_rel_types: HashMap<String, Vec<String>>,
    /// Node variables whose full label set is needed, as a [`LABELS_COL`] column.
    label_vars: HashSet<String>,
}

impl<'a> LowerCtx<'a> {
    /// The properties of the tables scanned for `var`, deduplicated by name.
    fn properties(&self, var: &str) -> Vec<&'a PropertySchema> {
        let manifest = self.graph.manifest();
        let schemas: Vec<&'a [PropertySchema]> = if let Some(labels) = self.var_labels.get(var) {
            match labels.first() {
                // an unlabeled scan concatenates every node table
                None => manifest.nodes.iter().map(|t| t.schema.as_slice()).collect(),
                // a multi-label scan reads the first label's table, which has every property
                Some(label) => manifest
                    .node_table(label)
                    .map(|t| t.schema.as_slice())
                    .into_iter()
                    .collect(),
            }
        } else if let Some(types) = self.var_rel_types.get(var) {
            if types.is_empty() {
                manifest
                    .relationships
                    .iter()
                    .map(|t| t.schema.as_slice())
                    .collect()
            } else {
                types
                    .iter()
                    .filter_map(|ty| manifest.rel_table(ty))
                    .map(|t| t.schema.as_slice())
                    .collect()
            }
        } else {
            Vec::new()
        };
        let mut seen = HashSet::new();
        schemas
            .into_iter()
            .flatten()
            .filter(|p| seen.insert(p.name.as_str()))
            .collect()
    }

    fn entity_kind(&self, var: &str) -> EntityKind {
        if self.var_labels.contains_key(var) {
            EntityKind::Node
        } else {
            EntityKind::Relationship
        }
    }

    /// Evaluate a `SKIP` or `LIMIT` expression, which must be a non-negative integer.
    fn row_count(&self, expr: &Expr) -> Result<i64, Error> {
        let n = match expr {
            Expr::Literal(Literal::Int(n), _) => Some(*n),
            Expr::Parameter(name, _) => match self.params.get(name) {
                Some(BoundParameter::Int(n)) => Some(*n),
                _ => None,
            },
            _ => None,
        };
        n.filter(|n| *n >= 0).ok_or_else(|| {
            self.error(
                expr.span(),
                "SKIP and LIMIT must be a non-negative integer literal or parameter",
            )
        })
    }

    /// An error with `message`, pointing at `span` in the query text.
    fn error(&self, span: Span, message: impl AsRef<str>) -> Error {
        Error::new(render(message.as_ref(), span, self.source))
    }

    fn unsupported(&self, span: Span, what: impl AsRef<str>) -> Error {
        self.error(span, format!("unsupported: {}", what.as_ref()))
    }

    /// The `name` property of the tables scanned for `var`.
    fn property(&self, var: &str, name: &str) -> Option<&'a PropertySchema> {
        self.properties(var).into_iter().find(|p| p.name == name)
    }
}

fn lower_plan(plan: &LogicalPlan, ctx: &LowerCtx) -> Result<LazyFrame, Error> {
    match plan {
        LogicalPlan::NodeScan {
            var,
            labels,
            filter,
        } => {
            let lf = scan_node_var(ctx, var, labels)?;
            apply_optional_filter(lf, filter, ctx)
        }
        LogicalPlan::Expand {
            input,
            from_var,
            rel_var,
            rel_types,
            direction,
            rel_filter,
            to,
        } => {
            let input_lf = lower_plan(input, ctx)?;
            lower_expand(
                input_lf, from_var, rel_var, rel_types, *direction, rel_filter, to, ctx,
            )
        }
        LogicalPlan::CrossJoin { left, right } => {
            let l = lower_plan(left, ctx)?;
            let r = lower_plan(right, ctx)?;
            Ok(l.cross_join(r, None))
        }
        LogicalPlan::Filter { input, predicate } => {
            let lf = lower_plan(input, ctx)?;
            let expr = lower_expr(predicate, ctx)?;
            Ok(lf.filter(expr))
        }
        LogicalPlan::Project {
            input,
            items,
            distinct,
        } => {
            let lf = lower_plan(input, ctx)?;
            let exprs = items
                .iter()
                .map(|item| lower_project_item(item, ctx))
                .collect::<Result<Vec<_>, _>>()?;
            let lf = lf.select(exprs);
            Ok(if *distinct {
                lf.unique_stable(None, UniqueKeepStrategy::First)
            } else {
                lf
            })
        }
        LogicalPlan::Aggregate {
            input,
            items,
            distinct,
        } => {
            let lf = lower_plan(input, ctx)?;
            let mut group_exprs = Vec::new();
            let mut agg_exprs = Vec::new();
            for item in items {
                match &item.expr {
                    ProjectExpr::Aggregate {
                        func,
                        arg,
                        distinct: agg_distinct,
                    } => {
                        agg_exprs.push(lower_aggregate(
                            *func,
                            arg.as_ref(),
                            *agg_distinct,
                            &item.name,
                            ctx,
                        )?);
                    }
                    _ => group_exprs.push(lower_project_item(item, ctx)?),
                }
            }
            let lf = if group_exprs.is_empty() {
                lf.select(agg_exprs)
            } else {
                lf.group_by(group_exprs).agg(agg_exprs)
            };
            Ok(if *distinct {
                lf.unique_stable(None, UniqueKeepStrategy::First)
            } else {
                lf
            })
        }
        LogicalPlan::Sort {
            input,
            keys,
            hidden,
        } => {
            let lf = lower_plan(input, ctx)?;
            let names: Vec<String> = keys.iter().map(|(n, _)| n.clone()).collect();
            let descending: Vec<bool> = keys.iter().map(|(_, asc)| !asc).collect();
            // as in Cypher, null sorts after every value, so last ascending and first descending
            let nulls_last: Vec<bool> = keys.iter().map(|(_, asc)| *asc).collect();
            let lf = lf.sort(
                names,
                SortMultipleOptions::default()
                    .with_order_descending_multi(descending)
                    .with_nulls_last_multi(nulls_last),
            );
            Ok(if hidden.is_empty() {
                lf
            } else {
                lf.select([all().exclude_cols(hidden.clone()).as_expr()])
            })
        }
        LogicalPlan::Slice { input, skip, limit } => {
            let lf = lower_plan(input, ctx)?;
            let offset = skip.as_ref().map(|e| ctx.row_count(e)).transpose()?;
            let len = limit.as_ref().map(|e| ctx.row_count(e)).transpose()?;
            // a limit beyond the index type's range can't truncate the rows anyway
            let len = len.map_or(IdxSize::MAX, |len| {
                IdxSize::try_from(len).unwrap_or(IdxSize::MAX)
            });
            Ok(lf.slice(offset.unwrap_or(0), len))
        }
    }
}

#[derive(Debug, Clone)]
struct NodeDecl {
    pub labels: Vec<String>,
    pub span: Span,
}

#[derive(Debug, Clone)]
struct RelDecl {
    pub rel_var: String,
    pub types: Vec<String>,
    pub direction: Direction,
    pub left: String,
    pub right: String,
    pub span: Span,
}

/// A relationship's variable, types, direction, and span, held between parsing its
/// [`PatternElement::Relationship`] and the node that follows it.
type PendingRel = (String, Vec<String>, Direction, Span);

/// The `var.key = value` equality per entry of an inline property map on `var`.
fn map_conjuncts<'a>(var: &'a str, map: &'a MapLiteral) -> impl Iterator<Item = Expr> + 'a {
    map.entries.iter().map(move |(key, value)| {
        let span = value.span();
        Expr::Binary {
            op: BinaryOp::Eq,
            lhs: Box::new(Expr::Property {
                base: Box::new(Expr::Variable(var.to_string(), span)),
                name: key.clone(),
                span,
            }),
            rhs: Box::new(value.clone()),
            span,
        }
    })
}

fn upsert_node(pg: &mut PatternGraph, var: &str, labels: &[String], span: Span) {
    if let Some(decl) = pg.node_mut(var) {
        for l in labels {
            if !decl.labels.contains(l) {
                decl.labels.push(l.clone());
            }
        }
    } else {
        pg.nodes.push((
            var.to_string(),
            NodeDecl {
                labels: labels.to_vec(),
                span,
            },
        ));
    }
}

/// Single-variable filters stored by variable name.
type VarFilters = HashMap<String, Expr>;

fn flatten_and(expr: &Expr) -> Vec<Expr> {
    match expr {
        Expr::Binary {
            op: BinaryOp::And,
            lhs,
            rhs,
            ..
        } => {
            let mut out = flatten_and(lhs);
            out.extend(flatten_and(rhs));
            out
        }
        other => vec![other.clone()],
    }
}

fn collect_variables(expr: &Expr, out: &mut HashSet<String>) {
    expr.visit(&mut |e| {
        if let Expr::Variable(name, _) = e {
            out.insert(name.clone());
        }
    });
}

/// Choose the next start variable among `candidates`. Prefers one with an equality predicate,
/// then the smallest table by estimated row count, otherwise the first encountered.
fn choose_start<'a>(
    graph: &PatternGraph,
    candidates: &[&'a str],
    filters: &HashMap<String, Expr>,
    manifest: &crate::layout::Manifest,
) -> &'a str {
    if let Some(v) = candidates
        .iter()
        .find(|v| filters.get(**v).is_some_and(has_equality))
    {
        return v;
    }
    candidates
        .iter()
        .min_by_key(|v| {
            let labels = graph.node(v).map(|d| d.labels.as_slice()).unwrap_or(&[]);
            estimated_row_count(manifest, labels)
        })
        .copied()
        .unwrap_or(candidates[0])
}

/// Reverse a relationship direction, used when a relationship is traversed from its textual right
/// endpoint toward its left one.
fn flip(direction: Direction) -> Direction {
    match direction {
        Direction::Left => Direction::Right,
        Direction::Right => Direction::Left,
        Direction::Undirected => Direction::Undirected,
    }
}

fn classify_return_expr(expr: &Expr, pg: &PatternGraph) -> Result<ProjectExpr, BindError> {
    if let Expr::Variable(name, _) = expr {
        let kind = if pg.node(name).is_some() {
            EntityKind::Node
        } else {
            EntityKind::Relationship
        };
        return Ok(ProjectExpr::Entity {
            var: name.clone(),
            kind,
        });
    }
    if let Expr::CountStar(_) = expr {
        return Ok(ProjectExpr::Aggregate {
            func: AggFunc::CountStar,
            arg: None,
            distinct: false,
        });
    }
    if let Expr::FunctionCall {
        name,
        args,
        distinct,
        span,
    } = expr
        && let Some(func) = agg_func(name)
    {
        if args.len() != 1 {
            return Err(BindError::unsupported(
                format!("`{name}` takes exactly one argument"),
                *span,
            ));
        }
        return Ok(ProjectExpr::Aggregate {
            func,
            arg: Some(args[0].clone()),
            distinct: *distinct,
        });
    }
    Ok(ProjectExpr::Scalar(expr.clone()))
}

fn contains_aggregate_call(expr: &Expr) -> bool {
    let mut found = false;
    expr.visit(&mut |e| {
        found |= match e {
            Expr::CountStar(_) => true,
            Expr::FunctionCall { name, .. } => agg_func(name).is_some(),
            _ => false,
        }
    });
    found
}

/// The name of the `RETURN` item that `expr` sorts by: an alias, or an equal expression.
fn resolve_order_target(
    expr: &Expr,
    items: &[ProjectItem],
    pg: &PatternGraph,
) -> Result<Option<String>, BindError> {
    if let Expr::Variable(name, _) = expr
        && let Some(item) = items.iter().find(|i| &i.name == name)
    {
        return Ok(Some(item.name.clone()));
    }
    let target = classify_return_expr(expr, pg)?;
    let item = items.iter().find(|item| match (&item.expr, &target) {
        (ProjectExpr::Scalar(a), ProjectExpr::Scalar(b)) => expr_eq_ignoring_span(a, b),
        (ProjectExpr::Entity { var: a, .. }, ProjectExpr::Entity { var: b, .. }) => a == b,
        (
            ProjectExpr::Aggregate {
                func: fa,
                arg: aa,
                distinct: da,
            },
            ProjectExpr::Aggregate {
                func: fb,
                arg: ab,
                distinct: db,
            },
        ) => {
            fa == fb
                && da == db
                && match (aa, ab) {
                    (Some(a), Some(b)) => expr_eq_ignoring_span(a, b),
                    (a, b) => a.is_none() && b.is_none(),
                }
        }
        _ => false,
    });
    Ok(item.map(|item| item.name.clone()))
}

fn scan_node_var(ctx: &LowerCtx, var: &str, labels: &[String]) -> Result<LazyFrame, Error> {
    let graph = ctx.graph;
    let lf = match labels {
        [] => {
            // every node table, including the unlabeled one
            let tables = &graph.manifest().nodes;
            if tables.is_empty() {
                return Err(Error::new("graph has no node tables"));
            }
            let frames = tables
                .iter()
                .map(|t| Ok(rename_node_columns(graph.scan(&t.file)?, var)))
                .collect::<Result<Vec<_>, Error>>()?;
            let combined = concat(
                frames,
                UnionArgs {
                    diagonal: true,
                    ..Default::default()
                },
            )
            .map_err(|e| Error::wrap("failed to concat node tables", e))?;
            combined.unique_stable(
                Some(cols([format!("{var}.{ID_COL}")])),
                UniqueKeepStrategy::First,
            )
        }
        [label] => rename_node_columns(graph.scan_node_table(label)?, var),
        [base, rest @ ..] => {
            let mut lf = rename_node_columns(graph.scan_node_table(base)?, var);
            for other in rest {
                let probe = graph.scan_node_table(other)?.select([col(ID_COL)]);
                lf = lf.join(
                    probe,
                    [col(format!("{var}.{ID_COL}"))],
                    [col(ID_COL)],
                    JoinArgs::new(JoinType::Semi),
                );
            }
            lf
        }
    };
    if ctx.label_vars.contains(var) {
        join_labels(lf, var, graph)
    } else {
        Ok(lf)
    }
}

/// Join the sorted list of every label of each node bound to `var`, as `var.`[`LABELS_COL`].
/// A node has a row in each of its labels' tables, so its labels are the tables with its id.
fn join_labels(lf: LazyFrame, var: &str, graph: &Graph) -> Result<LazyFrame, Error> {
    let id = format!("{var}.{ID_COL}");
    let labels = format!("{var}.{LABELS_COL}");
    let frames = graph
        .manifest()
        .nodes
        .iter()
        .map(|table| {
            let label = match &table.label {
                Some(label) => lit(label.clone()),
                None => lit(NULL).cast(DataType::String),
            };
            let lf = graph.scan(&table.file)?;
            Ok(lf.select([col(ID_COL).alias(id.as_str()), label.alias(labels.as_str())]))
        })
        .collect::<Result<Vec<_>, Error>>()?;
    // the unlabeled table's null label leaves its nodes with an empty list
    let lookup = concat(frames, UnionArgs::default())
        .map_err(|e| Error::wrap("failed to concat node labels", e))?
        .group_by([col(id.as_str())])
        .agg([col(labels.as_str())
            .drop_nulls()
            .sort(SortOptions::default())]);
    Ok(lf.join(
        lookup,
        [col(id.as_str())],
        [col(id.as_str())],
        JoinArgs::new(JoinType::Left),
    ))
}

fn apply_optional_filter(
    lf: LazyFrame,
    filter: &Option<Expr>,
    ctx: &LowerCtx,
) -> Result<LazyFrame, Error> {
    match filter {
        Some(expr) => Ok(lf.filter(lower_expr(expr, ctx)?)),
        None => Ok(lf),
    }
}

#[allow(clippy::too_many_arguments)]
fn lower_expand(
    input: LazyFrame,
    from_var: &str,
    rel_var: &str,
    rel_types: &[String],
    direction: Direction,
    rel_filter: &Option<Expr>,
    to: &ExpandTarget,
    ctx: &LowerCtx,
) -> Result<LazyFrame, Error> {
    let rel_lf = scan_rel_types(ctx.graph, rel_var, rel_types)?;
    let rel_lf = apply_optional_filter(rel_lf, rel_filter, ctx)?;

    // a single atomic `select` (rather than separate `rename`/`drop` calls) so projection
    // pushdown can't prune `from_col`/`to_col` out from under a later step that expects them
    let one_direction =
        |lf: LazyFrame, rel: LazyFrame, from_col: &'static str, to_col: &'static str| {
            lf.join(
                rel,
                [col(format!("{from_var}.{ID_COL}"))],
                [col(from_col)],
                keep_columns(JoinType::Inner),
            )
            .select([
                all().exclude_cols([from_col, to_col]).as_expr(),
                col(to_col).alias("__to_id"),
            ])
        };

    let joined = match direction {
        Direction::Right => one_direction(input, rel_lf, SRC_COL, DST_COL),
        Direction::Left => one_direction(input, rel_lf, DST_COL, SRC_COL),
        Direction::Undirected => {
            let forward = one_direction(input.clone(), rel_lf.clone(), SRC_COL, DST_COL);
            // as in Neo4j, an undirected pattern matches a self-loop once, not in both directions
            let rel_lf = rel_lf.filter(col(SRC_COL).neq(col(DST_COL)));
            let reverse = one_direction(input, rel_lf, DST_COL, SRC_COL);
            concat([forward, reverse], UnionArgs::default())
                .map_err(|e| Error::wrap("failed to union undirected expand", e))?
        }
    };

    match to {
        ExpandTarget::New(node) => {
            let target_lf = scan_node_var(ctx, &node.var, &node.labels)?;
            let target_lf = apply_optional_filter(target_lf, &node.filter, ctx)?;
            let out = joined
                .join(
                    target_lf,
                    [col("__to_id")],
                    [col(format!("{}.{ID_COL}", node.var))],
                    keep_columns(JoinType::Inner),
                )
                .select([all().exclude_cols(["__to_id"]).as_expr()]);
            Ok(out)
        }
        ExpandTarget::Existing { var } => {
            let out = joined
                .filter(col("__to_id").eq(col(format!("{var}.{ID_COL}"))))
                .select([all().exclude_cols(["__to_id"]).as_expr()]);
            Ok(out)
        }
    }
}

fn lower_expr(expr: &Expr, ctx: &LowerCtx) -> Result<PolarsExpr, Error> {
    Ok(match expr {
        Expr::Literal(lit_val, _) => lower_literal(lit_val),
        Expr::Parameter(name, span) => lower_param(name, *span, ctx)?,
        Expr::Variable(name, span) => {
            return Err(ctx.unsupported(*span, format!("bare variable `{name}` in an expression")));
        }
        Expr::Property { base, name, span } => match base.as_ref() {
            Expr::Variable(var, _) if name == ID_COL || ctx.property(var, name).is_some() => {
                col(format!("{var}.{name}"))
            }
            // as in Cypher, a property that none of the variable's tables have is null
            Expr::Variable(..) => lit(NULL),
            _ => {
                return Err(ctx.unsupported(*span, "property access on a non-variable expression"));
            }
        },
        Expr::Not(inner, _) => lower_expr(inner, ctx)?.not(),
        Expr::Neg(inner, _) => -lower_expr(inner, ctx)?,
        Expr::Binary { op, lhs, rhs, .. } => {
            let l = lower_expr(lhs, ctx)?;
            let r = lower_expr(rhs, ctx)?;
            match op {
                // Cypher's integer division truncates, where Polars' floors, but subtracting
                // the remainder first makes the division exact
                BinaryOp::Div if is_integer(lhs, ctx) && is_integer(rhs, ctx) => {
                    (l.clone() - remainder(l, r.clone())) / r
                }
                _ => lower_binary(*op, l, r),
            }
        }
        Expr::IsNull { expr, negated, .. } => {
            let e = lower_expr(expr, ctx)?;
            if *negated {
                e.is_not_null()
            } else {
                e.is_null()
            }
        }
        Expr::InList { expr, list, span } => {
            let needle = lower_expr(expr, ctx)?;
            match list.as_ref() {
                Expr::List(items, _) => {
                    // chained equality ORs, since list literals are small
                    let mut acc: Option<PolarsExpr> = None;
                    for item in items {
                        let v = lower_expr(item, ctx)?;
                        let eq = needle.clone().eq(v);
                        acc = Some(match acc {
                            Some(a) => a.or(eq),
                            None => eq,
                        });
                    }
                    acc.unwrap_or_else(|| lit(false))
                }
                _ => {
                    return Err(
                        ctx.unsupported(*span, "IN with a non-literal-list right-hand side")
                    );
                }
            }
        }
        Expr::StringMatch {
            op, expr, pattern, ..
        } => {
            let e = lower_expr(expr, ctx)?;
            let p = lower_expr(pattern, ctx)?;
            match op {
                StringMatchOp::StartsWith => e.str().starts_with(p),
                StringMatchOp::EndsWith => e.str().ends_with(p),
                StringMatchOp::Contains => e.str().contains_literal(p),
            }
        }
        Expr::List(items, span) => {
            if items.is_empty() {
                return Err(ctx.unsupported(*span, "empty list literal"));
            }
            let values = items
                .iter()
                .map(|i| lower_expr(i, ctx))
                .collect::<Result<Vec<_>, _>>()?;
            concat_list(values).map_err(|e| Error::wrap("failed to build list literal", e))?
        }
        Expr::FunctionCall {
            name, args, span, ..
        } => lower_function(name, args, *span, ctx)?,
        Expr::CountStar(span) => {
            return Err(ctx.unsupported(*span, "count(*) outside an aggregation"));
        }
    })
}

fn lower_project_item(item: &ProjectItem, ctx: &LowerCtx) -> Result<PolarsExpr, Error> {
    let expr = match &item.expr {
        ProjectExpr::Scalar(e) => lower_expr(e, ctx)?,
        ProjectExpr::Entity { var, kind } => lower_entity(var, *kind, ctx)?,
        ProjectExpr::Aggregate {
            func,
            arg,
            distinct,
        } => {
            return lower_aggregate(*func, arg.as_ref(), *distinct, &item.name, ctx);
        }
    };
    Ok(expr.alias(item.name.clone()))
}

fn lower_aggregate(
    func: AggFunc,
    arg: Option<&Expr>,
    distinct: bool,
    alias: &str,
    ctx: &LowerCtx,
) -> Result<PolarsExpr, Error> {
    let e = match (func, arg) {
        (AggFunc::CountStar, _) => return Ok(len().cast(DataType::Int64).alias(alias)),
        // a node or relationship is counted by its id, and collected as its entity struct
        (AggFunc::Count, Some(Expr::Variable(var, _))) => col(format!("{var}.{ID_COL}")),
        (AggFunc::Collect, Some(Expr::Variable(var, _))) => {
            lower_entity(var, ctx.entity_kind(var), ctx)?
        }
        (_, Some(arg)) => lower_expr(arg, ctx)?,
        (_, None) => return Err(Error::new(format!("`{alias}` requires an argument"))),
    };
    let e = if distinct { e.unique_stable() } else { e };
    let out = match func {
        AggFunc::Count => e.count().cast(DataType::Int64),
        AggFunc::Sum => e.sum(),
        AggFunc::Avg => e.mean(),
        AggFunc::Min => e.min(),
        AggFunc::Max => e.max(),
        // as in Cypher, `collect` skips nulls
        AggFunc::Collect => e.drop_nulls().implode(true),
        AggFunc::CountStar => unreachable!(),
    };
    Ok(out.alias(alias))
}

/// Does `expr` contain a top-level equality comparison (through `AND`)? Used to detect whether a
/// variable has an equality predicate for the join-order heuristic.
fn has_equality(expr: &Expr) -> bool {
    match expr {
        Expr::Binary {
            op: BinaryOp::Eq, ..
        } => true,
        Expr::Binary {
            op: BinaryOp::And,
            lhs,
            rhs,
            ..
        } => has_equality(lhs) || has_equality(rhs),
        _ => false,
    }
}

/// Estimate a node variable's scan size from the `manifest`, for the join-order heuristic. An
/// unlabeled variable is estimated as the sum of every node table (it scans all of them).
fn estimated_row_count(manifest: &crate::layout::Manifest, labels: &[String]) -> u64 {
    if labels.is_empty() {
        return manifest.nodes.iter().map(|t| t.row_count).sum();
    }
    labels
        .iter()
        .filter_map(|label| manifest.node_table(label).map(|t| t.row_count))
        .min()
        .unwrap_or(u64::MAX)
}

fn agg_func(name: &str) -> Option<AggFunc> {
    Some(match name.to_ascii_lowercase().as_str() {
        "count" => AggFunc::Count,
        "sum" => AggFunc::Sum,
        "avg" => AggFunc::Avg,
        "min" => AggFunc::Min,
        "max" => AggFunc::Max,
        "collect" => AggFunc::Collect,
        _ => return None,
    })
}

/// Structural equality ignoring [`Span`]s. An `ORDER BY` expression and the matching `RETURN`
/// item's expression are parsed separately, so their spans differ even when semantically
/// identical.
fn expr_eq_ignoring_span(a: &Expr, b: &Expr) -> bool {
    use Expr::*;
    match (a, b) {
        (Literal(la, _), Literal(lb, _)) => la == lb,
        (Parameter(na, _), Parameter(nb, _)) => na == nb,
        (Variable(na, _), Variable(nb, _)) => na == nb,
        (
            Property {
                base: ba, name: na, ..
            },
            Property {
                base: bb, name: nb, ..
            },
        ) => na == nb && expr_eq_ignoring_span(ba, bb),
        (Not(ia, _), Not(ib, _)) | (Neg(ia, _), Neg(ib, _)) => expr_eq_ignoring_span(ia, ib),
        (
            Binary {
                op: oa,
                lhs: la,
                rhs: ra,
                ..
            },
            Binary {
                op: ob,
                lhs: lb,
                rhs: rb,
                ..
            },
        ) => oa == ob && expr_eq_ignoring_span(la, lb) && expr_eq_ignoring_span(ra, rb),
        (
            IsNull {
                expr: ea,
                negated: na,
                ..
            },
            IsNull {
                expr: eb,
                negated: nb,
                ..
            },
        ) => na == nb && expr_eq_ignoring_span(ea, eb),
        (
            InList {
                expr: ea, list: la, ..
            },
            InList {
                expr: eb, list: lb, ..
            },
        ) => expr_eq_ignoring_span(ea, eb) && expr_eq_ignoring_span(la, lb),
        (
            StringMatch {
                op: oa,
                expr: ea,
                pattern: pa,
                ..
            },
            StringMatch {
                op: ob,
                expr: eb,
                pattern: pb,
                ..
            },
        ) => oa == ob && expr_eq_ignoring_span(ea, eb) && expr_eq_ignoring_span(pa, pb),
        (List(ia, _), List(ib, _)) => {
            ia.len() == ib.len() && ia.iter().zip(ib).all(|(x, y)| expr_eq_ignoring_span(x, y))
        }
        (
            FunctionCall {
                name: na,
                args: aa,
                distinct: da,
                ..
            },
            FunctionCall {
                name: nb,
                args: ab,
                distinct: db,
                ..
            },
        ) => {
            na == nb
                && da == db
                && aa.len() == ab.len()
                && aa.iter().zip(ab).all(|(x, y)| expr_eq_ignoring_span(x, y))
        }
        (CountStar(_), CountStar(_)) => true,
        _ => false,
    }
}

fn rename_node_columns(lf: LazyFrame, var: &str) -> LazyFrame {
    lf.select([col("*").name().prefix(&format!("{var}."))])
}

/// Scan the relationship `types` (all types if empty) for `var`, tagging each row with its source
/// type in a `var._type` column, ready to be joined on the [`SRC_COL`] and [`DST_COL`] columns.
fn scan_rel_types(graph: &Graph, var: &str, types: &[String]) -> Result<LazyFrame, Error> {
    let selected: Vec<String> = if types.is_empty() {
        graph.manifest().rel_types().map(str::to_string).collect()
    } else {
        types.to_vec()
    };
    if selected.is_empty() {
        return Err(Error::new("graph has no relationship tables"));
    }
    let frames = selected
        .iter()
        .map(|ty| {
            let lf = graph.scan_rel_table(ty)?.select([
                col(SRC_COL),
                col(DST_COL),
                lit(ty.clone()).alias(format!("{var}.{TYPE_COL}")),
                col(ID_COL).alias(format!("{var}.{ID_COL}")),
                all()
                    .exclude_cols([ID_COL, SRC_COL, DST_COL])
                    .as_expr()
                    .name()
                    .prefix(&format!("{var}.")),
            ]);
            Ok(lf)
        })
        .collect::<Result<Vec<_>, Error>>()?;
    if frames.len() == 1 {
        return Ok(frames.into_iter().next().unwrap());
    }
    concat(
        frames,
        UnionArgs {
            diagonal: true,
            ..Default::default()
        },
    )
    .map_err(|e| Error::wrap("failed to concat relationship tables", e))
}

/// `JoinArgs` for a join whose left/right key columns have different names. Inner joins coalesce
/// (silently drop the right-side key, renaming nothing) by default when the names differ, which
/// would break every later reference to that column. `one_direction` and the
/// `ExpandTarget::New` join both join on differently named keys, and rely on *both* surviving
/// under their own names.
fn keep_columns(how: JoinType) -> JoinArgs {
    JoinArgs {
        coalesce: JoinCoalesce::KeepColumns,
        ..JoinArgs::new(how)
    }
}

type PolarsExpr = polars::prelude::Expr;

fn lower_literal(lit_val: &Literal) -> PolarsExpr {
    match lit_val {
        Literal::Int(n) => lit(*n),
        Literal::Float(n) => lit(*n),
        Literal::Str(s) => lit(s.clone()),
        Literal::Bool(b) => lit(*b),
        Literal::Null => lit(NULL),
    }
}

fn lower_param(name: &str, span: Span, ctx: &LowerCtx) -> Result<PolarsExpr, Error> {
    let value = ctx
        .params
        .get(name)
        .ok_or_else(|| ctx.unsupported(span, format!("missing parameter `${name}`")))?;
    Ok(match value {
        BoundParameter::Int(n) => lit(*n),
        BoundParameter::Float(n) => lit(*n),
        BoundParameter::Str(s) => lit(s.clone()),
        BoundParameter::Bool(b) => lit(*b),
        BoundParameter::Null => lit(NULL),
    })
}

/// Cypher's remainder, which has the dividend's sign, where Polars' has the divisor's.
fn remainder(l: PolarsExpr, r: PolarsExpr) -> PolarsExpr {
    let rem = l.clone() % r.clone();
    let signs_differ = l.lt(lit(0)).xor(r.clone().lt(lit(0)));
    when(rem.clone().neq(lit(0)).and(signs_differ))
        .then(rem.clone() - r)
        .otherwise(rem)
}

/// Whether `expr` is statically known to evaluate to an integer.
fn is_integer(expr: &Expr, ctx: &LowerCtx) -> bool {
    match expr {
        Expr::Literal(Literal::Int(_), _) => true,
        Expr::Parameter(name, _) => matches!(ctx.params.get(name), Some(BoundParameter::Int(_))),
        Expr::Property { .. } => property_dtype(expr, ctx) == Some(PropertyType::Int64),
        Expr::Neg(inner, _) => is_integer(inner, ctx),
        Expr::Binary {
            op: BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div | BinaryOp::Mod,
            lhs,
            rhs,
            ..
        } => is_integer(lhs, ctx) && is_integer(rhs, ctx),
        Expr::FunctionCall { name, args, .. } => match name.to_ascii_lowercase().as_str() {
            "id" | "size" => true,
            "abs" => args.first().is_some_and(|arg| is_integer(arg, ctx)),
            _ => false,
        },
        _ => false,
    }
}

fn lower_binary(op: BinaryOp, l: PolarsExpr, r: PolarsExpr) -> PolarsExpr {
    match op {
        BinaryOp::Add => l + r,
        BinaryOp::Sub => l - r,
        BinaryOp::Mul => l * r,
        BinaryOp::Div => l / r,
        BinaryOp::Mod => remainder(l, r),
        BinaryOp::Eq => l.eq(r),
        BinaryOp::Ne => l.neq(r),
        BinaryOp::Lt => l.lt(r),
        BinaryOp::Le => l.lt_eq(r),
        BinaryOp::Gt => l.gt(r),
        BinaryOp::Ge => l.gt_eq(r),
        BinaryOp::And => l.and(r),
        BinaryOp::Or => l.or(r),
        BinaryOp::Xor => l.xor(r),
    }
}

fn lower_function(
    name: &str,
    args: &[Expr],
    span: Span,
    ctx: &LowerCtx,
) -> Result<PolarsExpr, Error> {
    match name.to_ascii_lowercase().as_str() {
        "id" => {
            let [Expr::Variable(var, _)] = args else {
                return Err(ctx.unsupported(span, "id() requires a single variable argument"));
            };
            Ok(col(format!("{var}.{ID_COL}")))
        }
        "coalesce" => {
            if args.is_empty() {
                return Err(ctx.unsupported(span, "coalesce() requires at least one argument"));
            }
            let mut it = args.iter();
            let first = lower_expr(it.next().unwrap(), ctx)?;
            it.try_fold(first, |acc, a| Ok(acc.fill_null(lower_expr(a, ctx)?)))
        }
        "toupper" => Ok(lower_expr(&args_one(args, span, ctx)?, ctx)?
            .str()
            .to_uppercase()),
        "tolower" => Ok(lower_expr(&args_one(args, span, ctx)?, ctx)?
            .str()
            .to_lowercase()),
        "abs" => Ok(lower_expr(&args_one(args, span, ctx)?, ctx)?.abs()),
        "labels" => match args {
            [Expr::Variable(var, _)] if ctx.var_labels.contains_key(var) => {
                Ok(col(format!("{var}.{LABELS_COL}")))
            }
            _ => Err(ctx.unsupported(span, "labels() requires a node variable")),
        },
        "type" => match args {
            [Expr::Variable(var, _)] if ctx.var_rel_types.contains_key(var) => {
                Ok(col(format!("{var}.{TYPE_COL}")))
            }
            _ => Err(ctx.unsupported(span, "type() requires a relationship variable")),
        },
        "size" => lower_size(args, span, ctx),
        other => Err(ctx.unsupported(span, format!("function `{other}`"))),
    }
}

fn lower_entity(var: &str, kind: EntityKind, ctx: &LowerCtx) -> Result<PolarsExpr, Error> {
    let mut fields = vec![col(format!("{var}.{ID_COL}")).alias(ID_COL)];
    fields.push(match kind {
        EntityKind::Node => col(format!("{var}.{LABELS_COL}")).alias(LABELS_COL),
        EntityKind::Relationship => col(format!("{var}.{TYPE_COL}")).alias(TYPE_COL),
    });
    fields.extend(
        ctx.properties(var)
            .into_iter()
            .map(|p| col(format!("{var}.{}", p.name)).alias(p.name.clone())),
    );
    Ok(as_struct(fields))
}

fn args_one(args: &[Expr], span: Span, ctx: &LowerCtx) -> Result<Expr, Error> {
    match args {
        [only] => Ok(only.clone()),
        _ => Err(ctx.unsupported(span, "expected exactly one argument")),
    }
}

fn lower_size(args: &[Expr], span: Span, ctx: &LowerCtx) -> Result<PolarsExpr, Error> {
    let arg = args_one(args, span, ctx)?;
    let dtype = property_dtype(&arg, ctx);
    let e = lower_expr(&arg, ctx)?;
    let len = match dtype {
        Some(ty) if ty.inner().is_some() => e.list().len(),
        _ => e.str().len_chars(),
    };
    // Cypher's `size` returns an INTEGER
    Ok(len.cast(DataType::Int64))
}

fn property_dtype(expr: &Expr, ctx: &LowerCtx) -> Option<PropertyType> {
    let Expr::Property { base, name, .. } = expr else {
        return None;
    };
    let Expr::Variable(var, _) = base.as_ref() else {
        return None;
    };
    ctx.property(var, name).map(|p| p.dtype)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::OnceLock;

    use chrono::NaiveDate;
    use polars::prelude::*;

    use crate::layout::Graph;
    use crate::{BoundParameter, GraphWriter, Properties, Value};

    #[test]
    fn return_entity() {
        let graph = movies();
        let df = run(graph, "MATCH (m:Movie {title: 'The Matrix'}) RETURN m");
        assert_eq!(df.height(), 1);
        let s = df.column("m").unwrap().struct_().unwrap();
        let title = s.field_by_name("title").unwrap();
        let title: &str = title.str().unwrap().get(0).unwrap();
        assert_eq!(title, "The Matrix");
        let labels = s.field_by_name("_labels").unwrap();
        assert_eq!(first_list(&labels.into_column()), vec!["Movie"]);
    }

    #[test]
    fn return_properties() {
        let graph = movies();
        let df = run(
            graph,
            "MATCH (p:Person) WHERE p.name = 'Keanu Reeves' RETURN p.name, p.born",
        );
        assert_eq!(df.height(), 1);
        let born: i64 = df.column("p.born").unwrap().i64().unwrap().get(0).unwrap();
        assert_eq!(born, 1964);
    }

    #[test]
    fn relationship_pattern_and_count() {
        let graph = movies();
        let df = run(
            graph,
            "MATCH (p:Person)-[:ACTED_IN]->(m:Movie {title: 'The Matrix'}) \
             RETURN p.name ORDER BY p.name",
        );
        let names: Vec<&str> = df
            .column("p.name")
            .unwrap()
            .str()
            .unwrap()
            .iter()
            .flatten()
            .collect();
        assert_eq!(
            names,
            vec![
                "Carrie-Anne Moss",
                "Emil Eifrem",
                "Hugo Weaving",
                "Keanu Reeves",
                "Laurence Fishburne"
            ]
        );
    }

    #[test]
    fn count_aggregation_with_implicit_grouping() {
        let graph = movies();
        let df = run(
            graph,
            "MATCH (p:Person)-[:DIRECTED]->(m:Movie) \
             RETURN p.name, count(*) AS c ORDER BY c DESC, p.name",
        );
        let names: Vec<&str> = df
            .column("p.name")
            .unwrap()
            .str()
            .unwrap()
            .iter()
            .flatten()
            .collect();
        let counts: Vec<i64> = df
            .column("c")
            .unwrap()
            .i64()
            .unwrap()
            .iter()
            .flatten()
            .collect();
        assert_eq!(
            names,
            vec!["Lana Wachowski", "Lilly Wachowski", "Taylor Hackford"]
        );
        assert_eq!(counts, vec![2, 2, 1]);
    }

    #[test]
    fn parameterized_where() {
        let graph = movies();
        let mut params = HashMap::new();
        params.insert("year".to_string(), BoundParameter::Int(1965));
        let df = run_with_params(
            graph,
            "MATCH (p:Person) WHERE p.born < $year RETURN count(*) AS c",
            params,
        );
        let c: i64 = df.column("c").unwrap().i64().unwrap().get(0).unwrap();
        assert!(c > 0, "expected at least one person born before 1965");
    }

    #[test]
    fn undirected_pattern() {
        let graph = movies();
        // DIRECTED only goes Person -> Movie; an undirected match from the Movie side should
        // still find directors
        let df = run(
            graph,
            "MATCH (m:Movie {title: 'The Matrix'})-[:DIRECTED]-(p:Person) \
             RETURN p.name ORDER BY p.name",
        );
        let names: Vec<&str> = df
            .column("p.name")
            .unwrap()
            .str()
            .unwrap()
            .iter()
            .flatten()
            .collect();
        assert_eq!(names, vec!["Lana Wachowski", "Lilly Wachowski"]);
    }

    #[test]
    fn distinct_order_skip_limit() {
        let graph = movies();
        let df = run(
            graph,
            "MATCH (p:Person)-[:ACTED_IN]->(m:Movie) \
             RETURN DISTINCT p.name AS name ORDER BY name SKIP 1 LIMIT 2",
        );
        assert_eq!(df.height(), 2);
    }

    #[test]
    fn multi_label_pattern() {
        let graph = movies();
        let df = run(
            graph,
            "MATCH (a:Actor:Artist) RETURN a.name ORDER BY a.name",
        );
        let names: Vec<&str> = df
            .column("a.name")
            .unwrap()
            .str()
            .unwrap()
            .iter()
            .flatten()
            .collect();
        assert_eq!(names, vec!["Ada", "Bo"]);
    }

    #[test]
    fn self_loop_pattern() {
        let graph = movies();
        let df = run(graph, "MATCH (a:Actor)-[:KNOWS]->(a) RETURN a.name");
        assert_eq!(df.height(), 1);
        let name: &str = df.column("a.name").unwrap().str().unwrap().get(0).unwrap();
        assert_eq!(name, "Ada");
    }

    #[test]
    fn functions_id_toupper_size_abs_coalesce() {
        let graph = movies();
        let df = run(
            graph,
            "MATCH (p:Artist) WHERE p.name = 'Ada' \
             RETURN id(p), toUpper(p.name), size(p.name), abs(-3), coalesce(p.score, 0.0)",
        );
        assert_eq!(df.height(), 1);
        let upper: &str = df
            .column("toUpper(p.name)")
            .unwrap()
            .str()
            .unwrap()
            .get(0)
            .unwrap();
        assert_eq!(upper, "ADA");
    }

    #[test]
    fn unknown_label_is_an_error() {
        let err = plan_error("MATCH (x:NoSuchLabel) RETURN x");
        assert!(err.contains("NoSuchLabel"), "message was: {err}");
    }

    #[test]
    fn unknown_variable_is_an_error() {
        let err = plan_error("MATCH (p:Person) WHERE x.name = 'Ada' RETURN p.name");
        assert!(err.contains("unknown variable `x`"), "message was: {err}");
    }

    #[test]
    fn default_column_names_are_the_item_text() {
        let df = run(
            movies(),
            "MATCH (a:Artist {name: 'Ada'}) RETURN a.name + 'x', coalesce(a.score, 1.0), (1+2)",
        );
        let names: Vec<&str> = df
            .get_column_names()
            .into_iter()
            .map(|n| n.as_str())
            .collect();
        assert_eq!(
            names,
            vec!["a.name + 'x'", "coalesce(a.score, 1.0)", "(1+2)"]
        );
    }

    #[test]
    fn count_and_collect_entities() {
        let df = run(
            movies(),
            "MATCH (p:Person)-[:DIRECTED]->(m:Movie) \
             RETURN m.title, count(p) AS c, collect(p) AS directors ORDER BY c DESC, m.title",
        );
        assert_eq!(ints(&df, "c"), vec![2, 2, 1]);
        let directors = df.column("directors").unwrap().list().unwrap();
        let first = directors.get_as_series(0).unwrap();
        let names = first.struct_().unwrap().field_by_name("name").unwrap();
        let mut names: Vec<&str> = names.str().unwrap().iter().flatten().collect();
        names.sort();
        assert_eq!(names, vec!["Lana Wachowski", "Lilly Wachowski"]);
    }

    #[test]
    fn order_by_aggregate() {
        let df = run(
            movies(),
            "MATCH (p:Person)-[:DIRECTED]->(:Movie) \
             RETURN p.name, count(*) ORDER BY count(*), p.name LIMIT 1",
        );
        assert_eq!(strings(&df, "p.name"), vec!["Taylor Hackford"]);
    }

    #[test]
    fn order_by_expression_that_is_not_returned() {
        let df = run(
            movies(),
            "MATCH (p:Person) RETURN p.name ORDER BY p.born LIMIT 2",
        );
        assert_eq!(df.get_column_names(), vec!["p.name"]);
        assert_eq!(strings(&df, "p.name"), vec!["Al Pacino", "Taylor Hackford"]);
        let err = plan_error("MATCH (p:Person) RETURN DISTINCT p.name ORDER BY p.born");
        assert!(err.contains("DISTINCT"), "message was: {err}");
    }

    #[test]
    fn skip_and_limit_parameters() {
        let params = HashMap::from([
            ("skip".to_string(), BoundParameter::Int(1)),
            ("limit".to_string(), BoundParameter::Int(2)),
        ]);
        let df = run_with_params(
            movies(),
            "MATCH (m:Movie) RETURN m.title ORDER BY m.title SKIP $skip LIMIT $limit",
            params,
        );
        assert_eq!(
            strings(&df, "m.title"),
            vec!["The Matrix", "The Matrix Reloaded"]
        );
        let err = plan_error("MATCH (m:Movie) RETURN m.title SKIP -1");
        assert!(err.contains("non-negative"), "message was: {err}");
    }

    #[test]
    fn string_match_parameter() {
        let params = HashMap::from([("prefix".to_string(), BoundParameter::from("Lan"))]);
        let df = run_with_params(
            movies(),
            "MATCH (p:Person) WHERE p.name STARTS WITH $prefix RETURN p.name",
            params,
        );
        assert_eq!(strings(&df, "p.name"), vec!["Lana Wachowski"]);
    }

    #[test]
    fn lowering_errors_point_at_the_query_text() {
        let err = plan_error("MATCH (p:Person) RETURN nope(p.name)");
        assert!(
            err.contains("unsupported: function `nope`"),
            "message was: {err}"
        );
        assert!(err.contains("--> line 1:25"), "message was: {err}");
    }

    #[test]
    fn unlabeled_nodes() {
        let df = run(
            movies(),
            "MATCH (:Actor {name: 'Bo'})-[:KNOWS]->(n) RETURN n.name, labels(n) AS labels",
        );
        assert_eq!(strings(&df, "n.name"), vec!["Stranger"]);
        assert!(first_list(df.column("labels").unwrap()).is_empty());
    }

    #[test]
    fn undirected_self_loop_matches_once() {
        let df = run(
            movies(),
            "MATCH (a:Actor)-[:KNOWS]-(a) RETURN count(*) AS c",
        );
        assert_eq!(ints(&df, "c"), vec![1]);
    }

    #[test]
    fn collect_skips_nulls() {
        let df = run(movies(), "MATCH (a:Artist) RETURN collect(a.tags) AS tags");
        let tags = df.column("tags").unwrap().list().unwrap();
        assert_eq!(tags.get_as_series(0).unwrap().len(), 1);
    }

    #[test]
    fn disconnected_patterns() {
        let df = run(
            movies(),
            "MATCH (s:Studio), (r:Release) RETURN s.name, r.title",
        );
        assert_eq!(strings(&df, "r.title"), vec!["Film A"]);
        let df = run(movies(), "MATCH (m:Movie), (s:Studio) RETURN count(*) AS c");
        assert_eq!(ints(&df, "c"), vec![3]);
    }

    #[test]
    fn anonymous_relationship_uniqueness() {
        // each of the five actors pairs with the four others, but never with themselves
        let df = run(
            movies(),
            "MATCH (:Person)-[:ACTED_IN]->(:Movie {title: 'The Matrix'})<-[:ACTED_IN]-(:Person) \
             RETURN count(*) AS c",
        );
        assert_eq!(ints(&df, "c"), vec![20]);
        let df = run(
            movies(),
            "MATCH (:Actor)-[:WORKED_WITH {project: 'Film A'}]->(:Actor) RETURN count(*) AS c",
        );
        assert_eq!(ints(&df, "c"), vec![1]);
    }

    #[test]
    fn missing_property_is_null() {
        let df = run(
            movies(),
            "MATCH (m:Movie) WHERE m.born IS NULL RETURN count(*) AS c",
        );
        assert_eq!(ints(&df, "c"), vec![3]);
    }

    #[test]
    fn labels_are_the_full_label_set() {
        let df = run(
            movies(),
            "MATCH (a:Actor {name: 'Ada'}) RETURN labels(a) AS labels, a",
        );
        assert_eq!(
            first_list(df.column("labels").unwrap()),
            vec!["Actor", "Artist"]
        );
        let a = df.column("a").unwrap().struct_().unwrap();
        let entity_labels = a.field_by_name("_labels").unwrap().into_column();
        assert_eq!(first_list(&entity_labels), vec!["Actor", "Artist"]);
        let df = run(
            movies(),
            "MATCH (n {name: 'Cy'}) RETURN labels(n) AS labels",
        );
        assert_eq!(
            first_list(df.column("labels").unwrap()),
            vec!["Artist", "Director"]
        );
    }

    #[test]
    fn size_of_relationship_list_property() {
        let df = run(
            movies(),
            "MATCH (:Person {name: 'Keanu Reeves'})-[r:ACTED_IN]->(:Movie {title: 'The Matrix'}) \
             RETURN size(r.roles) AS c",
        );
        assert_eq!(ints(&df, "c"), vec![1]);
    }

    #[test]
    fn integer_division_and_remainder_truncate() {
        let df = run(
            movies(),
            "MATCH (s:Studio) RETURN -7 / 2 AS q, -7 % 3 AS a, 7 % -3 AS b, 7 % 3 AS c, \
             -7.5 % 2 AS d, -7.0 / 2 AS f",
        );
        for (name, expected) in [("q", -3), ("a", -1), ("b", 1), ("c", 1)] {
            assert_eq!(ints(&df, name), vec![expected], "column {name}");
        }
        assert_eq!(df.column("d").unwrap().f64().unwrap().get(0), Some(-1.5));
        assert_eq!(df.column("f").unwrap().f64().unwrap().get(0), Some(-3.5));
    }

    #[test]
    fn order_by_sorts_nulls() {
        let cypher = "MATCH (n) WHERE n.name IN ['Bo', 'Cy', 'Stranger'] RETURN n.name \
                      ORDER BY n.score";
        let df = run(movies(), cypher);
        assert_eq!(strings(&df, "n.name"), vec!["Cy", "Bo", "Stranger"]);
        let df = run(movies(), &format!("{cypher} DESC"));
        assert_eq!(strings(&df, "n.name"), vec!["Stranger", "Bo", "Cy"]);
    }

    #[test]
    fn inline_property_map_referencing_variable() {
        // only Ada's self-loop joins two nodes with the same name
        let df = run(
            movies(),
            "MATCH (a:Actor)-[:KNOWS]->(b {name: a.name}) RETURN b.name",
        );
        assert_eq!(strings(&df, "b.name"), vec!["Ada"]);
        let err = plan_error("MATCH (a:Actor {name: zz.name}) RETURN a");
        assert!(err.contains("unknown variable `zz`"), "{err}");
    }

    #[test]
    fn rebinding_variable() {
        for (cypher, expected) in [
            ("MATCH (a)-[a]->(b) RETURN b", "`a` is already bound"),
            (
                "MATCH (a)-[r]->(r) RETURN a",
                "`r` is already bound to a relationship",
            ),
            (
                "MATCH (a)-[r]->(b), (b)-[r]->(c) RETURN a",
                "`r` is already bound",
            ),
        ] {
            let err = plan_error(cypher);
            assert!(err.contains(expected), "{cypher}: {err}");
        }
    }

    #[test]
    fn keywords_as_labels() {
        let err = plan_error("MATCH (o:Order) RETURN o");
        assert!(err.contains("unknown label `Order`"), "{err}");
    }

    /// A subset of `tests/fixtures/movies.cypher`, written with a [`GraphWriter`] then read back.
    fn movies() -> &'static Graph {
        static GRAPH: OnceLock<Graph> = OnceLock::new();
        GRAPH.get_or_init(|| {
            let out_dir = std::env::temp_dir()
                .join(format!("polars-cypher-plan-test-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&out_dir);
            let mut writer = GraphWriter::create(&out_dir).unwrap();
            add_movies(&mut writer);
            writer.write().unwrap();
            Graph::open(&out_dir).unwrap_or_else(|e| panic!("open failed: {e}"))
        })
    }

    fn add_movies(w: &mut GraphWriter) {
        let mut person = |name: &str, born: i64| {
            let properties = props([("name", string(name)), ("born", Value::Int(born))]);
            w.add_node(["Person"], properties).unwrap()
        };
        let keanu = person("Keanu Reeves", 1964);
        let carrie = person("Carrie-Anne Moss", 1967);
        let laurence = person("Laurence Fishburne", 1961);
        let hugo = person("Hugo Weaving", 1960);
        let lilly = person("Lilly Wachowski", 1967);
        let lana = person("Lana Wachowski", 1965);
        let joel = person("Joel Silver", 1952);
        let emil = person("Emil Eifrem", 1978);
        let taylor = person("Taylor Hackford", 1944);
        let charlize = person("Charlize Theron", 1975);
        let al = person("Al Pacino", 1940);

        let mut movie = |title: &str, released: i64, tagline: &str| {
            let properties = props([
                ("title", string(title)),
                ("released", Value::Int(released)),
                ("tagline", string(tagline)),
            ]);
            w.add_node(["Movie"], properties).unwrap()
        };
        let matrix = movie("The Matrix", 1999, "Welcome to the Real World");
        let reloaded = movie("The Matrix Reloaded", 2003, "Free your mind");
        let devils = movie("The Devil's Advocate", 1997, "Evil has its winning ways");

        let mut acted_in = |actor, movie, role: &str| {
            let roles = Value::List(vec![string(role)]);
            w.add_relationship(actor, movie, "ACTED_IN", props([("roles", roles)]))
                .unwrap()
        };
        acted_in(keanu, matrix, "Neo");
        acted_in(carrie, matrix, "Trinity");
        acted_in(laurence, matrix, "Morpheus");
        acted_in(hugo, matrix, "Agent Smith");
        acted_in(emil, matrix, "Emil");
        acted_in(keanu, reloaded, "Neo");
        acted_in(carrie, reloaded, "Trinity");
        acted_in(laurence, reloaded, "Morpheus");
        acted_in(hugo, reloaded, "Agent Smith");
        acted_in(keanu, devils, "Kevin Lomax");
        acted_in(charlize, devils, "Mary Ann Lomax");
        acted_in(al, devils, "John Milton");

        for (src, dst, rel_type) in [
            (lilly, matrix, "DIRECTED"),
            (lana, matrix, "DIRECTED"),
            (lilly, reloaded, "DIRECTED"),
            (lana, reloaded, "DIRECTED"),
            (taylor, devils, "DIRECTED"),
            (joel, matrix, "PRODUCED"),
            (joel, reloaded, "PRODUCED"),
        ] {
            w.add_relationship(src, dst, rel_type, Properties::new())
                .unwrap();
        }

        // edge cases, labeled `Artist` rather than `Person` and given properties the movies don't
        // have, since property types are unified by name across the whole graph
        let artist = |w: &mut GraphWriter, labels: [&str; 2], properties| {
            w.add_node(labels, properties).unwrap()
        };
        // `score` mixes floats with an integer, so it's widened to float
        let tags = Value::List(vec![string("lead"), string("stunt")]);
        let ada = artist(
            w,
            ["Actor", "Artist"],
            props([
                ("name", string("Ada")),
                ("score", Value::Float(4.5)),
                ("tags", tags),
            ]),
        );
        let bo = artist(
            w,
            ["Actor", "Artist"],
            props([("name", string("Bo")), ("score", Value::Int(3))]),
        );
        let cy = artist(
            w,
            ["Director", "Artist"],
            props([("name", string("Cy")), ("score", Value::Float(2.5))]),
        );
        let founded = NaiveDate::from_ymd_opt(1995, 6, 15)
            .unwrap()
            .and_hms_opt(0, 0, 0)
            .unwrap();
        let studio = w
            .add_node(
                ["Studio"],
                props([
                    ("name", string("Big Studio")),
                    (
                        "founded",
                        Value::Datetime {
                            micros: founded.and_utc().timestamp_micros(),
                            utc: true,
                        },
                    ),
                    (
                        "location",
                        string("point({x: -0.12, y: 51.5, crs: 'wgs-84'})"),
                    ),
                ]),
            )
            .unwrap();
        let opened = NaiveDate::from_ymd_opt(2001, 2, 3).unwrap();
        let premiere = opened.and_hms_opt(19, 30, 0).unwrap();
        let epoch = NaiveDate::from_ymd_opt(1970, 1, 1).unwrap();
        let ratings = Value::List([4, 5, 3].map(Value::Int).to_vec());
        w.add_node(
            ["Release"],
            props([
                ("title", string("Film A")),
                ("opened", Value::Date((opened - epoch).num_days() as i32)),
                (
                    "premiere",
                    Value::Datetime {
                        micros: premiere.and_utc().timestamp_micros(),
                        utc: false,
                    },
                ),
                ("runtime", string("PT1H45M")),
                ("ratings", ratings),
            ]),
        )
        .unwrap();

        w.add_relationship(ada, ada, "KNOWS", props([("since", Value::Int(2010))]))
            .unwrap();
        for project in ["Film A", "Film B"] {
            let properties = props([("project", string(project))]);
            w.add_relationship(ada, bo, "WORKED_WITH", properties)
                .unwrap();
        }
        w.add_relationship(cy, studio, "WORKS_FOR", Properties::new())
            .unwrap();
        let stranger = w
            .add_node(Vec::<&str>::new(), props([("name", string("Stranger"))]))
            .unwrap();
        w.add_relationship(bo, stranger, "KNOWS", Properties::new())
            .unwrap();
    }

    fn run(graph: &Graph, cypher: &str) -> DataFrame {
        run_with_params(graph, cypher, HashMap::new())
    }

    fn run_with_params(
        graph: &Graph,
        cypher: &str,
        params: HashMap<String, BoundParameter>,
    ) -> DataFrame {
        let (_, lf) = super::plan_query(graph, cypher, &params)
            .unwrap_or_else(|e| panic!("plan failed for `{cypher}`: {e}"));
        lf.collect()
            .unwrap_or_else(|e| panic!("collect failed for `{cypher}`: {e}"))
    }

    fn plan_error(cypher: &str) -> String {
        match super::plan_query(movies(), cypher, &HashMap::new()) {
            Ok(_) => panic!("`{cypher}` should fail to plan"),
            Err(e) => e.to_string(),
        }
    }

    fn strings(df: &DataFrame, name: &str) -> Vec<String> {
        let column = df.column(name).unwrap().str().unwrap();
        column.iter().flatten().map(str::to_string).collect()
    }

    fn ints(df: &DataFrame, name: &str) -> Vec<i64> {
        let column = df.column(name).unwrap().cast(&DataType::Int64).unwrap();
        column.i64().unwrap().iter().flatten().collect()
    }

    /// The strings in the first row of a list column.
    fn first_list(column: &Column) -> Vec<String> {
        let first = column.list().unwrap().get_as_series(0).unwrap();
        first
            .str()
            .unwrap()
            .iter()
            .flatten()
            .map(str::to_string)
            .collect()
    }

    fn props<const N: usize>(props: [(&str, Value); N]) -> Properties {
        props.into_iter().map(|(k, v)| (k.to_string(), v)).collect()
    }

    fn string(s: &str) -> Value {
        Value::String(s.to_string())
    }
}
