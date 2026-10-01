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

//! Parser and AST for the supported *Cypher* subset, built with [nom](https://docs.rs/nom)
//! parser combinators.

use std::cmp::Ordering;
use std::error::Error;
use std::fmt;
use std::iter;

use nom::branch::alt;
use nom::bytes::complete::{tag, take_till, take_until, take_while, take_while1};
use nom::character::complete::{char, digit1, multispace1, one_of, satisfy};
use nom::combinator::{cut, eof, opt, recognize, value, verify};
use nom::error::ErrorKind;
use nom::multi::{many0, many0_count, separated_list1};
use nom::sequence::{delimited, pair, preceded, separated_pair, terminated};
use nom::{IResult, Input as _, Parser};
use nom_locate::LocatedSpan;

/// Parse a *Cypher* query from `source`.
///
/// Returns a [`ParseError`] pointing at the offending text if `source` isn't _valid_ *Cypher*.
pub fn parse(source: &str) -> Result<Query, ParseError> {
    match query(Text::new(source)) {
        Ok((_, query)) => Ok(query),
        Err(nom::Err::Error(failure) | nom::Err::Failure(failure)) => Err(failure.into()),
        Err(nom::Err::Incomplete(_)) => unreachable!("parser is complete"),
    }
}

/// A supported *Cypher* query.
///
/// Includes comma-separated `MATCH` patterns, an optional `WHERE`, and a `RETURN`.
#[derive(Debug, Clone, PartialEq)]
pub struct Query {
    pub patterns: Vec<Pattern>,
    pub where_clause: Option<Expr>,
    pub return_clause: ReturnClause,
    pub span: Span,
}

/// A lexing or parsing failure at a specific location in the query text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseError {
    pub message: String,
    pub span: Span,
}

impl ParseError {
    /// Create a [`ParseError`] with `message` at `span`.
    pub fn new(message: impl Into<String>, span: Span) -> Self {
        Self {
            message: message.into(),
            span,
        }
    }
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} (at {}..{})",
            self.message, self.span.start, self.span.end
        )
    }
}

impl Error for ParseError {}

impl From<Failure> for ParseError {
    fn from(failure: Failure) -> Self {
        let message = failure
            .message
            .unwrap_or_else(|| match failure.expected.as_slice() {
                [] => "invalid syntax".to_string(),
                expected => format!("expected {}", expected.join(" or ")),
            });
        ParseError::new(message, failure.span)
    }
}

/// A comma-separated pattern in a `MATCH` clause.
///
/// Describes a chain of nodes connected by relationships.
#[derive(Debug, Clone, PartialEq)]
pub struct Pattern {
    pub elements: Vec<PatternElement>,
    pub span: Span,
}

/// An expression, e.g. in a `WHERE` clause, [`ReturnItem`], or [`MapLiteral`] value.
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    /// A literal value, e.g. `42`.
    Literal(Literal, Span),
    /// A query parameter, e.g. `$year`.
    Parameter(String, Span),
    /// A variable bound in a pattern, e.g. `p`.
    Variable(String, Span),
    /// `<base>.<name>`, e.g. `p.name`.
    Property {
        base: Box<Expr>,
        name: String,
        span: Span,
    },
    /// Boolean negation, e.g. `NOT x`.
    Not(Box<Expr>, Span),
    /// Arithmetic negation, e.g. `-x`.
    Neg(Box<Expr>, Span),
    /// A binary operation, e.g. `a + b` or `a AND b`.
    Binary {
        op: BinaryOp,
        lhs: Box<Expr>,
        rhs: Box<Expr>,
        span: Span,
    },
    /// A null check, e.g. `x IS NULL` or `x IS NOT NULL`.
    IsNull {
        expr: Box<Expr>,
        negated: bool,
        span: Span,
    },
    /// A list membership test, e.g. `x IN [1, 2]`.
    InList {
        expr: Box<Expr>,
        list: Box<Expr>,
        span: Span,
    },
    /// A string predicate, e.g. `name STARTS WITH 'A'`.
    StringMatch {
        op: StringMatchOp,
        expr: Box<Expr>,
        pattern: Box<Expr>,
        span: Span,
    },
    /// A list literal, e.g. `[1, 2, 3]`.
    List(Vec<Expr>, Span),
    /// A function call, e.g. `toUpper(p.name)`.
    FunctionCall {
        name: String,
        args: Vec<Expr>,
        distinct: bool,
        span: Span,
    },
    /// Specifically, a `count(*)` function call.
    CountStar(Span),
}

impl Expr {
    /// The [`Span`] of this expression in the query text.
    pub fn span(&self) -> Span {
        match self {
            Expr::Literal(_, span)
            | Expr::Parameter(_, span)
            | Expr::Variable(_, span)
            | Expr::Property { span, .. }
            | Expr::Not(_, span)
            | Expr::Neg(_, span)
            | Expr::Binary { span, .. }
            | Expr::IsNull { span, .. }
            | Expr::InList { span, .. }
            | Expr::StringMatch { span, .. }
            | Expr::List(_, span)
            | Expr::FunctionCall { span, .. }
            | Expr::CountStar(span) => *span,
        }
    }

    /// Call `f` on this expression, then on each of its subexpressions.
    pub fn visit(&self, f: &mut impl FnMut(&Expr)) {
        f(self);
        match self {
            Expr::Property { base: inner, .. }
            | Expr::Not(inner, _)
            | Expr::Neg(inner, _)
            | Expr::IsNull { expr: inner, .. } => inner.visit(f),
            Expr::Binary { lhs, rhs, .. }
            | Expr::InList {
                expr: lhs,
                list: rhs,
                ..
            }
            | Expr::StringMatch {
                expr: lhs,
                pattern: rhs,
                ..
            } => {
                lhs.visit(f);
                rhs.visit(f);
            }
            Expr::List(items, _) | Expr::FunctionCall { args: items, .. } => {
                items.iter().for_each(|item| item.visit(f));
            }
            Expr::Literal(..) | Expr::Parameter(..) | Expr::Variable(..) | Expr::CountStar(_) => {}
        }
    }
}

/// A `RETURN` clause with its optional `DISTINCT`, `ORDER BY`, `SKIP`, and/or `LIMIT`.
#[derive(Debug, Clone, PartialEq)]
pub struct ReturnClause {
    pub distinct: bool,
    pub items: Vec<ReturnItem>,
    pub order_by: Vec<OrderItem>,
    pub skip: Option<Expr>,
    pub limit: Option<Expr>,
    pub span: Span,
}

/// A half-open byte range `[start, end)` into the original query text. Every AST node maintains
/// this, so binder and planner errors can point at the offending construct.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

impl Span {
    /// Create a [`Span`] covering `start..end`.
    pub fn new(start: usize, end: usize) -> Self {
        Self { start, end }
    }

    /// The smallest span containing both `self` and `other`.
    pub fn to(self, other: Span) -> Span {
        Span {
            start: self.start.min(other.start),
            end: self.end.max(other.end),
        }
    }
}

/// A node or relationship in a [`Pattern`].
#[derive(Debug, Clone, PartialEq)]
pub enum PatternElement {
    /// A node, e.g. `(p:Person)`.
    Node(NodePattern),
    /// A relationship, e.g. `-[:ACTED_IN]->`.
    Relationship(RelPattern),
}

/// A literal value, e.g. `42`, `'Neo'`, or `null`.
#[derive(Debug, Clone, PartialEq)]
pub enum Literal {
    /// An integer, e.g. `42`.
    Int(i64),
    /// A float, e.g. `3.14` or `1e3`.
    Float(f64),
    /// A string, e.g. `'Neo'`.
    Str(String),
    /// `true` or `false`.
    Bool(bool),
    /// `null`.
    Null,
}

/// An arithmetic, comparison, or boolean infix operator in an [`Expr::Binary`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    /// `+`.
    Add,
    /// `-`.
    Sub,
    /// `*`.
    Mul,
    /// `/`.
    Div,
    /// `%`.
    Mod,
    /// `=`.
    Eq,
    /// `<>`.
    Ne,
    /// `<`.
    Lt,
    /// `<=`.
    Le,
    /// `>`.
    Gt,
    /// `>=`.
    Ge,
    /// `AND`.
    And,
    /// `OR`.
    Or,
    /// `XOR`.
    Xor,
}

/// A string predicate operator in an [`Expr::StringMatch`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StringMatchOp {
    /// `STARTS WITH`.
    StartsWith,
    /// `ENDS WITH`.
    EndsWith,
    /// `CONTAINS`.
    Contains,
}

/// A single `RETURN` projection, e.g. `p.name AS name`.
#[derive(Debug, Clone, PartialEq)]
pub struct ReturnItem {
    pub expr: Expr,
    pub alias: Option<String>,
    pub span: Span,
}

/// A single `ORDER BY` sort key, e.g. `p.name DESC`.
#[derive(Debug, Clone, PartialEq)]
pub struct OrderItem {
    pub expr: Expr,
    pub ascending: bool,
    pub span: Span,
}

/// A node in a [`Pattern`], e.g. `(p:Person {name: 'Keanu'})`.
#[derive(Debug, Clone, PartialEq)]
pub struct NodePattern {
    pub variable: Option<String>,
    pub labels: Vec<String>,
    pub properties: Option<MapLiteral>,
    pub span: Span,
}

/// A relationship in a [`Pattern`], e.g. `-[r:ACTED_IN]->`.
#[derive(Debug, Clone, PartialEq)]
pub struct RelPattern {
    pub variable: Option<String>,
    pub types: Vec<String>,
    pub direction: Direction,
    pub properties: Option<MapLiteral>,
    pub span: Span,
}

/// An inline property map, e.g. `{name: 'Keanu', born: 1964}`.
#[derive(Debug, Clone, PartialEq)]
pub struct MapLiteral {
    pub entries: Vec<(String, Expr)>,
    pub span: Span,
}

/// The direction of a [`RelPattern`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    /// `<-[...]-`.
    Left,
    /// `-[...]->`.
    Right,
    /// `-[...]-`.
    Undirected,
}

/// Render `message` with the line of `source` that `span` points into, and a caret under it.
pub fn render(message: &str, span: Span, source: &str) -> String {
    let line_start = source[..span.start].rfind('\n').map_or(0, |i| i + 1);
    let line_end = source[line_start..]
        .find('\n')
        .map_or(source.len(), |i| line_start + i);
    let line_no = source[..line_start].matches('\n').count() + 1;
    let col = span.start - line_start;
    let caret = " ".repeat(col) + &"^".repeat(span.end.saturating_sub(span.start).max(1));
    format!(
        "{message}\n  --> line {line_no}:{}\n{}\n{caret}",
        col + 1,
        &source[line_start..line_end]
    )
}

type Text<'a> = LocatedSpan<&'a str>;
type PResult<'a, T> = IResult<Text<'a>, T, Failure>;

/// Parse the *Cypher* query `text`.
///
/// Every parser skips its own leading whitespace and comments (see [`ws`]), so the input left
/// after a successful parse starts right after the last consumed token and spans end exactly
/// where their construct does. Expression precedence is climbed with one function per
/// precedence level, from [`expr`] (loosest) down to [`postfix`] (tightest).
fn query(text: Text) -> PResult<Query> {
    let span = Span::new(0, text.fragment().len());
    let (rest, (patterns, where_clause, return_clause)) = delimited(
        keyword("MATCH"),
        (
            comma_list1(pattern),
            opt(preceded(keyword("WHERE"), cut(expr))),
            return_clause,
        ),
        expect("end of query", eof),
    )
    .parse(text)?;
    let query = Query {
        patterns,
        where_clause,
        return_clause,
        span,
    };
    Ok((rest, query))
}

/// `RETURN [DISTINCT] items [ORDER BY keys] [SKIP n] [LIMIT n]`.
fn return_clause(i: Text) -> PResult<ReturnClause> {
    let order_by = preceded(
        (keyword("ORDER"), cut(keyword("BY"))),
        cut(comma_list1(order_item)),
    );
    let body = (
        opt(keyword("DISTINCT")).map(|distinct| distinct.is_some()),
        comma_list1(return_item),
        opt(order_by).map(Option::unwrap_or_default),
        opt(preceded(keyword("SKIP"), cut(expr))),
        opt(preceded(keyword("LIMIT"), cut(expr))),
    );
    let (rest, ((distinct, items, order_by, skip, limit), span)) =
        spanned(preceded(keyword("RETURN"), cut(body))).parse(i)?;
    let clause = ReturnClause {
        distinct,
        items,
        order_by,
        skip,
        limit,
        span,
    };
    Ok((rest, clause))
}

/// A case-insensitive keyword, e.g. `MATCH`, that isn't merely the prefix of a longer word.
fn keyword<'a>(kw: &'static str) -> impl Parser<Text<'a>, Output = (), Error = Failure> {
    let word = verify(take_while1(is_ident_continue), move |w: &Text| {
        w.eq_ignore_ascii_case(kw)
    });
    labeled(move || format!("`{kw}`"), value((), word))
}

/// One or more comma-separated `item`s.
fn comma_list1<'a, O>(
    item: fn(Text<'a>) -> PResult<'a, O>,
) -> impl Parser<Text<'a>, Output = Vec<O>, Error = Failure> {
    separated_list1(sym(","), cut(item))
}

/// A node followed by any number of relationship-node steps, e.g. `(a)-[:R]->(b)<-[:S]-(c)`.
fn pattern(i: Text) -> PResult<Pattern> {
    let steps = many0(pair(rel_pattern, cut(node_pattern)));
    let (rest, ((first, steps), span)) = spanned(pair(node_pattern, steps)).parse(i)?;
    let mut elements = vec![PatternElement::Node(first)];
    for (rel, node) in steps {
        elements.push(PatternElement::Relationship(rel));
        elements.push(PatternElement::Node(node));
    }
    Ok((rest, Pattern { elements, span }))
}

/// An expression, starting at the loosest precedence level: `OR`.
fn expr(i: Text) -> PResult<Expr> {
    infix_chain(
        i,
        xor_expr,
        value(Infix::Binary(BinaryOp::Or), keyword("OR")),
    )
}

/// Run `parser`, reporting a failure at its first character as expecting `what`.
fn expect<'a, O>(
    what: &'static str,
    parser: impl Parser<Text<'a>, Output = O, Error = Failure>,
) -> impl Parser<Text<'a>, Output = O, Error = Failure> {
    labeled(move || what.to_string(), parser)
}

/// The byte offset of `input` into the original query text.
fn offset(input: Text) -> usize {
    input.location_offset()
}

/// A sort key with an optional direction, e.g. `p.name DESC`, ascending by default.
fn order_item(i: Text) -> PResult<OrderItem> {
    let direction = alt((
        value(true, alt((keyword("ASCENDING"), keyword("ASC")))),
        value(false, alt((keyword("DESCENDING"), keyword("DESC")))),
    ));
    let (rest, ((expr, ascending), span)) = spanned(pair(expr, opt(direction))).parse(i)?;
    let item = OrderItem {
        expr,
        ascending: ascending.unwrap_or(true),
        span,
    };
    Ok((rest, item))
}

/// A projection with an optional alias, e.g. `p.name AS name`.
fn return_item(i: Text) -> PResult<ReturnItem> {
    let alias = opt(preceded(keyword("AS"), cut(identifier)));
    let (rest, ((expr, alias), span)) = spanned(pair(expr, alias)).parse(i)?;
    Ok((rest, ReturnItem { expr, alias, span }))
}

/// Skip whitespace, then run `parser`, also returning the span of what it consumed.
fn spanned<'a, O>(
    mut parser: impl Parser<Text<'a>, Output = O, Error = Failure>,
) -> impl Parser<Text<'a>, Output = (O, Span), Error = Failure> {
    move |i: Text<'a>| -> PResult<'a, (O, Span)> {
        let (i, ()) = ws(i)?;
        let (rest, output) = parser.parse(i)?;
        Ok((rest, (output, Span::new(offset(i), offset(rest)))))
    }
}

/// Whether `c` can appear after the first character of an identifier or keyword.
fn is_ident_continue(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_'
}

/// Skip whitespace, then run `parser`, reporting a failure at its very first character as
/// expecting `label()`.
fn labeled<'a, O>(
    label: impl Fn() -> String,
    mut parser: impl Parser<Text<'a>, Output = O, Error = Failure>,
) -> impl Parser<Text<'a>, Output = O, Error = Failure> {
    move |i: Text<'a>| -> PResult<'a, O> {
        let (i, ()) = ws(i)?;
        parser.parse(i).map_err(|e| {
            e.map(|failure| {
                if failure.message.is_none() && failure.span.start == offset(i) {
                    Failure::at(i, vec![label()])
                } else {
                    failure
                }
            })
        })
    }
}

/// A punctuation token, e.g. `(` or `->`.
fn sym<'a>(token: &'static str) -> impl Parser<Text<'a>, Output = Text<'a>, Error = Failure> {
    labeled(move || format!("`{token}`"), tag(token))
}

/// A relationship between two nodes, e.g. `-[r:KNOWS]->`, `<--`, or `-[]-`.
fn rel_pattern(i: Text) -> PResult<RelPattern> {
    let (rest, ((left, detail, right), span)) = spanned((
        alt((value(true, sym("<-")), value(false, sym("-")))),
        opt(rel_detail),
        cut(alt((value(true, sym("->")), value(false, sym("-"))))),
    ))
    .parse(i)?;
    let direction = match (left, right) {
        (true, false) => Direction::Left,
        (false, true) => Direction::Right,
        (false, false) => Direction::Undirected,
        (true, true) => return fail("a relationship cannot point in both directions", span),
    };
    let (variable, types, properties) = detail.unwrap_or_default();
    let rel = RelPattern {
        variable,
        types,
        direction,
        properties,
        span,
    };
    Ok((rest, rel))
}

/// A parenthesized node, e.g. `(p:Person:Actor {name: 'Keanu'})`.
fn node_pattern(i: Text) -> PResult<NodePattern> {
    let labels = many0(preceded(sym(":"), cut(name)));
    let body = (opt(identifier), labels, opt(map_literal));
    let (rest, ((variable, labels, properties), span)) =
        spanned(delimited(sym("("), cut(body), cut(sym(")")))).parse(i)?;
    let node = NodePattern {
        variable,
        labels,
        properties,
        span,
    };
    Ok((rest, node))
}

/// A left-associative chain of `operand`s joined by the infix operators `op` recognizes.
fn infix_chain<'a>(
    i: Text<'a>,
    operand: fn(Text<'a>) -> PResult<'a, Expr>,
    op: impl Parser<Text<'a>, Output = Infix, Error = Failure>,
) -> PResult<'a, Expr> {
    let (rest, (first, chain)) = pair(operand, many0(pair(op, cut(operand)))).parse(i)?;
    let expr = chain
        .into_iter()
        .fold(first, |lhs, (op, rhs)| op.apply(lhs, rhs));
    Ok((rest, expr))
}

/// `XOR`-joined operands, binding tighter than `OR`.
fn xor_expr(i: Text) -> PResult<Expr> {
    infix_chain(
        i,
        and_expr,
        value(Infix::Binary(BinaryOp::Xor), keyword("XOR")),
    )
}

/// An infix operator recognized by [`infix_chain`], before it's applied to its operands.
#[derive(Clone, Copy)]
enum Infix {
    Binary(BinaryOp),
    In,
    StringMatch(StringMatchOp),
}

impl Infix {
    /// Build the [`Expr`] for `lhs <op> rhs`, spanning both operands.
    fn apply(self, lhs: Expr, rhs: Expr) -> Expr {
        let span = lhs.span().to(rhs.span());
        let (lhs, rhs) = (Box::new(lhs), Box::new(rhs));
        match self {
            Infix::Binary(op) => Expr::Binary { op, lhs, rhs, span },
            Infix::In => Expr::InList {
                expr: lhs,
                list: rhs,
                span,
            },
            Infix::StringMatch(op) => Expr::StringMatch {
                op,
                expr: lhs,
                pattern: rhs,
                span,
            },
        }
    }
}

/// A plain identifier that isn't a reserved keyword, or any backtick-quoted text.
fn identifier(i: Text) -> PResult<String> {
    symbolic_name(i, false)
}

/// A label, relationship type, or property key, which unlike an [`identifier`] may be a reserved
/// keyword, e.g. the label in `(o:Order)`, since its position is unambiguous.
fn name(i: Text) -> PResult<String> {
    symbolic_name(i, true)
}

/// A plain identifier, a reserved keyword only if `allow_keywords`, or any backtick-quoted text.
fn symbolic_name(i: Text, allow_keywords: bool) -> PResult<String> {
    let quoted = delimited(char('`'), take_till(|c| c == '`'), char('`'));
    let plain = verify(
        recognize((
            satisfy(|c| c.is_ascii_alphabetic() || c == '_'),
            take_while(is_ident_continue),
        )),
        |w: &Text| allow_keywords || !RESERVED.iter().any(|kw| w.eq_ignore_ascii_case(kw)),
    );
    expect("an identifier", alt((quoted, plain)))
        .map(|name: Text| name.fragment().to_string())
        .parse(i)
}

/// Skip whitespace, `// line` comments, and `/* block */` comments.
fn ws(i: Text) -> PResult<()> {
    let line_comment = recognize((tag("//"), take_till(|c| c == '\n')));
    let block_comment = recognize((tag("/*"), take_until("*/"), tag("*/")));
    value(
        (),
        many0_count(alt((multispace1, line_comment, block_comment))),
    )
    .parse(i)
}

/// The bracketed part of a relationship, e.g. `[r:ACTED_IN|DIRECTED {role: 'Neo'}]`.
fn rel_detail(i: Text) -> PResult<(Option<String>, Vec<String>, Option<MapLiteral>)> {
    let rel_type = preceded(opt(sym(":")), name);
    let types = preceded(sym(":"), cut(separated_list1(sym("|"), cut(rel_type))));
    let body = (
        opt(identifier),
        opt(types).map(Option::unwrap_or_default),
        opt(map_literal),
    );
    delimited(sym("["), cut(body), cut(sym("]"))).parse(i)
}

/// Abort parsing with `message` at `span`, without backtracking into other alternatives.
fn fail<T>(message: impl Into<String>, span: Span) -> Result<T, nom::Err<Failure>> {
    Err(nom::Err::Failure(Failure::message(message, span)))
}

/// A braced property map, e.g. `{name: 'Keanu', born: 1964}`.
fn map_literal(i: Text) -> PResult<MapLiteral> {
    let entries = delimited(sym("{"), comma_list0(map_entry), cut(sym("}")));
    let (rest, (entries, span)) = spanned(entries).parse(i)?;
    Ok((rest, MapLiteral { entries, span }))
}

/// `AND`-joined operands, binding tighter than `XOR`.
fn and_expr(i: Text) -> PResult<Expr> {
    infix_chain(
        i,
        not_expr,
        value(Infix::Binary(BinaryOp::And), keyword("AND")),
    )
}

/// Keywords that can't be used as plain (unquoted) identifiers.
const RESERVED: [&str; 26] = [
    "MATCH",
    "WHERE",
    "RETURN",
    "AND",
    "OR",
    "NOT",
    "XOR",
    "IS",
    "NULL",
    "IN",
    "STARTS",
    "ENDS",
    "WITH",
    "CONTAINS",
    "DISTINCT",
    "AS",
    "ORDER",
    "BY",
    "ASC",
    "ASCENDING",
    "DESC",
    "DESCENDING",
    "SKIP",
    "LIMIT",
    "TRUE",
    "FALSE",
];

/// Zero or more comma-separated `item`s, committing to an item after each comma.
fn comma_list0<'a, O>(
    item: fn(Text<'a>) -> PResult<'a, O>,
) -> impl Parser<Text<'a>, Output = Vec<O>, Error = Failure> {
    opt(pair(item, many0(preceded(sym(","), cut(item))))).map(|items| {
        items
            .map(|(first, rest)| iter::once(first).chain(rest).collect())
            .unwrap_or_default()
    })
}

/// A `key: value` entry in a [`MapLiteral`].
fn map_entry(i: Text) -> PResult<(String, Expr)> {
    separated_pair(name, cut(sym(":")), cut(expr)).parse(i)
}

/// Any number of `NOT` prefixes applied to a comparison, binding tighter than `AND`.
fn not_expr(i: Text) -> PResult<Expr> {
    let not = prefixed(keyword("NOT"), not_expr, Expr::Not);
    expect("an expression", alt((not, comparison))).parse(i)
}

/// A prefix operator applied to `operand`, e.g. `NOT x` or `-x`.
fn prefixed<'a>(
    op: impl Parser<Text<'a>, Error = Failure>,
    operand: fn(Text<'a>) -> PResult<'a, Expr>,
    build: fn(Box<Expr>, Span) -> Expr,
) -> impl Parser<Text<'a>, Output = Expr, Error = Failure> {
    pair(spanned(op), cut(operand)).map(move |((_, op_span), operand)| {
        let span = op_span.to(operand.span());
        build(Box::new(operand), span)
    })
}

/// Comparison, `IN`, and string predicate operators, binding tighter than `NOT`.
fn comparison(i: Text) -> PResult<Expr> {
    let op = |op| Infix::Binary(op);
    let string_match = |op| Infix::StringMatch(op);
    let ops = alt((
        value(op(BinaryOp::Ne), sym("<>")),
        value(op(BinaryOp::Le), sym("<=")),
        value(op(BinaryOp::Ge), sym(">=")),
        value(op(BinaryOp::Eq), sym("=")),
        value(op(BinaryOp::Lt), sym("<")),
        value(op(BinaryOp::Gt), sym(">")),
        value(Infix::In, keyword("IN")),
        value(
            string_match(StringMatchOp::StartsWith),
            (keyword("STARTS"), cut(keyword("WITH"))),
        ),
        value(
            string_match(StringMatchOp::EndsWith),
            (keyword("ENDS"), cut(keyword("WITH"))),
        ),
        value(string_match(StringMatchOp::Contains), keyword("CONTAINS")),
    ));
    infix_chain(i, additive, ops)
}

/// `+` and `-` operators, binding tighter than comparisons.
fn additive(i: Text) -> PResult<Expr> {
    let ops = alt((
        value(Infix::Binary(BinaryOp::Add), sym("+")),
        value(Infix::Binary(BinaryOp::Sub), sym("-")),
    ));
    infix_chain(i, multiplicative, ops)
}

/// `*`, `/`, and `%` operators, binding tighter than `+` and `-`.
fn multiplicative(i: Text) -> PResult<Expr> {
    let ops = alt((
        value(Infix::Binary(BinaryOp::Mul), sym("*")),
        value(Infix::Binary(BinaryOp::Div), sym("/")),
        value(Infix::Binary(BinaryOp::Mod), sym("%")),
    ));
    infix_chain(i, negation, ops)
}

/// Any number of unary `-` prefixes applied to a [`postfix`] expression.
fn negation(i: Text) -> PResult<Expr> {
    let neg = prefixed(sym("-"), negation, Expr::Neg);
    expect("an expression", alt((neg, postfix))).parse(i)
}

/// An atom followed by any number of `.name` property accesses and `IS [NOT] NULL` checks.
fn postfix(i: Text) -> PResult<Expr> {
    enum Postfix {
        Property(String, Span),
        IsNull(bool, Span),
    }
    let property =
        preceded(sym("."), cut(spanned(name))).map(|(name, span)| Postfix::Property(name, span));
    let is_null = spanned(preceded(
        keyword("IS"),
        cut(pair(opt(keyword("NOT")), keyword("NULL"))),
    ))
    .map(|((negated, ()), span)| Postfix::IsNull(negated.is_some(), span));
    let (rest, (atom, ops)) = pair(atom, many0(alt((property, is_null)))).parse(i)?;
    let expr = ops.into_iter().fold(atom, |base, op| match op {
        Postfix::Property(name, name_span) => Expr::Property {
            span: base.span().to(name_span),
            base: Box::new(base),
            name,
        },
        Postfix::IsNull(negated, op_span) => Expr::IsNull {
            span: base.span().to(op_span),
            expr: Box::new(base),
            negated,
        },
    });
    Ok((rest, expr))
}

/// A self-contained expression: a literal, parameter, function call, variable, list, or
/// parenthesized expression.
fn atom(i: Text) -> PResult<Expr> {
    let literal = spanned(alt((
        number,
        string.map(Literal::Str),
        value(Literal::Bool(true), keyword("TRUE")),
        value(Literal::Bool(false), keyword("FALSE")),
        value(Literal::Null, keyword("NULL")),
    )))
    .map(|(literal, span)| Expr::Literal(literal, span));
    let parameter = spanned(preceded(char('$'), take_while1(is_ident_continue)))
        .map(|(name, span): (Text, _)| Expr::Parameter(name.fragment().to_string(), span));
    let variable = spanned(identifier).map(|(name, span)| Expr::Variable(name, span));
    let list = spanned(delimited(sym("["), comma_list0(expr), cut(sym("]"))))
        .map(|(items, span)| Expr::List(items, span));
    let parenthesized = delimited(sym("("), cut(expr), cut(sym(")")));
    alt((
        literal,
        parameter,
        function_call,
        variable,
        list,
        parenthesized,
    ))
    .parse(i)
}

/// An integer, or a float if it has a fraction or exponent, e.g. `42`, `3.14`, or `1e-3`.
fn number(i: Text) -> PResult<Literal> {
    let (rest, text) = recognize((
        digit1,
        opt((char('.'), digit1)),
        opt((one_of("eE"), opt(one_of("+-")), digit1)),
    ))
    .parse(i)?;
    let text = *text.fragment();
    if text.contains(['.', 'e', 'E']) {
        let value = text.parse().expect("the grammar only matches valid floats");
        return Ok((rest, Literal::Float(value)));
    }
    match text.parse() {
        Ok(value) => Ok((rest, Literal::Int(value))),
        Err(_) => fail(
            format!("invalid integer literal `{text}`"),
            Span::new(offset(i), offset(rest)),
        ),
    }
}

/// A single- or double-quoted string, with its escape sequences decoded.
fn string(i: Text) -> PResult<String> {
    let text = *i.fragment();
    let quote = match text.chars().next() {
        Some(quote @ ('\'' | '"')) => quote,
        _ => return Err(nom::Err::Error(Failure::at(i, Vec::new()))),
    };
    let start = offset(i);
    let mut value = String::new();
    let mut chars = text.char_indices().skip(1);
    while let Some((idx, c)) = chars.next() {
        if c == quote {
            return Ok((i.take_from(idx + 1), value));
        }
        if c != '\\' {
            value.push(c);
            continue;
        }
        let Some((_, escaped)) = chars.next() else {
            break;
        };
        // `len` is the escape sequence's length in bytes, including its backslash
        let invalid = |len: usize| {
            fail(
                format!("invalid escape sequence `{}`", &text[idx..idx + len]),
                Span::new(start + idx, start + idx + len),
            )
        };
        value.push(match escaped {
            'n' => '\n',
            't' => '\t',
            'r' => '\r',
            'b' => '\u{8}',
            'f' => '\u{c}',
            '\\' | '\'' | '"' => escaped,
            'u' | 'U' => {
                let digits = if escaped == 'u' { 4 } else { 8 };
                let hex: String = chars.by_ref().take(digits).map(|(_, c)| c).collect();
                let code = (hex.len() == digits && hex.chars().all(|c| c.is_ascii_hexdigit()))
                    .then(|| u32::from_str_radix(&hex, 16).ok().and_then(char::from_u32))
                    .flatten();
                match code {
                    Some(c) => c,
                    None => return invalid(2 + hex.len()),
                }
            }
            other => return invalid(1 + other.len_utf8()),
        });
    }
    fail("unterminated string literal", Span::new(start, start + 1))
}

/// A function call, e.g. `toUpper(p.name)` or `count(DISTINCT p)`, including `count(*)`.
fn function_call(i: Text) -> PResult<Expr> {
    enum Args {
        Star(Span),
        Exprs(bool, Vec<Expr>),
    }
    let args = alt((
        spanned(sym("*")).map(|(_, span)| Args::Star(span)),
        pair(opt(keyword("DISTINCT")), comma_list0(expr))
            .map(|(distinct, args)| Args::Exprs(distinct.is_some(), args)),
    ));
    let call = pair(
        identifier,
        preceded(sym("("), cut(terminated(args, sym(")")))),
    );
    let (rest, ((name, args), span)) = spanned(call).parse(i)?;
    match args {
        Args::Exprs(distinct, args) => {
            let call = Expr::FunctionCall {
                name,
                args,
                distinct,
                span,
            };
            Ok((rest, call))
        }
        Args::Star(_) if name.eq_ignore_ascii_case("count") => Ok((rest, Expr::CountStar(span))),
        Args::Star(star_span) => fail(
            format!("`{name}(*)` is not supported (only `count(*)`)"),
            star_span,
        ),
    }
}

/// A failure before it becomes a [`ParseError`]: either what was expected at a position, or a
/// specific message.
#[derive(Debug)]
struct Failure {
    span: Span,
    expected: Vec<String>,
    message: Option<String>,
}

impl Failure {
    /// A failure at the start of `input`, where one of `expected` was expected.
    fn at(input: Text, expected: Vec<String>) -> Self {
        let offset = input.location_offset();
        Self {
            span: Span::new(offset, offset),
            expected,
            message: None,
        }
    }

    /// A failure with a specific `message` at `span`.
    fn message(message: impl Into<String>, span: Span) -> Self {
        Self {
            span,
            expected: Vec::new(),
            message: Some(message.into()),
        }
    }
}

impl nom::error::ParseError<Text<'_>> for Failure {
    fn from_error_kind(input: Text, _: ErrorKind) -> Self {
        Self::at(input, Vec::new())
    }

    fn append(_: Text, _: ErrorKind, other: Self) -> Self {
        other
    }

    /// Keep whichever alternative got furthest, merging what was expected on a tie.
    fn or(mut self, other: Self) -> Self {
        match self.span.start.cmp(&other.span.start) {
            Ordering::Greater => self,
            Ordering::Less => other,
            Ordering::Equal if self.message.is_some() => self,
            Ordering::Equal if other.message.is_some() => other,
            Ordering::Equal => {
                for expected in other.expected {
                    if !self.expected.contains(&expected) {
                        self.expected.push(expected);
                    }
                }
                self
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::fmt::Write as _;

    use super::*;

    #[test]
    fn match_return() {
        parse_test("MATCH (p:Person) RETURN p.name");
    }

    #[test]
    fn relationship_pattern() {
        parse_test("MATCH (p:Person)-[:ACTED_IN]->(m:Movie) RETURN p.name, m.title");
    }

    #[test]
    fn undirected_and_left() {
        parse_test("MATCH (a)-[r:KNOWS]-(b) RETURN r");
        parse_test("MATCH (a)<-[:DIRECTED]-(b) RETURN a, b");
    }

    #[test]
    fn multi_label_and_multi_type() {
        parse_test("MATCH (a:Actor:Person)-[:ACTED_IN|DIRECTED]->(m:Movie) RETURN a");
    }

    #[test]
    fn where_clause() {
        parse_test(
            "MATCH (p:Person) WHERE p.born < 1970 AND NOT p.name STARTS WITH 'A' RETURN p.name",
        );
    }

    #[test]
    fn comma_patterns() {
        parse_test("MATCH (a:Person), (b:Person) WHERE a.name <> b.name RETURN a.name, b.name");
    }

    #[test]
    fn return_distinct_order_skip_limit() {
        parse_test(
            "MATCH (p:Person) RETURN DISTINCT p.name AS name ORDER BY name DESC SKIP 5 LIMIT 10",
        );
    }

    #[test]
    fn aggregations() {
        parse_test("MATCH (p:Person) RETURN p.name, count(*) AS c");
        parse_test("MATCH (p:Person)-[:ACTED_IN]->(m:Movie) RETURN p.name, collect(m.title)");
    }

    #[test]
    fn in_and_is_null() {
        parse_test("MATCH (p:Person) WHERE p.born IN [1960, 1970] AND p.died IS NULL RETURN p");
    }

    #[test]
    fn functions_and_params() {
        parse_test("MATCH (p:Person) WHERE p.born < $year RETURN toUpper(p.name), abs(p.born)");
    }

    #[test]
    fn property_map() {
        parse_test("MATCH (p:Person {name: 'Keanu Reeves'}) RETURN p");
    }

    #[test]
    fn arithmetic_precedence() {
        parse_test("MATCH (p:Person) WHERE p.born + 1 * 2 > 1970 RETURN p.name");
    }

    #[test]
    fn subtraction_and_unary_negation() {
        // `-` is shared with relationship patterns, so it must parse as subtraction or
        // negation in expression position
        parse_test("MATCH (p:Person) WHERE p.born - 1 > -1970 RETURN abs(-p.born)");
    }

    #[test]
    fn keywords_as_labels_types_and_property_keys() {
        parse_test("MATCH (o:Order {limit: 1})-[:IN|BY]->(b) RETURN o.desc, b.count");
        let err = parse("MATCH (order) RETURN order").unwrap_err();
        assert_eq!(err.span.start, 7);
    }

    #[test]
    fn string_escapes() {
        let query = parse(r#"MATCH (n) RETURN 'a\'b\"c\u00e9\U0001F600\b\f'"#).unwrap();
        let Expr::Literal(Literal::Str(s), _) = &query.return_clause.items[0].expr else {
            panic!("expected a string literal");
        };
        assert_eq!(s, "a'b\"c\u{e9}\u{1F600}\u{8}\u{c}");
        let err = parse(r"MATCH (n) RETURN 'a\u00zz'").unwrap_err();
        assert_eq!(err.message, r"invalid escape sequence `\u00zz`");
        assert_eq!(err.span, Span::new(19, 25));
    }

    #[test]
    fn error_span_on_unknown_token() {
        let err = parse("MATCH (p:Person RETURN p").unwrap_err();
        // missing closing paren before RETURN
        assert_eq!(err.span.start, 16);
    }

    #[test]
    fn error_span_on_unterminated_string() {
        let err = parse("MATCH (p:Person {name: 'oops}) RETURN p").unwrap_err();
        assert_eq!(err.message, "unterminated string literal");
        assert_eq!(err.span, Span::new(23, 24));
    }

    #[test]
    fn error_span_on_missing_return() {
        let err = parse("MATCH (p:Person)").unwrap_err();
        assert!(
            err.message.contains("RETURN"),
            "message was: {}",
            err.message
        );
    }

    /// Parse, pretty-print, then parse and pretty-print again. Spans differ between the two
    /// parses (the printed text isn't the original text), so this compares the two printed forms
    /// rather than the ASTs directly, since structurally equal trees print identically.
    fn parse_test(source: &str) {
        let query = parse(source).unwrap_or_else(|e| panic!("failed to parse `{source}`: {e}"));
        let printed = print_query(&query);
        let reparsed = parse(&printed)
            .unwrap_or_else(|e| panic!("failed to reparse pretty-printed `{printed}`: {e}"));
        let reprinted = print_query(&reparsed);
        assert_eq!(printed, reprinted, "roundtrip mismatch for `{source}`");
    }

    /// Render `query` back to *Cypher* text.
    fn print_query(query: &Query) -> String {
        let mut out = String::new();
        out.push_str("MATCH ");
        for (i, pattern) in query.patterns.iter().enumerate() {
            if i > 0 {
                out.push_str(", ");
            }
            print_pattern(pattern, &mut out);
        }
        if let Some(where_clause) = &query.where_clause {
            out.push_str(" WHERE ");
            print_expr_out(where_clause, &mut out);
        }
        out.push(' ');
        print_return(&query.return_clause, &mut out);
        out
    }

    /// Append the *Cypher* text of `pattern` to `out`.
    fn print_pattern(pattern: &Pattern, out: &mut String) {
        for element in &pattern.elements {
            match element {
                PatternElement::Node(node) => print_node(node, out),
                PatternElement::Relationship(rel) => print_rel(rel, out),
            }
        }
    }

    /// Append the *Cypher* text of `clause` to `out`.
    fn print_return(clause: &ReturnClause, out: &mut String) {
        out.push_str("RETURN ");
        if clause.distinct {
            out.push_str("DISTINCT ");
        }
        for (i, item) in clause.items.iter().enumerate() {
            if i > 0 {
                out.push_str(", ");
            }
            print_expr_out(&item.expr, out);
            if let Some(alias) = &item.alias {
                write!(out, " AS {alias}").unwrap();
            }
        }
        if !clause.order_by.is_empty() {
            out.push_str(" ORDER BY ");
            for (i, item) in clause.order_by.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                print_expr_out(&item.expr, out);
                if !item.ascending {
                    out.push_str(" DESC");
                }
            }
        }
        if let Some(skip) = &clause.skip {
            out.push_str(" SKIP ");
            print_expr_out(skip, out);
        }
        if let Some(limit) = &clause.limit {
            out.push_str(" LIMIT ");
            print_expr_out(limit, out);
        }
    }

    /// Append the *Cypher* text of `node` to `out`.
    fn print_node(node: &NodePattern, out: &mut String) {
        out.push('(');
        if let Some(v) = &node.variable {
            out.push_str(v);
        }
        for label in &node.labels {
            write!(out, ":{label}").unwrap();
        }
        if let Some(props) = &node.properties {
            print_map(props, out);
        }
        out.push(')');
    }

    /// Append the *Cypher* text of `rel` to `out`.
    fn print_rel(rel: &RelPattern, out: &mut String) {
        let has_detail =
            rel.variable.is_some() || !rel.types.is_empty() || rel.properties.is_some();

        if rel.direction == Direction::Left {
            out.push('<');
        }
        out.push('-');
        if has_detail {
            out.push('[');
            if let Some(v) = &rel.variable {
                out.push_str(v);
            }
            for (i, ty) in rel.types.iter().enumerate() {
                out.push_str(if i == 0 { ":" } else { "|" });
                out.push_str(ty);
            }
            if let Some(props) = &rel.properties {
                print_map(props, out);
            }
            out.push(']');
        }
        out.push('-');
        if rel.direction == Direction::Right {
            out.push('>');
        }
    }

    /// Append the *Cypher* text of `map` to `out`.
    fn print_map(map: &MapLiteral, out: &mut String) {
        out.push('{');
        for (i, (key, value)) in map.entries.iter().enumerate() {
            if i > 0 {
                out.push_str(", ");
            }
            write!(out, "{key}: ").unwrap();
            print_expr_out(value, out);
        }
        out.push('}');
    }

    /// Append the *Cypher* text of `expr` to `out`, fully parenthesizing operators so precedence is
    /// explicit.
    fn print_expr_out(expr: &Expr, out: &mut String) {
        match expr {
            Expr::Literal(lit, _) => print_literal(lit, out),
            Expr::Parameter(name, _) => write!(out, "${name}").unwrap(),
            Expr::Variable(name, _) => out.push_str(name),
            Expr::Property { base, name, .. } => {
                print_expr_out(base, out);
                write!(out, ".{name}").unwrap();
            }
            Expr::Not(inner, _) => {
                out.push_str("NOT (");
                print_expr_out(inner, out);
                out.push(')');
            }
            Expr::Neg(inner, _) => {
                out.push_str("-(");
                print_expr_out(inner, out);
                out.push(')');
            }
            Expr::Binary { op, lhs, rhs, .. } => {
                out.push('(');
                print_expr_out(lhs, out);
                write!(out, " {} ", binary_op_str(*op)).unwrap();
                print_expr_out(rhs, out);
                out.push(')');
            }
            Expr::IsNull { expr, negated, .. } => {
                print_expr_out(expr, out);
                out.push_str(if *negated { " IS NOT NULL" } else { " IS NULL" });
            }
            Expr::InList { expr, list, .. } => {
                print_expr_out(expr, out);
                out.push_str(" IN ");
                print_expr_out(list, out);
            }
            Expr::StringMatch {
                op, expr, pattern, ..
            } => {
                print_expr_out(expr, out);
                out.push(' ');
                out.push_str(match op {
                    StringMatchOp::StartsWith => "STARTS WITH",
                    StringMatchOp::EndsWith => "ENDS WITH",
                    StringMatchOp::Contains => "CONTAINS",
                });
                out.push(' ');
                print_expr_out(pattern, out);
            }
            Expr::List(items, _) => {
                out.push('[');
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    print_expr_out(item, out);
                }
                out.push(']');
            }
            Expr::FunctionCall {
                name,
                args,
                distinct,
                ..
            } => {
                write!(out, "{name}(").unwrap();
                if *distinct {
                    out.push_str("DISTINCT ");
                }
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    print_expr_out(arg, out);
                }
                out.push(')');
            }
            Expr::CountStar(_) => out.push_str("count(*)"),
        }
    }

    /// Append the *Cypher* text of `lit` to `out`.
    fn print_literal(lit: &Literal, out: &mut String) {
        match lit {
            Literal::Int(n) => write!(out, "{n}").unwrap(),
            Literal::Float(n) => write!(out, "{n}").unwrap(),
            Literal::Str(s) => write!(out, "'{}'", s.replace('\'', "\\'")).unwrap(),
            Literal::Bool(b) => write!(out, "{b}").unwrap(),
            Literal::Null => out.push_str("null"),
        }
    }

    /// The *Cypher* symbol or keyword for `op`.
    fn binary_op_str(op: BinaryOp) -> &'static str {
        match op {
            BinaryOp::Add => "+",
            BinaryOp::Sub => "-",
            BinaryOp::Mul => "*",
            BinaryOp::Div => "/",
            BinaryOp::Mod => "%",
            BinaryOp::Eq => "=",
            BinaryOp::Ne => "<>",
            BinaryOp::Lt => "<",
            BinaryOp::Le => "<=",
            BinaryOp::Gt => ">",
            BinaryOp::Ge => ">=",
            BinaryOp::And => "AND",
            BinaryOp::Or => "OR",
            BinaryOp::Xor => "XOR",
        }
    }
}
