#!/usr/bin/env bash

if [ $# -gt 0 ]; then
  echo 'Demonstrate polars-cypher: export the Movies graph from Neo4j then query it.'
  echo ''
  echo 'Usage:'
  echo '  ./run.sh'
  exit 0
fi

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "$0")" >/dev/null 2>&1; pwd -P)"
ROOT_DIR="$(cd -- "$SCRIPT_DIR/.." >/dev/null 2>&1; pwd -P)"

function cleanup() {
  cd "$SCRIPT_DIR" && docker compose down -v --remove-orphans
}

trap cleanup EXIT

function cli() {
  cargo run -m "$ROOT_DIR/Cargo.toml" -p polars-cypher-cli -- "$@"
}

cd "$SCRIPT_DIR" || exit 1

echo 'Starting Neo4j'
docker compose up -d --remove-orphans --wait

echo 'Loading the Movies dataset'
docker compose exec -T neo4j cypher-shell -u neo4j -p password -f /init.cypher

echo 'Exporting the graph'
NEO4J_PASSWORD=password cli export bolt://localhost:7687 "$SCRIPT_DIR/graph"

echo ''
echo 'Schema:'
cli schema "$SCRIPT_DIR/graph"

echo ''
echo 'Who acted in The Matrix?'
cli query "$SCRIPT_DIR/graph" \
  "MATCH (p:Person)-[:ACTED_IN]->(m:Movie {title: 'The Matrix'}) RETURN p.name ORDER BY p.name"

echo ''
echo 'Who has directed the most movies?'
cli query "$SCRIPT_DIR/graph" \
  "MATCH (p:Person)-[:DIRECTED]->(m:Movie) RETURN p.name, count(*) AS movies \
  ORDER BY movies DESC LIMIT 5"
