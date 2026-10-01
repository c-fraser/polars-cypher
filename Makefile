.PHONY: default help all build check clean msrv test

default: help

help: ## show this help
	@echo 'usage: make [target]'
	@echo ''
	@echo 'targets:'
	@egrep '^(.+)\:\ .*##\ (.+)' ${MAKEFILE_LIST} | sed 's/:.*##/#/' | column -t -c 2 -s '#'

all: check build test ## check, build, and test all code

build: clean ## build rust release binaries
	cargo build --release

# in CI, verify the formatting and license headers instead of fixing them
check: ## check, format, and lint rust code
	cargo check --workspace --all-features
	cargo fmt --all $(if $(CI),-- --check)
	cargo clippy --workspace --all-targets --all-features -- -D warnings
	cargo clippy -p polars-cypher --all-targets -- -D warnings
	RUSTDOCFLAGS="-D warnings" cargo doc -p polars-cypher --no-deps --all-features
	docker run --rm -v $(CURDIR):/src -w /src ghcr.io/google/addlicense $(if $(CI),-check) \
		-c c-fraser -l apache -y 2026 polars-cypher/src polars-cypher/tests polars-cypher-cli/src

msrv: ## check that the libraries and CLI build with the declared minimum Rust version
	$(eval MSRV := $(shell cargo metadata --no-deps --format-version 1 | jq -r '.packages[0].rust_version'))
	rustup toolchain install $(MSRV) --profile minimal --no-self-update
	cargo +$(MSRV) check --workspace --locked

clean: ## remove build files
	cargo clean

test: ## run rust tests
	cargo test --workspace --all-features
