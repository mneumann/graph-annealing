RUSTFMT=rustfmt

build:
	cargo build --release

run-ga: build
	time cargo run --release --bin ga

run-nsga2: build
	time cargo run --release --bin nsga2

run-nsga2-edge: build
	time cargo run --release --bin nsga2_edge -- --ngen 10000

run-nsga2-edge-ops: build
	time cargo run --release --bin nsga2_edge_ops -- --ngen 10000 --n 100

reformat:
	find . -name "*.rs" | xargs -I % ${RUSTFMT} %
