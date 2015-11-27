RUST_PATH=/scratch/rust-env

build: build-ga build-nsga2 build-nsga2-edge build-nsga2-edge-ops

format:
	${RUST_PATH}/bin/rustfmt src/lib.rs
	${RUST_PATH}/bin/rustfmt examples/ga.rs
	${RUST_PATH}/bin/rustfmt examples/nsga2.rs
	${RUST_PATH}/bin/rustfmt examples/nsga2_edge.rs
	${RUST_PATH}/bin/rustfmt examples/nsga2_edge_ops.rs

build-ga:
	LD_LIBRARY_PATH=${RUST_PATH}/lib PATH=${RUST_PATH}/bin:${PATH} ${RUST_PATH}/bin/cargo build --release --example ga

build-nsga2:
	LD_LIBRARY_PATH=${RUST_PATH}/lib PATH=${RUST_PATH}/bin:${PATH} ${RUST_PATH}/bin/cargo build --release --example nsga2

build-nsga2-edge:
	LD_LIBRARY_PATH=${RUST_PATH}/lib PATH=${RUST_PATH}/bin:${PATH} ${RUST_PATH}/bin/cargo build --release --example nsga2_edge

build-nsga2-edge-ops:
	LD_LIBRARY_PATH=${RUST_PATH}/lib PATH=${RUST_PATH}/bin:${PATH} ${RUST_PATH}/bin/cargo build --release --example nsga2_edge_ops


run-ga: build-ga
	time ./target/release/examples/ga

run-nsga2: build-nsga2
	time ./target/release/examples/nsga2

run-nsga2-edge: build-nsga2-edge
	time ./target/release/examples/nsga2_edge --ngen 10000

run-nsga2-edge-ops: build-nsga2-edge-ops
	time ./target/release/examples/nsga2_edge_ops --ngen 10000 --n 100
