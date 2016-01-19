RUSTFMT=rustfmt

build:
	cargo build --release

run-ga: build
	time cargo run --release --bin ga

run-nsga2: build
	time cargo run --release --bin nsga2

run-nsga2-edge: build
	time cargo run --release --bin nsga2_edge -- --ngen 10000

run-edgeop: build
	time cargo run --release --bin edgeop -- --ngen 10000 --n 40 --mu 1000 --lambda 500 --seed 45234341,12343423,123239 --pmut 0.025 --ilen 1,40

clean:
	rm -rf target
	find . -name "*.bk" | xargs -I % rm %
