# APIC Fluid Simulator
Final project for CS888 W2023 at UWaterloo

## Build
See [BUILD.md](https://github.com/LuisaGroup/LuisaCompute/blob/next/BUILD.md) for more information.


## Run
```bash
# fluid simulator
cargo run --release --bin fluid -- scene res dt
# replay
cargo run --release --bin replay -- scene
# surface reconstruction from replay
cargo run --release --bin recon -- scene grid_res raidus [frame start] [frame end] [frame step]
```
