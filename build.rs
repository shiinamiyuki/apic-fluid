fn main() {
    let dst = cmake::Config::new("eigen_pcgsolver")
        .define("CMAKE_BUILD_TYPE", "Release")
        .build();

    println!("cargo:rustc-link-search=native={}/build", dst.display());
    println!("cargo:rustc-link-lib=dylib=solve");
}
