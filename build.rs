fn main() {
    let dst = cmake::Config::new("cpp")
        .define("CMAKE_BUILD_TYPE", "Release")
        .build();

    println!("cargo:rustc-link-search=native={}/build", dst.display());
    println!("cargo:rustc-link-lib=dylib=cpp_extra");
}
