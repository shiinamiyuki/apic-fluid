fn main() {
    let dst = cmake::build("eigen_pcgsolver");

    println!("cargo:rustc-link-search=native={}/build", dst.display());
    println!("cargo:rustc-link-lib=static=solve");
}