fn main() {
    std::process::Command::new("mdbook")
        .args(&["build"])
        .status()
        .unwrap();
}
