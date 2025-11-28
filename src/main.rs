use std::process::{Command, Stdio};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("--- 📚 Starting mdBook Server on http://0.0.0.0:8000 ---");
    println!("--- Access the book from your local machine at http://localhost:8000 ---");
    
    // mdbook serve runs indefinitely until killed.
    // We use --hostname 0.0.0.0 so the server is accessible outside the container's localhost,
    // allowing the VS Code port forwarding to work correctly.
    let status = Command::new("mdbook")
        .args(&["serve", "--hostname", "0.0.0.0", "--port", "8000"])
        .stdout(Stdio::inherit()) // Pipe command output directly to console
        .stderr(Stdio::inherit())
        .status()?;

    if !status.success() {
        return Err(format!("mdbook serve failed with exit code: {}", 
                           status.code().unwrap_or(-1)).into());
    }

    Ok(())
}
