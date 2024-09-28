fn main() -> std::io::Result<()> {
    println!("cargo:rerun-if-canged=build.rs");
    println!("cargo:rerun-if-canged=icon.ico");
    println!("cargo:rerun-if-canged=manifest.manifest");

    winres::WindowsResource::new()
        .set_icon("icon.ico")
        .set_manifest_file("manifest.manifest")
        .compile()?;

    Ok(())
}
