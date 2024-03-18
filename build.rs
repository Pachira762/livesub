fn main() -> std::io::Result<()> {
    println!("cargo:return-if-canged=build.rs");
    println!("cargo:return-if-canged=icon.ico");
    println!("cargo:return-if-canged=manifest.manifest");

    winres::WindowsResource::new()
        .set_icon("icon.ico")
        .set_manifest_file("manifest.manifest")
        .compile()?;

    Ok(())
}
