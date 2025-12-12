use axm_rs::{program::Program, space::Space};
use std::env;
use std::fs;
use std::path::PathBuf;

fn usage() {
    eprintln!("Usage: axm-cli <summary|query> <path> [major]");
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1).collect::<Vec<_>>();
    if args.is_empty() {
        usage();
        std::process::exit(1);
    }
    let command = args.remove(0);
    let path = args.get(0).cloned().unwrap_or_default();
    if path.is_empty() {
        usage();
        std::process::exit(1);
    }
    let data_path = PathBuf::from(&path);
    let program = if data_path.is_dir() {
        Program::load_dir(&data_path)?
    } else {
        let bytes = fs::read(&data_path)?;
        Program::load_zip_bytes(&bytes)?
    };

    match command.as_str() {
        "summary" => {
            let manifest = program.manifest();
            println!("{}", serde_json::to_string_pretty(&manifest)?);
        }
        "query" => {
            let major = args.get(1).and_then(|m| m.parse::<u32>().ok());
            let space = Space::new(&program);
            let nodes = space.query(major, None, None, None, None, None);
            println!("{}", serde_json::to_string_pretty(&nodes)?);
        }
        _ => usage(),
    }
    Ok(())
}
