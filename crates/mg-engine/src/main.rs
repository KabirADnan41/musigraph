//! Musigraph native fullscreen visualizer (P3.1: `--gpu-report` only).

use std::io::Write as _;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.as_slice() {
        [flag] if flag == "--gpu-report" => {
            // A closed pipe must not panic the process.
            let _ = std::io::stdout().write_all(mg_gfx::report::adapter_report().as_bytes());
        }
        [] => println!("musigraph engine (P3.1 skeleton): try --gpu-report"),
        _ => {
            eprintln!("unsupported arguments; supported: --gpu-report");
            std::process::exit(2);
        }
    }
}
