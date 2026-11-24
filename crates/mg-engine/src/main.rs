//! Musigraph native fullscreen visualizer (P3.1: `--gpu-report` only).

fn main() {
    let arg = std::env::args().nth(1);
    match arg.as_deref() {
        Some("--gpu-report") => print!("{}", mg_gfx::report::adapter_report()),
        None => println!("musigraph engine (P3.1 skeleton): try --gpu-report"),
        Some(other) => {
            eprintln!("unknown option {other:?}; supported: --gpu-report");
            std::process::exit(2);
        }
    }
}
