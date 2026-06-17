//! Progress bar utilities using `indicatif`.
//!
//! Provides rich progress display for the CV loop including
//! ETA, elapsed time, and convergence counters.

use crate::spec::Verbosity;
use indicatif::{ProgressBar, ProgressStyle};

/// Creates a progress bar configured for CV folds.
///
/// # Arguments
/// * `n_folds` — Total number of folds
/// * `verbosity` — Output level (Silent returns a hidden bar)
pub fn create_progress_bar(n_folds: usize, verbosity: Verbosity) -> ProgressBar {
    match verbosity {
        Verbosity::Silent => {
            let pb = ProgressBar::hidden();
            pb
        }
        Verbosity::ProgressBar => {
            let pb = ProgressBar::new(n_folds as u64);
            pb.set_style(
                ProgressStyle::with_template(
                    "  {spinner:.green} Fold {pos}/{len} {bar:30.cyan/dim} {percent:>3}% | \
                     Tempo: {elapsed_precise} | ETA: {eta_precise} | {msg}"
                )
                .unwrap()
                .progress_chars("█▓▒░"),
            );
            pb
        }
        Verbosity::Detailed => {
            let pb = ProgressBar::new(n_folds as u64);
            pb.set_style(
                ProgressStyle::with_template(
                    "  {spinner:.green} Fold {pos}/{len} {bar:30.cyan/dim} {percent:>3}% | \
                     Fold: {per_sec} | Tempo: {elapsed_precise} | ETA: {eta_precise} | {msg}"
                )
                .unwrap()
                .progress_chars("█▓▒░"),
            );
            pb
        }
    }
}

/// Prints the CV header banner to stdout.
pub fn print_header(
    arma_p: usize,
    arma_q: usize,
    garch_m: usize,
    garch_s: usize,
    distribution: &str,
    n_folds: usize,
    solver: &str,
    parallel: bool,
) {
    let mode = if parallel { "PARALELO" } else { "SEQUENCIAL" };
    println!();
    println!("╔══════════════════════════════════════════════════════════╗");
    println!(
        "║  ARMA({},{})−GARCH({},{}) | Dist: {:<4} | Folds: {:<6}",
        arma_p, arma_q, garch_m, garch_s, distribution, n_folds
    );
    println!(
        "║  Modo: {} | Solver: {:<8}",
        mode, solver
    );
    println!("╚══════════════════════════════════════════════════════════╝");
    println!();
}

/// Prints the final summary of CV results to stdout.
pub fn print_summary(
    arma_p: usize,
    arma_q: usize,
    garch_m: usize,
    garch_s: usize,
    distribution: &str,
    solver: &str,
    window_size: usize,
    test_size: usize,
    purge_size: usize,
    n_total: usize,
    n_converged: usize,
    conv_rate: f64,
    total_time: f64,
    rmse: f64,
    mae: f64,
    medae: f64,
    mape: Option<f64>,
    me: f64,
) {
    println!();
    println!("══════════════════════════════════════════════════════════");
    println!("  RESULTADOS DA VALIDAÇÃO CRUZADA");
    println!("══════════════════════════════════════════════════════════");
    println!(
        "  Modelo:        ARMA({},{})-GARCH({},{})",
        arma_p, arma_q, garch_m, garch_s
    );
    println!("  Distribuição:  {}", distribution);
    println!("  Solver:        {}", solver);
    println!(
        "  Janela:        {} obs | Teste: {} obs | Purga: {} obs",
        window_size, test_size, purge_size
    );
    println!(
        "  Folds:         {} total | {} OK | {} falhas ({:.1}%)",
        n_total,
        n_converged,
        n_total - n_converged,
        conv_rate
    );
    println!("  Tempo total:   {:.2} segundos", total_time);
    println!("──────────────────────────────────────────────────────────");
    println!("  RMSE:   {:.6}", rmse);
    println!("  MAE:    {:.6}", mae);
    println!("  MedAE:  {:.6}", medae);
    match mape {
        Some(m) => println!("  MAPE:   {:.2}%", m),
        None => println!("  MAPE:   N/A"),
    }
    println!("  ME:     {:.6} (viés)", me);
    println!("══════════════════════════════════════════════════════════");
    println!();
}
