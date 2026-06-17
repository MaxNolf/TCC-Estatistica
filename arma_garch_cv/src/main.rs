//! CLI entry point for ARMA-GARCH purged cross-validation.

use arma_garch_cv::spec::*;
use arma_garch_cv::cv::purged::run_purged_cv;

fn main() {
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║  arma-garch-cv v0.1.0                                   ║");
    println!("║  ARMA-GARCH com Validação Cruzada Temporal Purged       ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    println!();

    // ── Example usage (will fail until estimation is implemented) ──────
    // Generate synthetic data for demonstration
    let data: Vec<f64> = (0..200)
        .map(|i| (i as f64 * 0.1).sin() + 0.5)
        .collect();

    let spec = ModelSpec::new(
        ArmaOrder::new(1, 1),
        GarchOrder::new(1, 1),
        Distribution::Normal,
        Solver::LBFGS,
    );

    let params = CvParams::default_for_length(data.len());

    println!("  Modelo: {}", spec);
    println!("  Dados:  {} observações", data.len());
    println!("  Janela: {} | Teste: {} | Purga: {}",
        params.window_size, params.test_size, params.purge_size
    );
    println!();

    match run_purged_cv(&data, &spec, &params, false, Verbosity::ProgressBar) {
        Ok(result) => {
            println!("Resultados:");
            println!("{}", result.metrics);
        }
        Err(e) => {
            eprintln!("⚠ Erro esperado: {}", e);
            eprintln!();
            eprintln!("  A estimação GARCH ainda não foi implementada.");
            eprintln!("  Implemente src/estimation/solver.rs para rodar a CV completa.");
        }
    }
}
