import logging
from pathlib import Path

import click
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def heritability_to_k(heritability: float) -> float:
    """Convert heritability to the paper's noise scaling factor k
    (noise_var = k * genetic_var), for logging/parity with Yelmen et al."""
    return (1.0 - heritability) / heritability


def resolve_snp_pool(
    variant_df: pd.DataFrame,
    chromosome: str,
    snp_pool_path: str | None,
) -> np.ndarray:
    """Return an array of eligible variant *positional indices* to sample
    causal SNPs from, honoring --chromosome and/or --snp-pool. Defaults to
    the whole genome (all variants) when neither is given, matching the
    paper's main SCZ simulation scenarios."""

    pool_mask = np.ones(len(variant_df), dtype=bool)

    if chromosome is not None:
        if "CHROM" not in variant_df.columns:
            raise click.UsageError(
                "--chromosome requires CHROM column in the variant metadata."
            )
        pool_mask &= (variant_df["CHROM"].astype(str) == str(chromosome)).to_numpy()

    if snp_pool_path is not None:
        wanted_ids = pd.Series(set(Path(snp_pool_path).read_text().split()))
        if "ID" not in variant_df.columns:
            raise click.UsageError(
                "--snp-pool requires an ID column in the variant metadata."
            )
        pool_mask &= variant_df["ID"].isin(wanted_ids).to_numpy()

    pool = np.where(pool_mask)[0]
    if len(pool) == 0:
        raise click.UsageError(
            "No variants remain after applying --chromosome / --snp-pool filters."
        )
    return pool


def validate_ratios(
    ratio_dominant: float,
    ratio_recessive: float,
    ratio_pair: float,
    ratio_triple: float,
) -> float:
    """Validate user-supplied ratios and return their sum. Ratios need not
    sum to any particular value (they're relative weights, not percentages)
    but must be non-negative with at least one positive entry."""

    ratios = {
        "--ratio-dominant": ratio_dominant,
        "--ratio-recessive": ratio_recessive,
        "--ratio-pair": ratio_pair,
        "--ratio-triple": ratio_triple,
    }

    for name, value in ratios.items():
        if value < 0:
            raise click.UsageError(f"{name} must be non-negative (got {value}).")

    total = sum(ratios.values())
    if total <= 0:
        raise click.UsageError(
            "At least one of --ratio-dominant/--ratio-recessive/--ratio-pair/"
            "--ratio-triple must be positive."
        )

    zeroed = [name for name, value in ratios.items() if value == 0]
    if zeroed:
        logger.info(
            "%s set to 0 — that effect type will be excluded from the simulation.",
            ", ".join(zeroed),
        )

    return total


def log_ratio_percentages(
    ratio_dominant: float,
    ratio_recessive: float,
    ratio_pair: float,
    ratio_triple: float,
    total_ratio: float,
    n_causal: int,
) -> None:
    """Print the effective percentage split for a human-readable summary,
    without requiring the input ratios themselves to sum to 100."""

    pct = {
        "dominant": 100 * ratio_dominant / total_ratio,
        "recessive": 100 * ratio_recessive / total_ratio,
        "pair": 100 * ratio_pair / total_ratio,
        "triple": 100 * ratio_triple / total_ratio,
    }

    logger.info(
        "Ratio %s:%s:%s:%s -> %.1f%% dominant, %.1f%% recessive, %.1f%% pair, "
        "%.1f%% triple (n_causal=%d)",
        ratio_dominant,
        ratio_recessive,
        ratio_pair,
        ratio_triple,
        pct["dominant"],
        pct["recessive"],
        pct["pair"],
        pct["triple"],
        n_causal,
    )


def allocate_causal_positions(
    n_causal: int,
    pool: np.ndarray,
    ratio_dominant: float,
    ratio_recessive: float,
    ratio_pair: float,
    ratio_triple: float,
    total_ratio: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]], list[tuple[int, int, int]]]:
    """Allocate disjoint causal positions across dominant/recessive/pair/triple
    effect types by SNP-count ratio, following Yelmen et al. (2026, default
    ratio 5:5:4:6). Pair/triple SNP counts are trimmed to whole terms
    (divisible by 2/3 respectively)."""

    if n_causal > len(pool):
        raise click.UsageError(
            f"--n-causal ({n_causal}) exceeds available SNP pool ({len(pool)})."
        )

    unit = n_causal / total_ratio
    n_dominant = round(ratio_dominant * unit)
    n_recessive = round(ratio_recessive * unit)
    n_pair = round(ratio_pair * unit)
    n_triple = round(ratio_triple * unit)

    n_pair -= n_pair % 2
    n_triple -= n_triple % 3

    allocated = n_dominant + n_recessive + n_pair + n_triple

    if allocated != n_causal:
        logger.warning(
            "Allocated %d of %d requested causal SNPs after applying ratio "
            "%s:%s:%s:%s (pair/triple counts trimmed to whole terms of 2/3 SNPs).",
            allocated,
            n_causal,
            ratio_dominant,
            ratio_recessive,
            ratio_pair,
            ratio_triple,
        )

    chosen = rng.choice(pool, size=allocated, replace=False)
    rng.shuffle(chosen)

    cursor = 0
    dominant_idx = chosen[cursor : cursor + n_dominant]
    cursor += n_dominant
    recessive_idx = chosen[cursor : cursor + n_recessive]
    cursor += n_recessive
    pair_flat = chosen[cursor : cursor + n_pair]
    cursor += n_pair
    triple_flat = chosen[cursor : cursor + n_triple]
    cursor += n_triple

    interaction_pairs = [
        (int(pair_flat[i]), int(pair_flat[i + 1])) for i in range(0, len(pair_flat), 2)
    ]

    interaction_triplets = [
        (int(triple_flat[i]), int(triple_flat[i + 1]), int(triple_flat[i + 2]))
        for i in range(0, len(triple_flat), 3)
    ]

    return dominant_idx, recessive_idx, interaction_pairs, interaction_triplets
