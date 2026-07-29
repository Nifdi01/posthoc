set -euo pipefail

RAW_DIR="datasets/data/raw"
OUT_DIR="datasets/data/processed"
mkdir -p "$RAW_DIR" "$OUT_DIR"

PGEN_URL="https://www.dropbox.com/s/j72j6uciq5zuzii/all_hg38.pgen.zst?dl=1"
PVAR_URL="https://www.dropbox.com/scl/fi/fn0bcm5oseyuawxfvkcpb/all_hg38_rs.pvar.zst?rlkey=przncwb78rhz4g4ukovocdxaz&dl=1"
PSAM_URL="https://www.dropbox.com/scl/fi/u5udzzaibgyvxzfnjcvjc/hg38_corrected.psam?rlkey=oecjnk4vmbhc8b1p202l0ih4x&dl=1"

# QC thresholds — change these and rerun to regenerate a new filtered pfile
MAF_MIN=0.01
GENO_MAX=0.05 # max per-SNP missingness
MIND_MAX=0.10 # max per-sample missingness

cd "$RAW_DIR"
[ -f all_hg38.pgen.zst ] || wget "$PGEN_URL" -O all_hg38.pgen.zst
[ -f all_hg38.pvar.zst ] || wget "$PVAR_URL" -O all_hg38.pvar.zst
[ -f all_hg38.psam ] || wget "$PSAM_URL" -O all_hg38.psam

[ -f all_hg38.pvar ] || plink2 --zst-decompress all_hg38.pvar.zst >all_hg38.pvar
[ -f all_hg38.pgen ] || plink2 --zst-decompress all_hg38.pgen.zst >all_hg38.pgen
cd - >/dev/null

# Extract subset (chromosome 22, EUR+AMR) — structural filtering only
plink2 --pfile "$RAW_DIR/all_hg38" \
  --chr 22 --from-bp 20000000 --to-bp 25000000 \
  --keep-cat-pheno SuperPop --keep-cat-names EUR,AMR \
  --make-pgen --out "$OUT_DIR/chr22_raw"

# QC filtering — this is now the authoritative, cleaned dataset
plink2 --pfile "$OUT_DIR/chr22_raw" \
  --maf "$MAF_MIN" \
  --geno "$GENO_MAX" \
  --mind "$MIND_MAX" \
  --make-pgen --out "$OUT_DIR/chr22_subset"
