set -euo pipefail

RAW_DIR="datasets/data/raw"
OUT_DIR="datasets/data/processed"
mkdir -p "$RAW_DIR" "$OUT_DIR"

PGEN_URL="https://www.dropbox.com/s/j72j6uciq5zuzii/all_hg38.pgen.zst?dl=1"
PVAR_URL="https://www.dropbox.com/scl/fi/id642dpdd858uy41og8qi/all_hg38_rs_noannot.pvar.zst?rlkey=sskyiyam1bsqweujjmxqv1h55&dl=1"
PSAM_URL="https://www.dropbox.com/scl/fi/u5udzzaibgyvxzfnjcvjc/hg38_corrected.psam?rlkey=oecjnk4vmbhc8b1p202l0ih4x&dl=1"

cd "$RAW_DIR"
[ -f all_hg38.pgen.zst ] || wget -q "$PGEN_URL" -O all_hg38.pgen.zst
[ -f all_hg38.pvar.zst ] || wget -q "$PVAR_URL" -O all_hg38.pvar.zst
[ -f all_hg38.psam ] || wget -q "$PSAM_URL" -O all_hg38.psam
[ -f all_hg38.pgen ] || plink2 --zst-decompress all_hg38.pgen.zst all_hg38.pgen
cd - >/dev/null

# Extract chr22 subset only — no QC/allele-frequency/missingness filtering
plink2 --pfile "$RAW_DIR/all_hg38" vzs \
  --chr 22 --from-bp 15000000 --to-bp 25000000 \
  --make-pgen --out "$OUT_DIR/chr22_subset"
