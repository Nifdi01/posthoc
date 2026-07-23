set -euo pipefail

RAW_DIR="data/raw"
OUT_DIR="data/processed"
mkdir -p "$RAW_DIR" "$OUT_DIR"

PGEN_URL="https://www.dropbox.com/s/j72j6uciq5zuzii/all_hg38.pgen.zst?dl=1"
PVAR_URL="https://www.dropbox.com/scl/fi/fn0bcm5oseyuawxfvkcpb/all_hg38_rs.pvar.zst?rlkey=przncwb78rhz4g4ukovocdxaz&dl=1"
PSAM_URL="https://www.dropbox.com/scl/fi/u5udzzaibgyvxzfnjcvjc/hg38_corrected.psam?rlkey=oecjnk4vmbhc8b1p202l0ih4x&dl=1"

cd "$RAW_DIR"
[ -f all_hg38.pgen.zst ] || wget "$PGEN_URL" -O all_hg38.pgen.zst
[ -f all_hg38.pvar.zst ] || wget "$PVAR_URL" -O all_hg38.pvar.zst
[ -f all_hg38.psam ] || wget "$PSAM_URL" -O all_hg38.psam

[ -f all_hg38.pvar ] || plink2 --zst-decompress all_hg38.pvar.zst >all_hg38.pvar
[ -f all_hg38.pgen ] || plink2 --zst-decompress all_hg38.pgen.zst >all_hg38.pgen
cd - >/dev/null

# Extract Subset (chromosome 22)
plink2 --pfile "$RAW_DIR/all_hg38" \
  --chr 22 --from-bp 20000000 --to-bp 25000000 \
  --keep-cat-pheno SuperPop --keep-cat-names EUR,AMR \
  --make-pgen --out "$OUT_DIR/chr22_subset"

# Check Subset
plink2 --pfile "$OUT_DIR/chr22_subset" --freq --out "$OUT_DIR/chr22_check"
plink2 --pfile "$OUT_DIR/chr22_subset" --missing --out "$OUT_DIR/chr22_check"
plink2 --pfile "$OUT_DIR/chr22_subset" --maf 0.01 --pca 10 --out "$OUT_DIR/chr22_check"
