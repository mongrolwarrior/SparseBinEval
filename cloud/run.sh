#!/usr/bin/env bash
# Container entrypoint: run the held-out size sweep on this box's GPU and (if a reference is
# bundled) check the held-out QE reproduces within tolerance.
#
# Env knobs (all optional):
#   CORPUS_DIR  where the transferred MEDLINE split lives        (default /data)
#   UNTIL       ceiling mode: mem:6 | mem:GB | oom1 | size:N
#                 default mem:6 == the reference config, so the run is directly comparable.
#                 UNTIL=auto -> mem:<90% of detected VRAM> for a full-capacity run instead.
#   OPTIONS     which implementations                            (default all five)
#   OUT_DIR     where sweep.csv is written                       (default /out/<ts>)
#   REFERENCE   reference sweep.csv to check against             (default cloud/reference/sweep.csv)
#   TOL         held-QE relative tolerance for the check         (default 0.02)
set -euo pipefail
cd "${REPO_DIR:-/opt/SparseBinEval}"          # /opt in the image; set REPO_DIR when run on a bare pod

CORPUS_DIR="${CORPUS_DIR:-/data}"
TRAIN="$CORPUS_DIR/corpus.train.sbcsr"
VAL="$CORPUS_DIR/corpus.val.sbcsr"
TEST="$CORPUS_DIR/corpus.test.sbcsr"
LIBSVM="$CORPUS_DIR/corpus.train.libsvm"
OPTIONS="${OPTIONS:-sbsom-bin,sbsom-float,ssom-feat,ssom-node,somoclu}"
OUT_DIR="${OUT_DIR:-/out/repro_$(date +%Y%m%d-%H%M%S)}"
REFERENCE="${REFERENCE:-cloud/reference/sweep.csv}"
TOL="${TOL:-0.02}"

echo "=== GPU ==="
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || { echo "no GPU"; exit 1; }
MEM_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)

UNTIL="${UNTIL:-mem:6}"
if [ "$UNTIL" = "auto" ]; then                       # full-capacity run for THIS card
    UNTIL="mem:$(python3 -c "print(round(${MEM_MIB}/1024*0.9,1))")"
fi
echo "=== config: --until $UNTIL ; options $OPTIONS ==="

# Fetch the corpus from Zenodo if not already present. The tarball ships the unsplit corpus.sbcsr;
# split into train/val/test here so ratios and seed live in code, not the distribution artifact.
if [ ! -f "$CORPUS_DIR/corpus.sbcsr" ] && [ ! -f "$TRAIN" ] && [ -n "${DATA_URL:-}" ]; then
    echo "=== fetching corpus from $DATA_URL -> $CORPUS_DIR ==="
    mkdir -p "$CORPUS_DIR"
    curl -fSL "$DATA_URL" | tar -xz -C "$CORPUS_DIR"
fi
if [ -f "$CORPUS_DIR/corpus.sbcsr" ] && [ ! -f "$VAL" ]; then
    echo "=== splitting corpus into train/val/test (80/10/10) ==="
    python3 split_corpus.py "$CORPUS_DIR/corpus.sbcsr" \
        --train-frac 0.80 --val-frac 0.10 --seed 0 \
        --train-out "$TRAIN" --val-out "$VAL" --test-out "$TEST"
fi
for f in "$TRAIN" "$VAL" "$TEST"; do
    [ -f "$f" ] || { echo "missing corpus file: $f
  mount the split into $CORPUS_DIR, or set DATA_URL to a Zenodo tarball"; exit 1; }
done

# ── MODE=elbow: find the optimal SOM size on a big GPU (sbsom-bin only, full ladder) ──
# Knobs: ELBOW_START (default 32 = full range), UNTIL_ELBOW (default oom1 = this GPU's cap),
#         TRAIN_SUBSAMPLE (default 1.0 = full train set; 0.25 = 25% for ~4× speedup),
#         INIT_PCA (default 1 = use PCA init; 0 = random init).
# With TRAIN_SUBSAMPLE < 1 and INIT_PCA=1, the sweep trains on a subsample with PCA-initialized
# codebooks. The final eval at the elbow trains on the FULL training set.
if [ "${MODE:-repro}" = "elbow" ]; then
    START="${ELBOW_START:-32}"; CEIL="${UNTIL_ELBOW:-oom1}"
    SUB_FRAC="${TRAIN_SUBSAMPLE:-1.0}"
    DO_PCA="${INIT_PCA:-1}"

    SWEEP_TRAIN="$TRAIN"
    PCA_FLAG=""

    # Subsample the training set for faster sweep (val/test stay full-size)
    if python3 -c "import sys; sys.exit(0 if float('$SUB_FRAC') < 1.0 else 1)"; then
        SWEEP_TRAIN="$CORPUS_DIR/corpus.train.sub.sbcsr"
        if [ ! -f "$SWEEP_TRAIN" ]; then
            echo "=== subsampling training set to ${SUB_FRAC} ==="
            python3 subsample_sbcsr.py "$TRAIN" --frac "$SUB_FRAC" --seed 0 --out "$SWEEP_TRAIN"
        fi
    fi

    # Compute PCA components for codebook initialization (~360 KB file, one-time cost)
    if [ "$DO_PCA" = "1" ]; then
        PCA_FILE="$CORPUS_DIR/pca_components.sompca"
        if [ ! -f "$PCA_FILE" ]; then
            echo "=== computing PCA components from training set ==="
            python3 pca_init.py "$TRAIN" --out "$PCA_FILE"
        fi
        PCA_FLAG="--init-pca $PCA_FILE"
    fi

    echo "=== elbow sweep: sbsom-bin, full ladder from edge $START, ceiling '$CEIL' ==="
    echo "  train: $SWEEP_TRAIN  val: $VAL (elbow selection)  test: $TEST (final eval)"
    [ -n "$PCA_FLAG" ] && echo "  init: PCA ($PCA_FILE)"
    # Sweep scores the val corpus for elbow selection (test is held out for the final number)
    python3 -u compare_sweep.py --corpus "$SWEEP_TRAIN" --test-corpus "$VAL" \
        --options sbsom-bin --until "$CEIL" --start "$START" $PCA_FLAG --out "$OUT_DIR/sweep"
    echo "=== elbow analysis (val set) ==="
    python3 find_elbow.py "$OUT_DIR/sweep/sweep.csv" --impl sbsom-bin || true
    # Extract the elbow edge from find_elbow output (the ELBOW line)
    ELBOW_EDGE=$(python3 find_elbow.py "$OUT_DIR/sweep/sweep.csv" --impl sbsom-bin 2>&1 \
        | grep -oP '(?<=ELBOW.*edge )\d+' || echo "")
    if [ -n "$ELBOW_EDGE" ]; then
        echo "=== final eval at elbow edge $ELBOW_EDGE on held-out TEST set ==="
        # Final eval trains on the FULL training set (not subsample), still with PCA init
        python3 -u compare_sweep.py --corpus "$TRAIN" --test-corpus "$TEST" \
            --options sbsom-bin --until "$CEIL" --edges "$ELBOW_EDGE" $PCA_FLAG --out "$OUT_DIR/test_eval"
        echo "=== test-set metrics at elbow ==="
        cat "$OUT_DIR/test_eval/sweep.csv"
    else
        echo "no elbow found (QE still falling) — skipping test-set eval"
    fi
    echo "elbow run complete -> $OUT_DIR/"
    exit 0
fi

# ── MODE=kernel-sweep: benchmark BMU kernel variants at a fixed map size ─────
# Knobs: KERNEL_EDGE (default 513), KERNELS (default tile16,tile32,cluster,tma),
#        INIT_PCA (default 1 = use PCA init; 0 = random init).
if [ "${MODE:-repro}" = "kernel-sweep" ]; then
    KERNEL_EDGE="${KERNEL_EDGE:-513}"
    KERNELS="${KERNELS:-tile16,tile32,cluster,tma}"
    DO_PCA="${INIT_PCA:-1}"
    PCA_FLAG=""

    if [ "$DO_PCA" = "1" ]; then
        PCA_FILE="$CORPUS_DIR/pca_components.sompca"
        if [ ! -f "$PCA_FILE" ]; then
            echo "=== computing PCA components from training set ==="
            python3 pca_init.py "$TRAIN" --out "$PCA_FILE"
        fi
        PCA_FLAG="--init-pca $PCA_FILE"
    fi

    echo "=== kernel sweep: edge=$KERNEL_EDGE, kernels=$KERNELS ==="
    python3 -u kernel_sweep.py --corpus "$TRAIN" --test-corpus "$VAL" \
        --edge "$KERNEL_EDGE" --kernels "$KERNELS" $PCA_FLAG --out-dir "$OUT_DIR"
    echo "kernel sweep complete -> $OUT_DIR/"
    exit 0
fi

# somoclu trains on the train libsvm; regenerate from train.sbcsr if it wasn't transferred.
if echo "$OPTIONS" | grep -q somoclu && [ ! -f "$LIBSVM" ]; then
    echo "=== regenerating $LIBSVM from train split (somoclu input) ==="
    python3 -c "from sbeval import somoclu_run; somoclu_run.to_libsvm('$TRAIN','$LIBSVM')"
fi

echo "=== sweep ==="
python3 -u compare_sweep.py --corpus "$TRAIN" --test-corpus "$VAL" \
    --until "$UNTIL" --options "$OPTIONS" --somoclu-libsvm "$LIBSVM" --out "$OUT_DIR"

if [ -f "$REFERENCE" ]; then
    echo "=== reproducibility check vs $REFERENCE (tol $TOL) ==="
    python3 cloud/check_repro.py "$OUT_DIR/sweep.csv" "$REFERENCE" --tol "$TOL"
else
    echo "(no reference bundled — skipping repro check; sweep.csv in $OUT_DIR is the new reference)"
fi
