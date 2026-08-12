#!/usr/bin/env bash
# Snapshot the nifti-inference code into the shared `patchset` conda env, so anyone who can
# activate the env can run `patchset-infer` WITHOUT access to this (personal) repo.
#
#   bash scripts/sync_patchset_env.sh                 # default env at /software/anaconda3/envs/patchset
#   bash scripts/sync_patchset_env.sh /path/to/env    # custom env prefix
#
# Re-run after changing inference code/config to refresh the env's copy. The env's Python
# deps are managed separately (see requirements-patchset-infer.txt) — this only syncs code.
set -euo pipefail

ENV_PREFIX="${1:-/software/anaconda3/envs/patchset}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUNDLE="$ENV_PREFIX/share/patchset_infer"
LAUNCHER="$ENV_PREFIX/bin/patchset-infer"

[ -x "$ENV_PREFIX/bin/python" ] || { echo "no python at $ENV_PREFIX/bin/python — wrong env prefix?" >&2; exit 1; }

echo "repo    : $REPO"
echo "env     : $ENV_PREFIX"
echo "bundle  : $BUNDLE"

# Mirror only source files (py/yaml/json), preserving the repo tree so the flat sibling
# imports + sys.path/__file__ logic resolve unchanged. --delete keeps the snapshot clean of
# files removed upstream. Big data/output dirs are pruned so the bundle stays ~code-sized.
mkdir -p "$BUNDLE"
# -rltm (no -og): NFS rejects chgrp/chown, so don't preserve owner/group.
rsync -rltm --delete --no-owner --no-group \
  --exclude='.git/' --exclude='.venv*' --exclude='wandb/' --exclude='results/' \
  --exclude='outputs/' --exclude='artifacts/' --exclude='__pycache__/' \
  --include='*/' \
  --include='*.py' --include='*.yaml' --include='*.yml' --include='*.json' \
  --exclude='*' \
  "$REPO/" "$BUNDLE/"

# Console launcher: run the env's Python on the bundled CLI. Path is resolved relative to the
# launcher so the env stays relocatable.
cat > "$LAUNCHER" <<'EOF'
#!/usr/bin/env bash
HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
exec "$HERE/python" "$HERE/../share/patchset_infer/experiments/3d/infer_cli.py" "$@"
EOF
chmod +x "$LAUNCHER"

echo "bundle size: $(du -sh "$BUNDLE" | cut -f1)"
echo "launcher   : $LAUNCHER"
echo "done. usage: conda activate patchset && patchset-infer --help"
