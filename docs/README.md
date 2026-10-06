# Docs

## Compile tex files
# local textlive install 
export PATH=/home/dpxuser/texlive/2026/bin/x86_64-linux:$PATH
# thesis
cd /home/dpxuser/dev/patch_icl/results/publications/thesis/ics_3d_medical
latexmk -synctex=1 -interaction=nonstopmode -file-line-error -pdf -output-directory=output main.tex
makeglossaries -d output main
latexmk -synctex=1 -interaction=nonstopmode -file-line-error -pdf -output-directory=output main.tex
# figures
cd /home/dpxuser/dev/patch_icl/results/publications/thesis/ics_3d_medical/imgs/method
latexmk -synctex=1 -interaction=nonstopmode -file-line-error -pdf arch_fusion_family

## Syncing the thesis with Overleaf

The thesis (`results/publications/thesis/ics_3d_medical/`) is also an Overleaf
project, kept as a separate, persistent clone at
`~/dev/thesis/6abacee51bee9087ddbb00c8` (its own git repo, remote = Overleaf's
git bridge). Overleaf's repo has no shared history with `patch_icl`, so sync
is plain file copy (via `rsync`, driven by each side's `git ls-files`) plus a
normal commit+push on whichever side received the files — not `git subtree`,
which doesn't fit here (Overleaf disallows force-push and the two repos share
no ancestry, so subtree push/pull both fight the mismatch instead of working
with it).

Always dry-run the `rsync` first and skim the output before applying — if the
other side has independent edits since the last sync, a blind copy picks a
winner silently.

### Push (patch_icl → Overleaf)

```bash
CLONE=~/dev/thesis/6abacee51bee9087ddbb00c8
SRC=~/dev/patch_icl/results/publications/thesis/ics_3d_medical
LIST=/tmp/overleaf_filelist.txt

cd "$CLONE" && git pull        # avoid a non-fast-forward reject on push

git -C ~/dev/patch_icl ls-files -- results/publications/thesis/ics_3d_medical \
  | sed 's#^results/publications/thesis/ics_3d_medical/##' > "$LIST"

rsync -avc --dry-run --files-from="$LIST" "$SRC/" "$CLONE/"   # preview
rsync -avc --files-from="$LIST" "$SRC/" "$CLONE/"              # apply

cd "$CLONE"
git add -A && git commit -m "sync from patch_icl" && git push
```

### Pull (Overleaf → patch_icl)

```bash
CLONE=~/dev/thesis/6abacee51bee9087ddbb00c8
DEST=~/dev/patch_icl/results/publications/thesis/ics_3d_medical
LIST=/tmp/overleaf_filelist.txt

cd "$CLONE" && git pull

git -C "$CLONE" ls-files > "$LIST"

rsync -avc --dry-run --files-from="$LIST" "$CLONE/" "$DEST/"   # preview
rsync -avc --files-from="$LIST" "$CLONE/" "$DEST/"              # apply

cd ~/dev/patch_icl
git add results/publications/thesis/ics_3d_medical
git commit -m "thesis: pull edits from Overleaf"
git push origin main
```

### Notes

- If `git pull`/`push` in the clone fails authentication, the remote URL
  needs a fresh Overleaf git token (Overleaf → Account Settings → Git
  Authentication Tokens):
  `git remote set-url origin "https://git:<TOKEN>@git.overleaf.com/<project_id>"`.
- `results/publications/thesis/ics_3d_medical/thesis_full.tex` is a flattened
  single-file export, regenerated with
  `python3 flatten_tex.py main.tex > thesis_full.tex` (run from that
  directory) whenever the modular sources change and a single-file copy is
  needed. It's tracked, so it syncs like any other file above.
- Never run `pdflatex`/`lualatex` on the thesis sources after editing —
  compilation is the user's own step.