import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell
def _():
    import re
    import wandb
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr

    wandb.login()
    api = wandb.Api()

    # === CONFIGURATION ===
    PROJECT_NAME = "patch_icl_2d_exps_train"
    DATASET = "omnisynth"

    model_configs = {
        "omnisynth": {"universeg": "ur9l61x5", "patchset_cnn": "um0h88cm"},
    }

    runs = {k: {"wandb_name": v} for k, v in model_configs[DATASET].items()}
    print(f"Dataset: {DATASET}")
    return PROJECT_NAME, api, np, pd, plt, re, runs, sns, spearmanr


@app.function
def get_latest_table(run, table_key_substring="val/samples.table.json"):
    """Find and download the latest version of a logged table artifact for a run."""
    artifacts = [
        a for a in run.logged_artifacts()
        if a.type == "run_table"
    ]
    if not artifacts:
        return None
    latest = max(artifacts, key=lambda a: a.version)  # v49 > v33 etc.
    table = latest.get(table_key_substring)  # or whatever the actual logged key is
    return table.get_dataframe()


@app.function
def cell_rc(s):
    """'[[0, 1]]' / '[[[1, 0]]]' -> (row, col) of the (single) grid cell."""
    import re as _re
    n = _re.findall(r"-?\d+", s)
    return int(n[0]), int(n[1])


@app.function
def ink_area_by_class(cell_size=64):
    """Per-character object size = mean ink fraction of the Omniglot glyph,
    averaged over its renditions, for every eval (val+test) class. Keyed by the
    global class_id used in the `character` label ('alphabet/class_id')."""
    import sys
    import numpy as np
    import pandas as pd
    sys.path.insert(0, "/home/dpxuser/dev/patch_icl/src")
    from datasets.omniSynth.config import OmniDiversityConfig
    from datasets.omniSynth.bank import get_or_build_bank

    bank = get_or_build_bank(OmniDiversityConfig(), cell_size=cell_size)
    ids = bank.task_ids("val") + bank.task_ids("test")
    return pd.DataFrame(
        [(cid, float(np.mean([r.mean() for r in bank.get(cid)]))) for cid in ids],
        columns=["class_id", "ink_frac"],
    )


@app.function
def eta_sq(df, group_key, y):
    """Fraction of variance in y explained by a categorical grouping (ANOVA eta^2)."""
    grand = df[y].mean()
    sst = ((df[y] - grand) ** 2).sum()
    ssb = df.groupby(group_key)[y].apply(
        lambda g: len(g) * (g.mean() - grand) ** 2
    ).sum()
    return ssb / sst


@app.function
def eta_sq_cont(df, x, y, q=5):
    """eta^2 of y explained by quantile-binning a continuous feature x into q bins."""
    import pandas as pd
    binned = pd.qcut(df[x], q, duplicates="drop")
    return eta_sq(df.assign(_b=binned), "_b", y)


@app.cell
def _(PROJECT_NAME, api, runs):
    # Load raw per-sample tables. The two runs use *different* schemas:
    #   universeg    -> explicit cols (character, target_pos, context_pos, transforms)
    #   patchset_cnn -> everything packed into a `detail` string.
    for run_name, run_data in runs.items():
        run_data["df"] = get_latest_table(
            api.run(f"tidiane/{PROJECT_NAME}/{run_data['wandb_name']}")
        )
    du = runs["universeg"]["df"]
    dp = runs["patchset_cnn"]["df"]
    print("universeg :", du.shape, list(du.columns))
    print("patchset  :", dp.shape, list(dp.columns))
    return dp, du


@app.cell
def _(dp, du, np, pd, re):
    # Parse `detail` for both runs (same schema):
    # "Angelic/964 mode=aug cells=[[0, 1]] tf=r+7,s0.95,dx-0.13,dy+0.01"
    _pat = re.compile(
        r"^(?P<character>\S+) mode=(?P<target_mode>\S+) "
        r"cells=(?P<target_pos>\[\[.*?\]\]) tf=(?P<transforms>.+)$"
    )
    du2 = pd.concat([du, du["detail"].str.extract(_pat)], axis=1)
    dp2 = pd.concat([dp, dp["detail"].str.extract(_pat)], axis=1)

    def _to_long(df, model):
        out = df[["dataset", "character", "target_pos", "transforms", "dice"]].copy()
        out["model"] = model
        return out

    long = pd.concat(
        [_to_long(du2, "universeg"), _to_long(dp2, "patchset_cnn")], ignore_index=True
    )

    # decode "r+7,s0.95,dx-0.13,dy+0.01" -> numeric aug features
    _tf = long["transforms"].str.extract(
        r"r(?P<rot>[+-]?\d+),s(?P<scale>[\d.]+),"
        r"dx(?P<dx>[+-]?[\d.]+),dy(?P<dy>[+-]?[\d.]+)"
    )
    long["rot"] = _tf["rot"].astype(float)
    long["scale"] = _tf["scale"].astype(float)
    long["dx"] = _tf["dx"].astype(float)
    long["dy"] = _tf["dy"].astype(float)
    long["abs_rot"] = long["rot"].abs()
    long["scale_dev"] = (long["scale"] - 1).abs()
    long["trans"] = np.hypot(long["dx"], long["dy"])

    # Paired frame — universeg keeps context_pos for grid-distance analysis
    _u = du2[["dataset", "character", "target_pos", 
              "transforms", "dice"]].rename(columns={"dice": "dice_uni"})
    _p = dp2[["dataset", "character", "target_pos",
              "transforms", "dice"]].rename(columns={"dice": "dice_pat"})
    mg = _u.merge(_p, on=["dataset", "character", "target_pos", "transforms"])
    mg["delta"] = mg["dice_uni"] - mg["dice_pat"]

    mg["class_id"] = pd.to_numeric(
        mg["character"].astype(str).str.split("/").str.get(1),
        errors="coerce"
    )
    mg.dropna(subset=["class_id"], inplace=True)
    mg["class_id"] = mg["class_id"].astype(int)

    _mtf = mg["transforms"].str.extract(
        r"r(?P<rot>[+-]?\d+),s(?P<scale>[\d.]+),"
        r"dx(?P<dx>[+-]?[\d.]+),dy(?P<dy>[+-]?[\d.]+)"
    )
    mg["abs_rot"] = _mtf["rot"].astype(float).abs()
    mg["scale_dev"] = (_mtf["scale"].astype(float) - 1).abs()
    mg["trans"] = np.hypot(_mtf["dx"].astype(float), _mtf["dy"].astype(float))
    _t = mg["target_pos"].map(cell_rc)
    _c = mg["context_pos"].map(cell_rc)
    mg["tc_dist"] = [abs(a[0] - b[0]) + abs(a[1] - b[1]) for a, b in zip(_t, _c)]
    mg = mg.merge(ink_area_by_class(), on="class_id", how="left")
    print(f"long: {len(long)} rows | paired: {len(mg)} rows "
          f"| ink NaN: {mg.ink_frac.isna().sum()}")
    return long, mg


@app.cell
def _(mg):
    mg.head()
    return


@app.cell
def _(long):
    # === Overall weakness summary ===
    summ = long.groupby("model")["dice"].agg(["mean", "median", "std"])
    summ["frac_fail(<0.1)"] = long.assign(f=long.dice < 0.1).groupby("model")["f"].mean()
    summ["frac_perfect(>0.9)"] = long.assign(p=long.dice > 0.9).groupby("model")["p"].mean()
    summ.round(3)
    return


@app.cell
def _(long, plt, sns):
    # === Dice distribution: universeg is bimodal, patchset caps out ~0.7 ===
    fig1, ax1 = plt.subplots(figsize=(7, 4))
    sns.histplot(
        data=long, x="dice", hue="model", bins=40,
        element="step", stat="density", common_norm=False, ax=ax1,
    )
    ax1.set_title("Per-sample Dice distribution")
    fig1
    return


@app.cell
def _(long, pd, spearmanr):
    # === Which aug factors correlate with Dice? (Spearman) ===
    rows = []
    for m, sub in long.groupby("model"):
        for f in ["abs_rot", "scale_dev", "trans"]:
            rho, _ = spearmanr(sub[f], sub["dice"])
            rows.append({"model": m, "feature": f, "spearman_rho": round(rho, 3)})
    pd.DataFrame(rows).pivot(index="feature", columns="model", values="spearman_rho")
    return


@app.cell
def _(long, pd, plt):
    # === Dice vs rotation magnitude — small rotations are HARDEST ===
    rot_bin = pd.cut(long["abs_rot"], [0, 5, 10, 15, 20, 100])
    by_rot = long.groupby([rot_bin, "model"], observed=True)["dice"].mean().unstack(1)
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    by_rot.plot(marker="o", ax=ax2)
    ax2.set_ylabel("mean Dice")
    ax2.set_title("Dice vs |rotation| (deg)")
    fig2
    return


@app.cell
def _(mg, plt):
    # === Paired comparison: where does each model win? ===
    print("mean delta (uni - pat):", round(mg["delta"].mean(), 3))
    print("uni wins (>0.05):", round((mg.delta > 0.05).mean(), 3),
          "| pat wins:", round((mg.delta < -0.05).mean(), 3),
          "| tie:", round((mg.delta.abs() <= 0.05).mean(), 3))
    fig3, ax3 = plt.subplots(figsize=(5, 5))
    ax3.scatter(mg["dice_pat"], mg["dice_uni"], s=3, alpha=0.1)
    ax3.plot([0, 1], [0, 1], "r--", lw=1)
    ax3.set_xlabel("patchset_cnn Dice")
    ax3.set_ylabel("universeg Dice")
    ax3.set_title("Paired per-sample Dice")
    fig3
    return


@app.cell
def _(mg, pd):
    # === DRIVER ANALYSIS: fraction of Dice variance each driver explains (eta^2) ===
    # eta^2 for categorical drivers; quantile-binned eta^2 for continuous ones.
    drivers = {
        "object size (ink)":  ("cont", "ink_frac"),
        "character id":       ("cat",  "character"),
        "dataset / alphabet": ("cat",  "dataset"),
        "target |rotation|":  ("cont", "abs_rot"),
        "scale deviation":    ("cont", "scale_dev"),
        "translation":        ("cont", "trans"),
        "tgt-ctx distance":   ("cat",  "tc_dist"),
    }
    _rows = []
    for _name, (_kind, _col) in drivers.items():
        _f = eta_sq_cont if _kind == "cont" else eta_sq
        _rows.append({"driver": _name,
                      "universeg": round(_f(mg, _col, "dice_uni"), 3),
                      "patchset_cnn": round(_f(mg, _col, "dice_pat"), 3)})
    eta_tbl = pd.DataFrame(_rows).set_index("driver")
    print(eta_tbl)
    return (eta_tbl,)


@app.cell
def _(eta_tbl, plt):
    # universeg is driven by AUGMENTATION (rotation); patchset by OBJECT IDENTITY/SIZE.
    fig4, ax4 = plt.subplots(figsize=(8, 4.5))
    eta_tbl.plot.barh(ax=ax4)
    ax4.set_xlabel(r"$\eta^2$  (fraction of Dice variance explained)")
    ax4.set_title("Accuracy drivers by model")
    ax4.invert_yaxis()
    fig4
    return


@app.cell
def _(mg, pd, plt):
    # === Directional views: object size helps; grid distance does not ===
    fig5, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11, 4))
    _q = pd.qcut(mg["ink_frac"], 6)
    mg.groupby(_q, observed=True)[["dice_uni", "dice_pat"]].mean().plot(marker="o", ax=ax_a)
    ax_a.set_title("Dice vs object size (ink fraction)")
    ax_a.set_ylabel("mean Dice")
    ax_a.set_xlabel("ink-fraction quantile")
    ax_a.tick_params(axis="x", rotation=30)
    mg.groupby("tc_dist")[["dice_uni", "dice_pat"]].mean().plot(marker="o", ax=ax_b)
    ax_b.set_title("Dice vs target-context grid distance")
    ax_b.set_xlabel("manhattan cell distance")
    fig5
    return


@app.cell
def _(mg, plt, spearmanr):
    # === Are the two models hard on the SAME objects? (shared vs distinct failures) ===
    ch = mg.groupby("character").agg(
        uni=("dice_uni", "mean"), pat=("dice_pat", "mean"), ink=("ink_frac", "first")
    )
    _rho = spearmanr(ch["uni"], ch["pat"])[0]
    fig6, ax6 = plt.subplots(figsize=(5.5, 5))
    sc = ax6.scatter(ch["pat"], ch["uni"], c=ch["ink"], s=10, cmap="viridis")
    ax6.plot([0, 1], [0, 1], "r--", lw=1)
    ax6.set_xlabel("patchset_cnn per-character Dice")
    ax6.set_ylabel("universeg per-character Dice")
    ax6.set_title(f"Per-character difficulty (Spearman rho={_rho:.2f})")
    fig6.colorbar(sc, label="object size (ink fraction)")
    fig6
    return


if __name__ == "__main__":
    app.run()
