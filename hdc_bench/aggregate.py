"""Aggregate run records into tidy CSVs, markdown tables and figures.

The raw records under ``results/raw/`` are flattened into a tidy table and then
summarised over seeds.  Everything the README quotes is produced here, so the
numbers in the README are reproducible with:

  python scripts/make_report.py
"""

from __future__ import annotations

import os

import numpy as np

from .paths import FIGURE_DIR, TABLE_DIR
from .records import load_records

HEADS = ["hdc4096", "hdc10000", "linear", "mlp", "proto", "net"]


def flatten(rec: dict, prefix: str = "", out: dict | None = None) -> dict:
    out = {} if out is None else out
    for k, v in rec.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            flatten(v, f"{key}_", out)
        elif isinstance(v, list):
            continue
        else:
            out[key] = v
    return out


def suite_table(suite: str):
    import pandas as pd

    rows = [flatten(r) for r in load_records(suite)]
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _sv(series, fmt="{:.3f}"):
    vals = [v for v in series if isinstance(v, (int, float)) and not np.isnan(v)]
    if not vals:
        return "-"
    if len(vals) == 1:
        return fmt.format(vals[0])
    return f"{fmt.format(np.mean(vals))} ± {fmt.format(np.std(vals))}"


def head_table(df, value_col: str, key: str, groups: list[str],
               group_order: list | None = None, heads: list[str] = HEADS,
               fmt="{:.3f}") -> str:
    """Markdown table: rows heads, columns = groups (mean ± std over rows)."""
    lines = ["| head | " + " | ".join(str(g) for g in groups) + " |",
             "| :--- | " + " | ".join(["---:"] * len(groups)) + " |"]
    for head in heads:
        cells = []
        for g in groups:
            sub = df
            for keyname, gv in zip(group_order or [key], [g]):
                sub = sub[sub[keyname] == gv]
            col = value_col if len(heads) == 1 and head == "" else f"{value_col}_{head}"
            if col not in sub:
                cells.append("-")
                continue
            cells.append(_sv(sub[col], fmt))
        lines.append(f"| {head} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def save_md(name: str, text: str):
    path = os.path.join(TABLE_DIR, name)
    with open(path, "w") as f:
        f.write(text + "\n")
    print(f"[table] {path}")


def save_fig(fig, name: str):
    path = os.path.join(FIGURE_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[figure] {path}")


# ---------------------------------------------------------------------------
# dataset-specific summaries
# ---------------------------------------------------------------------------

def image_report(suite: str, key: str, k_series_n: int = 20,
                 n_series_k: int = 50):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = suite_table(suite)
    if df.empty:
        print(f"[report] no records for {suite}")
        return df
    df.to_csv(os.path.join(TABLE_DIR, f"tidy_{key}.csv"), index=False)
    heads = ["hdc4096", "hdc10000", "linear", "mlp", "proto"]

    for backbone in sorted(df["backbone"].unique()):
        bb = df[df["backbone"] == backbone].copy()
        tag = f"{key}_{backbone}"
        kk = sorted(bb["K"].unique())
        kser = bb[bb["N"] == k_series_n].copy()   # N fixed, K varies
        nser = bb[bb["K"] == n_series_k].copy()   # K fixed, N varies

        # --- ID accuracy (N-independent, include all rows) ----------------
        save_md(f"id_{tag}_by_K.md", f"{key} / {backbone}: ID accuracy vs K\n\n" +
                head_table(bb, "id", "K", kk, heads=heads))
        save_md(f"id_{tag}_by_N.md", f"{key} / {backbone}: ID accuracy vs N (K={n_series_k})\n\n" +
                head_table(nser, "id", "N", sorted(nser["N"].unique()), heads=heads))

        # --- dimension delta ---------------------------------------------
        lines = ["| K | hdc4096 | hdc10000 | linear | mlp | proto | hdc10k - hdc4k |",
                 "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for k in kk:
            sub = bb[bb["K"] == k]

            def m(col):
                v = sub[col].dropna()
                return float(np.mean(v)) if len(v) else float("nan")

            lines.append(f"| {k} | {_sv(sub['id_hdc4096'])} | {_sv(sub['id_hdc10000'])} | "
                         f"{_sv(sub['id_linear'])} | {_sv(sub['id_mlp'])} | {_sv(sub['id_proto'])} | "
                         f"{m('id_hdc10000') - m('id_hdc4096'):+.3f} |")
        save_md(f"id_{tag}_dims.md", f"{key} / {backbone}: ID accuracy by HD dimension\n\n" +
                "\n".join(lines))

        # --- novel no-label clustering (K-series, N fixed) ----------------
        lines = ["| space | " + " | ".join(str(k) for k in kk) + " |",
                 "| :--- | " + " | ".join(["---:"] * len(kk)) + " |"]
        for r in ["cluster_feat", "cluster_hdc4096", "cluster_hdc10000"]:
            cells = [_sv(kser[kser["K"] == k][f"novel_{r}"]) if f"novel_{r}" in kser else "-"
                     for k in kk]
            lines.append(f"| {r} | " + " | ".join(cells) + " |")
        save_md(f"novel_cluster_{tag}_by_K.md",
                f"{key} / {backbone}: novel no-label k-means accuracy vs K (N={k_series_n})\n\n" +
                "\n".join(lines))

        # --- OOD AUROC ---------------------------------------------------
        aurocs = ["auroc_hdc4096", "auroc_hdc10000", "auroc_linear", "auroc_mlp", "auroc_proto"]
        lines = ["| score | " + " | ".join(str(k) for k in kk) + " |",
                 "| :--- | " + " | ".join(["---:"] * len(kk)) + " |"]
        for a in aurocs:
            cells = [_sv(kser[kser["K"] == k][f"novel_{a}"]) for k in kk]
            lines.append(f"| {a} | " + " | ".join(cells) + " |")
        save_md(f"ood_auroc_{tag}_by_K.md",
                f"{key} / {backbone}: OOD AUROC (known vs novel) vs K (N={k_series_n})\n\n" +
                "\n".join(lines))

        # --- robustness (strongest variant) ------------------------------
        rob_cols = [c for c in bb.columns if c.startswith("robust_")]
        if rob_cols:
            variants = sorted({c[len("robust_"):].rsplit("_", 1)[0] for c in rob_cols})
            lines = ["| K | variant | " + " | ".join(heads) + " |",
                     "| ---: | :--- | " + " | ".join(["---:"] * len(heads)) + " |"]
            for k in kk:
                sub = bb[bb["K"] == k]
                for var in variants:
                    cells = []
                    for h in heads:
                        col = f"robust_{var}_{h}"
                        cells.append(_sv(sub[col]) if col in sub else "-")
                    lines.append(f"| {k} | {var} | " + " | ".join(cells) + " |")
            save_md(f"robust_{tag}_by_K.md",
                    f"{key} / {backbone}: robustness vs K\n\n" + "\n".join(lines))

        # --- N-series ----------------------------------------------------
        if not nser.empty:
            nn = sorted(nser["N"].unique())
            lines = ["| metric | " + " | ".join(str(n) for n in nn) + " |",
                     "| :--- | " + " | ".join(["---:"] * len(nn)) + " |"]
            for r in ["cluster_feat", "cluster_hdc4096", "cluster_hdc10000",
                      "auroc_hdc4096", "auroc_hdc10000", "auroc_linear", "auroc_mlp",
                      "auroc_proto"]:
                cells = [_sv(nser[nser["N"] == n][f"novel_{r}"]) for n in nn]
                lines.append(f"| {r} | " + " | ".join(cells) + " |")
            for m in ["hdc4096", "hdc10000", "proto", "linear"]:
                cells = [_sv(nser[nser["N"] == n][f"novel_fewshot5_{m}"]) for n in nn]
                lines.append(f"| fewshot5_{m} | " + " | ".join(cells) + " |")
            save_md(f"novel_{tag}_by_N.md",
                    f"{key} / {backbone}: novel metrics vs N (K={n_series_k})\n\n" +
                    "\n".join(lines))

        # --- figures -----------------------------------------------------
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for h in heads:
            ys = [np.mean(bb[bb["K"] == k][f"id_{h}"].dropna()) for k in kk]
            axes[0].plot(kk, ys, marker="o", label=h)
        axes[0].set_title(f"ID top-1 - {key}/{backbone}")
        for h in ["cluster_feat", "cluster_hdc4096", "cluster_hdc10000"]:
            col = f"novel_{h}"
            if col not in kser:
                continue
            ys = [np.mean(kser[kser["K"] == k][col].dropna()) for k in kk]
            axes[1].plot(kk, ys, marker="o", label=h.replace("cluster_", ""))
        axes[1].set_title("novel cluster")
        for h in ["hdc4096", "hdc10000", "linear", "mlp", "proto"]:
            col = f"novel_auroc_{h}"
            ys = [np.mean(kser[kser["K"] == k][col].dropna()) for k in kk]
            axes[2].plot(kk, ys, marker="o", label=h)
        axes[2].set_title("OOD AUROC")
        for ax in axes:
            ax.set_xscale("log")
            ax.set_xlabel("K")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
        save_fig(fig, f"summary_{tag}.png")
        plt.close(fig)
    return df


def synthetic_report():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = suite_table("synthetic")
    if df.empty:
        print("[report] no synthetic records")
        return df
    df.to_csv(os.path.join(TABLE_DIR, "tidy_synthetic.csv"), index=False)

    for gen in sorted(df["generator"].unique()):
        g = df[df["generator"] == gen]
        native = gen == "bits"
        ks = sorted(g[g["N"] == 20]["K"].unique())
        for dim in (sorted(g["hd_dim"].dropna().unique()) if native else [4096]):
            dimtag = str(int(dim)) if native else "proj"
            sub = g[g["N"] == 20]
            if native:
                sub = sub[sub["hd_dim"] == dim]
                heads = [f"hdc{int(dim)}", "linear", "mlp", "proto"]
                title = f"{gen} (D={int(dim)})"
            else:
                heads = ["hdc4096", "hdc10000", "linear", "mlp", "proto"]
                title = gen
            lines = ["| head | " + " | ".join(str(k) for k in ks) + " |",
                     "| :--- | " + " | ".join(["---:"] * len(ks)) + " |"]
            for h in heads:
                col = f"id_{h}"
                cells = [_sv(sub[sub["K"] == k][col]) for k in ks]
                lines.append(f"| {h} | " + " | ".join(cells) + " |")
            save_md(f"id_{gen}_D{dimtag}.md",
                    f"ID accuracy vs K - {title}\n\n" + "\n".join(lines))

            if not native:
                lines = ["| K | hdc4096 | hdc10000 | linear | mlp | proto | hdc10k - hdc4k |",
                         "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
                for k in ks:
                    subk = sub[sub["K"] == k]

                    def mk(col):
                        v = subk[col].dropna()
                        return float(np.mean(v)) if len(v) else float("nan")

                    lines.append(
                        f"| {k} | {_sv(subk['id_hdc4096'])} | {_sv(subk['id_hdc10000'])} | "
                        f"{_sv(subk['id_linear'])} | {_sv(subk['id_mlp'])} | "
                        f"{_sv(subk['id_proto'])} | "
                        f"{mk('id_hdc10000') - mk('id_hdc4096'):+.3f} |")
                save_md(f"id_{gen}_Ddims.md", f"ID accuracy by HD dimension - {gen}\n\n" +
                        "\n".join(lines))

            # robustness
            rob_cols = [c for c in sub.columns if c.startswith("robust_") and sub[c].notna().any()]
            if rob_cols:
                variants = sorted({c[len("robust_"):].rsplit("_", 1)[0] for c in rob_cols})
                lines = ["| K | variant | " + " | ".join(heads) + " |",
                         "| ---: | :--- | " + " | ".join(["---:"] * len(heads)) + " |"]
                for k in ks:
                    subk = sub[sub["K"] == k]
                    for var in variants:
                        cells = []
                        for h in heads:
                            col = f"robust_{var}_{h}"
                            cells.append(_sv(subk[col]) if col in subk else "-")
                        lines.append(f"| {k} | {var} | " + " | ".join(cells) + " |")
                save_md(f"robust_{gen}_D{dimtag}.md",
                        f"Robustness vs K - {title}\n\n" + "\n".join(lines))

            # novel cluster + auroc + fewshot tables
            for metric, label in [("novel_cluster", "Novel no-label cluster acc"),
                                  ("novel_auroc", "OOD AUROC"),
                                  ("novel_fewshot5", "5-shot novel acc")]:
                rows = [c for c in sub.columns if c.startswith(f"{metric}_")
                        and "time" not in c and sub[c].notna().any()]
                lines = ["| metric | " + " | ".join(str(k) for k in ks) + " |",
                         "| :--- | " + " | ".join(["---:"] * len(ks)) + " |"]
                for r in sorted(set(rows)):
                    cells = [_sv(sub[sub["K"] == k][r], "{:.3f}") for k in ks]
                    lines.append(f"| {r.replace(metric + '_', '')} | " + " | ".join(cells) + " |")
                if len(lines) > 2:
                    save_md(f"{metric}_{gen}_D{dimtag}.md",
                            f"{label} vs K - {title}\n\n" + "\n".join(lines))

            # figure
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            for h in heads:
                col = f"id_{h}"
                ys = [np.mean(sub[sub["K"] == k][col].dropna()) for k in ks]
                axes[0].plot(ks, ys, marker="o", label=h)
            axes[0].set_title(f"ID top-1 - {title}")
            for h in (["cluster_feat", f"cluster_hdc{int(dim)}"] if native
                      else ["cluster_feat", "cluster_hdc4096", "cluster_hdc10000"]):
                col = f"novel_{h}"
                if col not in sub:
                    continue
                ys = [np.mean(sub[sub["K"] == k][col].dropna()) for k in ks]
                axes[1].plot(ks, ys, marker="o", label=h.replace("cluster_", ""))
            axes[1].set_title(f"novel cluster - {title}")
            for h in (["hdc4096", "hdc10000", "linear", "mlp", "proto"] if not native
                      else [f"hdc{int(dim)}", "linear", "mlp", "proto"]):
                col = f"novel_auroc_{h}"
                if col not in sub:
                    continue
                ys = [np.mean(sub[sub["K"] == k][col].dropna()) for k in ks]
                axes[2].plot(ks, ys, marker="o", label=h)
            axes[2].set_title(f"OOD AUROC - {title}")
            for ax in axes:
                ax.set_xscale("log")
                ax.set_xlabel("K")
                ax.grid(alpha=0.3)
                ax.legend(fontsize=8)
            save_fig(fig, f"synthetic_{gen}_D{dimtag}.png")
            plt.close(fig)
    return df


def pretrain_report():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for suite, key in [("pretrain/mlp_synthetic", "pretrain_mlp"),
                       ("pretrain/cifar", "pretrain_cifar"),
                       ("pretrain/tiny", "pretrain_tiny")]:
        df = suite_table(suite)
        if df.empty:
            continue
        df.to_csv(os.path.join(TABLE_DIR, f"tidy_{key}.csv"), index=False)
        ks = sorted(df["K"].unique())
        lines = ["| head | " + " | ".join(str(k) for k in ks) + " |",
                 "| :--- | " + " | ".join(["---:"] * len(ks)) + " |"]
        for h in ["net", "linear", "mlp", "hdc4096", "hdc10000", "proto"]:
            cells = [_sv(df[df["K"] == k][f"id_{h}"]) for k in ks]
            lines.append(f"| {h} | " + " | ".join(cells) + " |")
        save_md(f"id_{key}.md", f"ID accuracy vs pretraining K ({key})\n\n" + "\n".join(lines))

        lines = ["| metric | " + " | ".join(str(k) for k in ks) + " |",
                 "| :--- | " + " | ".join(["---:"] * len(ks)) + " |"]
        for r in ["cluster_feat", "cluster_hdc4096", "cluster_hdc10000",
                  "auroc_net", "auroc_hdc4096", "auroc_hdc10000", "auroc_linear",
                  "auroc_mlp", "auroc_proto", "fewshot5_hdc4096", "fewshot5_hdc10000",
                  "fewshot5_proto", "fewshot5_linear"]:
            col = f"novel_{r}"
            if col not in df:
                continue
            cells = [_sv(df[df["K"] == k][col]) for k in ks]
            lines.append(f"| {r} | " + " | ".join(cells) + " |")
        if len(lines) > 2:
            save_md(f"novel_{key}.md", f"Novel metrics vs pretraining K ({key})\n\n" + "\n".join(lines))

        rob_cols = [c for c in df.columns if c.startswith("robust_") and df[c].notna().any()]
        if rob_cols:
            variants = sorted({c[len("robust_"):].rsplit("_", 1)[0] for c in rob_cols})
            lines = ["| K | variant | " + " | ".join(["net", "linear", "mlp", "hdc4096",
                                                       "hdc10000", "proto"]) + " |",
                     "| ---: | :--- | " + " | ".join(["---:"] * 6) + " |"]
            for k in ks:
                sub = df[df["K"] == k]
                for var in variants:
                    cells = []
                    for h in ["net", "linear", "mlp", "hdc4096", "hdc10000", "proto"]:
                        col = f"robust_{var}_{h}"
                        cells.append(_sv(sub[col]) if col in sub else "-")
                    lines.append(f"| {k} | {var} | " + " | ".join(cells) + " |")
            save_md(f"robust_{key}.md", f"Robustness vs pretraining K ({key})\n\n" + "\n".join(lines))

        fig, ax = plt.subplots(figsize=(6, 4))
        for h in ["net", "linear", "mlp", "hdc4096", "hdc10000", "proto"]:
            col = f"id_{h}"
            if col not in df:
                continue
            ys = [np.mean(df[df["K"] == k][col].dropna()) for k in ks]
            ax.plot(ks, ys, marker="o", label=h)
        ax.set_xscale("log")
        ax.set_xlabel("pretraining K")
        ax.set_ylabel("ID top-1")
        ax.set_title(key)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        save_fig(fig, f"{key}_vs_K.png")
        plt.close(fig)


def cross_summary():
    """Compact all-benchmark summary tables used by the README."""
    import numpy as np
    import pandas as pd

    rows = []

    def add_bench(bench, backbone, df):
        if df is None or df.empty:
            return
        df = df.copy()
        for col in ["id_hdc4096", "id_hdc10000", "id_linear", "id_mlp", "id_proto",
                    "novel_cluster_feat", "novel_cluster_hdc4096", "novel_cluster_hdc10000",
                    "novel_auroc_hdc4096", "novel_auroc_linear", "novel_auroc_mlp",
                    "novel_auroc_proto", "novel_fewshot5_hdc4096", "novel_fewshot5_linear",
                    "novel_fewshot5_proto"]:
            if col not in df:
                df[col] = np.nan
        ks = sorted(df["K"].dropna().unique())
        kmin, kmax = int(ks[0]), int(ks[-1])
        M = {"bench": bench, "backbone": backbone, "Kmin": kmin, "Kmax": kmax,
             "n_runs": int(len(df))}

        def mean_where(frame, col, **conds):
            sub = frame
            for k, v in conds.items():
                sub = sub[sub[k] == v]
            v = sub[col].dropna()
            return float(np.mean(v)) if len(v) else float("nan")

        # primary HD head: 4k normally, 10k-only for the D=10000 bits rows
        h_primary = "hdc4096" if df.get("id_hdc4096", pd.Series(dtype=float)).notna().any() \
            else "hdc10000"
        other = "hdc10000" if h_primary == "hdc4096" else "hdc4096"
        M["h_primary"] = h_primary
        M[f"id_{h_primary}_kmin"] = mean_where(df, f"id_{h_primary}", K=kmin)
        M[f"id_{h_primary}_kmax"] = mean_where(df, f"id_{h_primary}", K=kmax)
        for h in ["linear", "mlp", "proto"]:
            M[f"id_{h}_kmin"] = mean_where(df, f"id_{h}", K=kmin)
            M[f"id_{h}_kmax"] = mean_where(df, f"id_{h}", K=kmax)
        both = df[[f"id_{h_primary}", f"id_{other}"]].dropna()
        M["dim_delta"] = (float((both[f"id_{other}"] - both[f"id_{h_primary}"]).mean())
                          if len(both) and {h_primary, other} == {"hdc4096", "hdc10000"} else
                          float("nan"))

        kser = df[df["N"] == 20]
        if not kser.empty:
            kk = sorted(kser["K"].unique())
            nkmin, nkmax = int(kk[0]), int(kk[-1])
            M["Nkmin"], M["Nkmax"] = nkmin, nkmax
            for h in ["feat", h_primary]:
                M[f"cluster_{h}_n"] = mean_where(kser, f"novel_cluster_{h}", K=nkmin)
            for h in [h_primary, "linear", "mlp", "proto"]:
                M[f"auroc_{h}_nmin"] = mean_where(kser, f"novel_auroc_{h}", K=nkmin)
                M[f"auroc_{h}_nmax"] = mean_where(kser, f"novel_auroc_{h}", K=nkmax)
            for h in [h_primary, "linear", "proto"]:
                M[f"fs5_{h}"] = mean_where(kser, f"novel_fewshot5_{h}", K=nkmin)
        # N-series endpoint (K fixed, largest N)
        nser = df[df["K"] == (50 if bench == "CIFAR-100" else 100)]
        if not nser.empty and nser["N"].max() > 20:
            nmax = nser["N"].max()
            M["cluster_feat_nmax"] = mean_where(nser, "novel_cluster_feat", N=nmax)
            M["cluster_hdc_nmax"] = mean_where(nser, f"novel_cluster_{h_primary}", N=nmax)
        rows.append(M)

    for suite, key in [("images/cifar100", "CIFAR-100"), ("images/tinyimagenet", "TinyImageNet")]:
        df = suite_table(suite)
        if not df.empty:
            for bb in sorted(df["backbone"].unique()):
                add_bench(key, bb, df[df["backbone"] == bb])
    sdf = suite_table("synthetic")
    if not sdf.empty:
        for gen in sorted(sdf["generator"].unique()):
            g = sdf[sdf["generator"] == gen]
            if gen == "bits":
                for dim in sorted(g["hd_dim"].dropna().unique()):
                    add_bench(f"synth {gen}", f"D={int(dim)}", g[g["hd_dim"] == dim])
            else:
                add_bench(f"synth {gen}", "proj", g)
    p = suite_table("pretrain/mlp_synthetic")
    if not p.empty:
        add_bench("pretrain MLP", "MLP", p)
    for suite, key in [("pretrain/cifar", "pretrain CIFAR"), ("pretrain/tiny", "pretrain Tiny")]:
        q = suite_table(suite)
        if not q.empty:
            add_bench(key, "SmallResNet", q)

    if not rows:
        print("[report] no cross summary rows")
        return

    def f(r, k):
        v = r.get(k, float("nan"))
        return f"{v:.3f}" if isinstance(v, float) and not np.isnan(v) else "-"

    lines = ["| benchmark | backbone | ID hdc Kmin→Kmax | ID linear Kmin→Kmax | ID mlp Kmax | "
             "ID proto Kmax | Δ(10k-4k) | cluster feat/hdc @Kmin,N=20 | "
             "AUROC hdc Kmin→Kmax | 5-shot hdc |",
             "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in rows:
        hp = r["h_primary"]
        auroc = f"{f(r, f'auroc_{hp}_nmin')} → {f(r, f'auroc_{hp}_nmax')}"
        lines.append(
            f"| {r['bench']} | {r['backbone']} | "
            f"{f(r, f'id_{hp}_kmin')} → {f(r, f'id_{hp}_kmax')} | "
            f"{f(r, 'id_linear_kmin')} → {f(r, 'id_linear_kmax')} | "
            f"{f(r, 'id_mlp_kmax')} | {f(r, 'id_proto_kmax')} | "
            f"{f(r, 'dim_delta')} | "
            f"{f(r, 'cluster_feat_n')}/{f(r, f'cluster_{hp}_n')} | "
            f"{auroc} | {f(r, f'fs5_{hp}')} |")
    save_md("summary_all.md", "Cross-benchmark summary (means over seeds)\n\n" + "\n".join(lines))
    pd.DataFrame(rows).to_csv(os.path.join(TABLE_DIR, "summary_all.csv"), index=False)


def main():
    synthetic_report()
    image_report("images/cifar100", "cifar100", k_series_n=20, n_series_k=50)
    image_report("images/tinyimagenet", "tinyimagenet", k_series_n=20, n_series_k=100)
    pretrain_report()
    cross_summary()


if __name__ == "__main__":
    main()
