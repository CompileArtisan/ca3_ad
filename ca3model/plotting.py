"""Plotly figure builders (spec 6.9). No Streamlit imports; nothing is displayed here."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .patterns import to_image

# Okabe-Ito colour-blind-safe palette
COLORS = {"healthy": "#0072B2", "synaptic": "#E69F00", "neuron": "#D55E00",
          "combined": "#CC79A7", "matched_synaptic": "#009E73"}
_ALIASES = {"synaptic_only": "synaptic", "neuron_only": "neuron", "healthy_learn": "healthy",
            "damage_then_learn": "combined", "learn_then_damage": "synaptic", "damaged": "combined"}


def color_for(name: str) -> str:
    return COLORS.get(_ALIASES.get(name, name), "#666666")


def _rgba(hex_color: str, a: float) -> str:
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{a})"


def _img(vec, alive, shape):
    """+1 -> 1 (white), -1 -> 0 (black), dead/0 -> 0.5 (grey)."""
    v = np.where(np.asarray(vec) == 0, 0.5, (np.asarray(vec) + 1) / 2.0)
    return to_image(v, shape)


def fig_pattern_panel(original, cue, healthy_recon, damaged_recon, shape=None,
                      titles=("Original", "Corrupted cue", "Healthy reconstruction",
                              "Damaged reconstruction")) -> go.Figure:
    """1x4 black/white heatmaps; neurons that are lost (state 0) are grey."""
    fig = make_subplots(rows=1, cols=4, subplot_titles=list(titles), horizontal_spacing=0.03)
    for c, vec in enumerate([original, cue, healthy_recon, damaged_recon], start=1):
        fig.add_trace(go.Heatmap(z=_img(vec, None, shape), zmin=0, zmax=1, showscale=False,
                                 colorscale=[[0, "black"], [0.5, "#999999"], [1, "white"]],
                                 hovertemplate="row %{y}, col %{x}<extra></extra>"), row=1, col=c)
        fig.update_yaxes(autorange="reversed", visible=False, scaleanchor=f"x{c}" if c > 1 else "x",
                         row=1, col=c)
        fig.update_xaxes(visible=False, row=1, col=c)
    fig.update_layout(height=280, margin=dict(l=10, r=10, t=40, b=10))
    return fig


def _wm(fig, W, alive, title, row=1, col=1, zmax=None):
    zmax = zmax or float(np.abs(W).max() or 1.0)
    fig.add_trace(go.Heatmap(z=W, zmin=-zmax, zmax=zmax, colorscale="RdBu_r", zmid=0,
                             colorbar=dict(title="weight") if col == 1 else None,
                             showscale=(col == 1),
                             hovertemplate="post %{y}, pre %{x}: %{z:.4f}<extra></extra>"),
                  row=row, col=col)
    for i in np.flatnonzero(~alive):  # mark lost neurons as thin grey lines on both axes
        fig.add_shape(type="line", x0=i, x1=i, y0=-0.5, y1=W.shape[0] - 0.5,
                      line=dict(color="rgba(80,80,80,0.35)", width=1), row=row, col=col)
        fig.add_shape(type="line", y0=i, y1=i, x0=-0.5, x1=W.shape[0] - 0.5,
                      line=dict(color="rgba(80,80,80,0.35)", width=1), row=row, col=col)
    fig.update_yaxes(autorange="reversed", row=row, col=col)


def fig_weight_matrix(W_eff, alive, title="Effective weights") -> go.Figure:
    """Diverging heatmap centred on 0; lost neurons marked by grey rows/columns."""
    fig = make_subplots(rows=1, cols=1, subplot_titles=[title])
    _wm(fig, W_eff, alive, title)
    fig.update_layout(height=450, margin=dict(l=10, r=10, t=40, b=10))
    return fig


def fig_weight_compare(W_healthy, W_damaged, alive) -> go.Figure:
    """Healthy vs damaged effective weights on a shared colour scale."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Healthy", "Damaged"], horizontal_spacing=0.06)
    z = float(np.abs(W_healthy).max() or 1.0)
    _wm(fig, W_healthy, np.ones(W_healthy.shape[0], bool), "Healthy", 1, 1, z)
    _wm(fig, W_damaged, alive, "Damaged", 1, 2, z)
    fig.update_layout(height=430, margin=dict(l=10, r=10, t=40, b=10))
    return fig


def fig_weight_hist(W_healthy, W_damaged) -> go.Figure:
    """Histogram of off-diagonal weights (log counts; damaged shows the spike at 0)."""
    off = ~np.eye(W_healthy.shape[0], dtype=bool)
    fig = go.Figure()
    for name, W, col in (("healthy", W_healthy, COLORS["healthy"]), ("damaged", W_damaged, COLORS["combined"])):
        fig.add_trace(go.Histogram(x=W[off], name=name, opacity=0.65, marker_color=col, nbinsx=60))
    fig.update_layout(barmode="overlay", yaxis_type="log", xaxis_title="weight value",
                      yaxis_title="count (log)", height=320, margin=dict(l=10, r=10, t=20, b=10))
    return fig


def fig_energy_trace(trace) -> go.Figure:
    """E(s) = -1/2 s^T W s at start and after each sweep."""
    fig = go.Figure(go.Scatter(x=list(range(len(trace))), y=trace, mode="lines+markers",
                               line=dict(color=COLORS["combined"]),
                               hovertemplate="sweep %{x}: E=%{y:.3f}<extra></extra>"))
    fig.update_layout(xaxis_title="sweep (0 = cue)", yaxis_title="energy E (a.u.)",
                      height=300, margin=dict(l=10, r=10, t=20, b=10))
    return fig


def fig_metric_vs_severity(summary_df, x, y, color="condition", ylabel=None,
                           xlabel="severity (model parameter)") -> go.Figure:
    """Line + 95% CI ribbon. ``y`` is the base metric name (expects y_mean/y_lo/y_hi)."""
    fig = go.Figure()
    for name, g in summary_df.groupby(color, sort=False):
        g = g.sort_values(x)
        c = color_for(str(name))
        lo, hi = g[f"{y}_lo"].to_numpy(), g[f"{y}_hi"].to_numpy()
        if not np.all(np.isnan(lo)):
            fig.add_trace(go.Scatter(x=np.r_[g[x], g[x][::-1]], y=np.r_[hi, lo[::-1]], fill="toself",
                                     fillcolor=_rgba(c, 0.18), line=dict(width=0), hoverinfo="skip",
                                     showlegend=False))
        fig.add_trace(go.Scatter(x=g[x], y=g[f"{y}_mean"], name=str(name), mode="lines+markers",
                                 line=dict(color=c), customdata=np.c_[lo, hi],
                                 hovertemplate=f"{name}<br>{x}=%{{x}}<br>mean=%{{y:.3f}} "
                                               "(95%% CI %{customdata[0]:.3f}-%{customdata[1]:.3f})<extra></extra>"))
    fig.update_layout(xaxis_title=xlabel if x in ("synaptic_severity", "neuron_loss", "level") else x,
                      yaxis_title=ylabel or y, height=380, margin=dict(l=10, r=10, t=20, b=10),
                      legend=dict(orientation="h", y=-0.25))
    return fig


def fig_condition_bars(summary_df, y, label_col="condition", ylabel=None) -> go.Figure:
    """Bars with 95% CI error bars."""
    d = summary_df
    err_hi = (d[f"{y}_hi"] - d[f"{y}_mean"]).fillna(0)
    err_lo = (d[f"{y}_mean"] - d[f"{y}_lo"]).fillna(0)
    fig = go.Figure(go.Bar(x=d[label_col], y=d[f"{y}_mean"], marker_color=[color_for(str(c)) for c in d[label_col]],
                           error_y=dict(type="data", symmetric=False, array=err_hi, arrayminus=err_lo),
                           hovertemplate="%{x}: mean=%{y:.3f}<extra></extra>"))
    fig.update_layout(yaxis_title=ylabel or y, height=340, margin=dict(l=10, r=10, t=20, b=10))
    return fig


def fig_heatmap_grid(grid_df, x, y, z, zlabel=None, zmin=None, zmax=None, colorscale="Viridis") -> go.Figure:
    """Heatmap of z over a (x, y) grid, e.g. success over synaptic severity x neuron loss."""
    piv = grid_df.pivot_table(index=y, columns=x, values=z, aggfunc="mean").sort_index()
    fig = go.Figure(go.Heatmap(z=piv.values, x=piv.columns, y=piv.index, colorscale=colorscale,
                               zmin=zmin, zmax=zmax, colorbar=dict(title=zlabel or z),
                               hovertemplate=f"{x}=%{{x}}<br>{y}=%{{y}}<br>{z}=%{{z:.3f}}<extra></extra>"))
    fig.update_layout(xaxis_title=f"{x} (model parameter)", yaxis_title=f"{y} (model parameter)",
                      height=420, margin=dict(l=10, r=10, t=20, b=10))
    return fig


def fig_new_learning(summary_df, x="level", y="success", group_col="pattern_group") -> go.Figure:
    """New vs old pattern success per condition (columns: condition, pattern_group, level, y_mean/lo/hi)."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=["New patterns", "Old patterns"], shared_yaxes=True)
    for col, grp in ((1, "new"), (2, "old")):
        for name, g in summary_df[summary_df[group_col] == grp].groupby("condition", sort=False):
            g = g.sort_values(x)
            c = color_for(str(name))
            lo, hi = g[f"{y}_lo"].to_numpy(), g[f"{y}_hi"].to_numpy()
            fig.add_trace(go.Scatter(x=g[x], y=g[f"{y}_mean"], name=str(name), mode="lines+markers",
                                     line=dict(color=c), legendgroup=str(name), showlegend=(col == 1),
                                     error_y=dict(type="data", symmetric=False,
                                                  array=np.nan_to_num(hi - g[f"{y}_mean"].to_numpy()),
                                                  arrayminus=np.nan_to_num(g[f"{y}_mean"].to_numpy() - lo),
                                                  thickness=1)), row=1, col=col)
    fig.update_xaxes(title_text="damage level (model parameter)")
    fig.update_yaxes(title_text="retrieval success rate", range=[-0.02, 1.02], col=1)
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.3))
    return fig


def fig_memory_load(summary_df, n_neurons, x="n_patterns", y="success", color="condition",
                    alpha_c=0.138) -> go.Figure:
    """Success vs P with the theoretical alpha_c*N reference line (not an exact prediction)."""
    fig = fig_metric_vs_severity(summary_df, x, y, color, "retrieval success rate", "stored patterns P")
    fig.add_vline(x=alpha_c * n_neurons, line_dash="dash", line_color="#444",
                  annotation_text=f"α_c≈{alpha_c} (P≈{alpha_c * n_neurons:.0f})", annotation_position="top right")
    fig.update_yaxes(range=[-0.02, 1.02])
    return fig
