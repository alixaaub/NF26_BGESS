"""
Dashboard CO₂ BGES — alimenté dynamiquement depuis export/warehouse/.
Préfère les snapshots final/ ou latest/ ; sinon reconstruit depuis les dossiers journaliers.
Lancer : python dashboard.py
"""

from __future__ import annotations

import re
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import dash
from dash import Dash, Input, Output, State, callback, dcc, html
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ── Chemins & constantes ─────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent
WAREHOUSE_DIR = ROOT / "export" / "warehouse"
SNAPSHOT_DIRS = ("final", "latest")
DATE_DIR_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
PK_DIM = {"DIM_MISSION": "ID_MISSION", "DIM_MATERIEL": "ID_MATERIEL"}
PK_FAIT = {"FAIT_MISSION": "ID_MISSION", "FAIT_MATERIEL": "ID_MATERIEL"}
DATE_COL = {"FAIT_MISSION": "ID_DATE_MISSION", "FAIT_MATERIEL": "ID_DATE_ACHAT"}
PLOTLY_TEMPLATE = "plotly_dark"
KG_TO_TCO2E = 1000.0

SECTEURS_ORDRE = [
    "Ingénieur Data",
    "Ingénieur Informaticien",
    "Cadre",
    "Economiste",
    "DRH",
]

TRANSPORT_ORDER = ["Avion", "Train", "Taxi", "Transports en commun", "Non renseigné"]


# ── Chargement entrepôt ──────────────────────────────────────────────────────


def _read_csv(path: Path, parse_dates: list[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if parse_dates:
        for c in parse_dates:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


def _snapshot_path(table: str) -> Path | None:
    for sub in SNAPSHOT_DIRS:
        p = WAREHOUSE_DIR / sub / f"{table}.csv"
        if p.exists():
            return p
    return None


def _list_day_dirs() -> list[Path]:
    if not WAREHOUSE_DIR.is_dir():
        return []
    return sorted(
        p for p in WAREHOUSE_DIR.iterdir()
        if p.is_dir() and DATE_DIR_RE.match(p.name)
    )


def _load_table_snapshot_or_daily(table: str) -> pd.DataFrame:
    snap = _snapshot_path(table)
    dates = DATE_COL.get(table)
    parse = [dates] if dates else None
    if snap:
        return _read_csv(snap, parse_dates=parse)

    parts: list[pd.DataFrame] = []
    for day_dir in _list_day_dirs():
        f = day_dir / f"{table}.csv"
        if f.exists():
            parts.append(_read_csv(f, parse_dates=parse))
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    pk = PK_DIM.get(table) or PK_FAIT.get(table)
    if pk and pk in out.columns:
        out = out.drop_duplicates(subset=[pk], keep="last")
    return out


@lru_cache(maxsize=1)
def load_warehouse() -> dict:
    """Charge les 5 tables ; cache invalidé via refresh_warehouse()."""
    dim_pers = _load_table_snapshot_or_daily("DIM_PERSONNEL")
    if dim_pers.empty:
        init_p = WAREHOUSE_DIR / "init" / "DIM_PERSONNEL.csv"
        dim_pers = _read_csv(init_p)

    tables = {
        "DIM_PERSONNEL": dim_pers,
        "FAIT_MISSION": _load_table_snapshot_or_daily("FAIT_MISSION"),
        "DIM_MISSION": _load_table_snapshot_or_daily("DIM_MISSION"),
        "FAIT_MATERIEL": _load_table_snapshot_or_daily("FAIT_MATERIEL"),
        "DIM_MATERIEL": _load_table_snapshot_or_daily("DIM_MATERIEL"),
    }
    meta = {
        "loaded_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": "snapshot" if _snapshot_path("FAIT_MISSION") else "daily_folders",
        "day_folders": len(_list_day_dirs()),
    }
    return {"tables": tables, "meta": meta}


def refresh_warehouse() -> dict:
    load_warehouse.cache_clear()
    return load_warehouse()


def _to_tco2e(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0) / KG_TO_TCO2E


def _ensure_id_site(df: pd.DataFrame) -> pd.DataFrame:
    """Une seule colonne ID_SITE (fusion fait / personnel peut créer _x / _y)."""
    if df.empty:
        return df
    out = df.copy()
    if "ID_SITE" not in out.columns:
        for col in ("ID_SITE_x", "ID_SITE_y"):
            if col in out.columns:
                out["ID_SITE"] = out[col] if "ID_SITE" not in out.columns else out["ID_SITE"].fillna(out[col])
    else:
        for col in ("ID_SITE_x", "ID_SITE_y"):
            if col in out.columns:
                out["ID_SITE"] = out["ID_SITE"].fillna(out[col])
    if "ID_SITE" not in out.columns and "ID_PERSONNEL" in out.columns:
        out["ID_SITE"] = out["ID_PERSONNEL"].astype(str).str.split("_").str[1]
    drop = [c for c in out.columns if c in ("ID_SITE_x", "ID_SITE_y")]
    return out.drop(columns=drop, errors="ignore")


def _group_sum(df: pd.DataFrame, group_col: str, val_col: str = "tCO2e") -> pd.Series:
    """groupby + sum tolérant aux colonnes absentes."""
    if df.empty or group_col not in df.columns or val_col not in df.columns:
        return pd.Series(dtype=float)
    return df.groupby(group_col, dropna=False)[val_col].sum()


def build_analytical_frames(wh: dict) -> dict:
    t = wh["tables"]
    dim_pers = t["DIM_PERSONNEL"].copy()

    # ── Missions
    missions = t["FAIT_MISSION"].merge(
        t["DIM_MISSION"], on="ID_MISSION", how="left", suffixes=("", "_dim")
    )
    if "ID_DATE_MISSION" in missions.columns:
        missions["DATE"] = pd.to_datetime(missions["ID_DATE_MISSION"], errors="coerce")
    missions["tCO2e"] = _to_tco2e(missions.get("IMPACT", pd.Series(dtype=float)))
    missions["MOIS"] = missions["DATE"].dt.to_period("M").astype(str)
    pers_join = ["ID_PERSONNEL", "FONCTION_PERSONNEL"]
    if "ID_SITE" not in missions.columns and "ID_SITE" in dim_pers.columns:
        pers_join.append("ID_SITE")
    missions = missions.merge(dim_pers[pers_join], on="ID_PERSONNEL", how="left")
    missions = _ensure_id_site(missions)
    if "TRANSPORT" in missions.columns:
        missions["TRANSPORT"] = missions["TRANSPORT"].fillna("Non renseigné").astype(str)

    dest_cols = ["VILLE_DESTINATION", "PAYS_DESTINATION"]
    if all(c in missions.columns for c in dest_cols):
        missions["DESTINATION"] = (
            missions["VILLE_DESTINATION"].fillna("?").astype(str)
            + " ("
            + missions["PAYS_DESTINATION"].fillna("?").astype(str)
            + ")"
        )
    else:
        missions["DESTINATION"] = "Inconnue"

    # ── Matériel
    materiel = t["FAIT_MATERIEL"].merge(
        t["DIM_MATERIEL"], on="ID_MATERIEL", how="left", suffixes=("", "_dim")
    )
    if "ID_DATE_ACHAT" in materiel.columns:
        materiel["DATE"] = pd.to_datetime(materiel["ID_DATE_ACHAT"], errors="coerce")
    materiel["tCO2e"] = _to_tco2e(materiel.get("IMPACT", pd.Series(dtype=float)))
    materiel["MOIS"] = materiel["DATE"].dt.to_period("M").astype(str)
    mat_join = ["ID_PERSONNEL", "FONCTION_PERSONNEL"]
    if "ID_SITE" not in materiel.columns and "ID_SITE" in dim_pers.columns:
        mat_join.append("ID_SITE")
    materiel = materiel.merge(dim_pers[mat_join], on="ID_PERSONNEL", how="left")
    materiel = _ensure_id_site(materiel)

    if "TYPE" in materiel.columns:
        materiel["TYPE"] = materiel["TYPE"].fillna("Non renseigné")

    impact_raw = t["DIM_MATERIEL"].get("IMPACT", pd.Series(dtype=float))
    missing_pct = (
        100.0 * pd.to_numeric(impact_raw, errors="coerce").isna().sum() / len(impact_raw)
        if len(impact_raw) > 0
        else 0.0
    )

    all_dates = pd.concat(
        [
            missions["DATE"].dropna() if "DATE" in missions.columns else pd.Series(dtype="datetime64[ns]"),
            materiel["DATE"].dropna() if "DATE" in materiel.columns else pd.Series(dtype="datetime64[ns]"),
        ],
        ignore_index=True,
    )
    date_min = all_dates.min() if not all_dates.empty else pd.Timestamp("2026-04-29")
    date_max = all_dates.max() if not all_dates.empty else pd.Timestamp("2026-11-14")

    return {
        "missions": missions,
        "materiel": materiel,
        "dim_personnel": dim_pers,
        "missing_materiel_pct": missing_pct,
        "date_min": date_min,
        "date_max": date_max,
        "meta": wh["meta"],
    }


def filter_by_period(df: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    if df.empty or "DATE" not in df.columns:
        return df
    out = df.copy()
    if start:
        out = out[out["DATE"] >= pd.Timestamp(start)]
    if end:
        out = out[out["DATE"] <= pd.Timestamp(end) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)]
    return out


def _apply_mission_filters(
    m: pd.DataFrame,
    *,
    site: str | None = None,
    typ: str | None = None,
    transport: str | None = None,
) -> pd.DataFrame:
    """Filtres site, type de mission et transport (onglet Missions)."""
    out = m
    if site and "ID_SITE" in out.columns:
        out = out[out["ID_SITE"] == site]
    if typ and "TYPE_MISSION" in out.columns:
        out = out[out["TYPE_MISSION"] == typ]
    if transport and "TRANSPORT" in out.columns:
        out = out[out["TRANSPORT"] == transport]
    return out


def _fig_bar_chart(
    df: pd.DataFrame,
    cat_col: str,
    val_col: str,
    title: str,
    *,
    horizontal: bool = False,
    top_n: int | None = None,
    log_value_axis: bool = False,
) -> go.Figure:
    """Barres via graph_objects (évite bug plotly.express + Python 3.14)."""
    if df.empty or cat_col not in df.columns or val_col not in df.columns:
        return _empty_fig()
    d = df[[cat_col, val_col]].copy()
    d[cat_col] = (
        d[cat_col]
        .astype(str)
        .str.strip()
        .replace({"nan": "Non renseigné", "None": "Non renseigné", "": "Non renseigné"})
    )
    d[val_col] = pd.to_numeric(d[val_col], errors="coerce").fillna(0)
    d = d[d[val_col] > 0]
    if d.empty:
        return _empty_fig()
    if top_n:
        d = d.nlargest(top_n, val_col)
    d = d.sort_values(val_col, ascending=horizontal)
    labels = d[val_col].map(lambda v: f"{v:,.2f}")

    if horizontal:
        fig = go.Figure(
            go.Bar(
                x=d[val_col].tolist(),
                y=d[cat_col].tolist(),
                orientation="h",
                text=labels.tolist(),
                textposition="outside",
            )
        )
        fig.update_layout(xaxis_title=f"{val_col}" + (" (log)" if log_value_axis else ""), yaxis_title="")
        if log_value_axis:
            fig.update_xaxes(type="log")
        fig.update_layout(margin=dict(l=200))
    else:
        fig = go.Figure(
            go.Bar(
                x=d[cat_col].tolist(),
                y=d[val_col].tolist(),
                orientation="v",
                text=labels.tolist(),
                textposition="outside",
            )
        )
        fig.update_layout(xaxis_title=cat_col, yaxis_title=val_col)
        fig.update_xaxes(tickangle=-40)
        fig.update_layout(margin=dict(b=140))

    fig.update_layout(template=PLOTLY_TEMPLATE, title=title, height=420)
    return fig


def _fig_top_missions_chart(
    m: pd.DataFrame, top_n: int = 10, *, filter_note: str = ""
) -> go.Figure:
    """Top missions : trajet + date + ID sur l'axe Y, valeur à l'intérieur de la barre."""
    if m.empty or "ID_MISSION" not in m.columns or "tCO2e" not in m.columns:
        return _empty_fig("Aucune mission pour les filtres sélectionnés")

    d = m.drop_duplicates(subset=["ID_MISSION"]).copy()
    d["tCO2e"] = pd.to_numeric(d["tCO2e"], errors="coerce").fillna(0)
    d = d[d["tCO2e"] > 0]
    if d.empty:
        return _empty_fig()

    d = d.nlargest(top_n, "tCO2e").sort_values("tCO2e", ascending=True).reset_index(drop=True)
    n = len(d)
    depart = (
        d["VILLE_DEPART"].fillna("?").astype(str)
        if "VILLE_DEPART" in d.columns
        else pd.Series("?", index=d.index)
    )
    dest = (
        d["VILLE_DESTINATION"].fillna("?").astype(str)
        if "VILLE_DESTINATION" in d.columns
        else pd.Series("?", index=d.index)
    )
    if "DATE" in d.columns:
        dates = pd.to_datetime(d["DATE"], errors="coerce")
    elif "ID_DATE_MISSION" in d.columns:
        dates = pd.to_datetime(d["ID_DATE_MISSION"], errors="coerce")
    else:
        dates = pd.Series(pd.NaT, index=d.index)
    date_str = dates.dt.strftime("%d/%m/%Y").fillna("?")

    y_pos = list(range(n))
    ticktext = [
        f"{depart.iloc[i]} → {dest.iloc[i]}<br>{date_str.iloc[i]}<br>{d['ID_MISSION'].iloc[i]}"
        for i in range(n)
    ]
    labels = d["tCO2e"].map(lambda v: f"{v:,.2f} tCO2e")
    xmax = float(d["tCO2e"].max())
    route_lens = (depart + " → " + dest).str.len()
    left_margin = int(min(420, max(220, 40 + route_lens.max() * 6.5)))

    fig = go.Figure(
        go.Bar(
            x=d["tCO2e"].tolist(),
            y=y_pos,
            orientation="h",
            text=labels.tolist(),
            textposition="inside",
            insidetextanchor="end",
            textfont=dict(color="white", size=11),
            cliponaxis=False,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "%{customdata[1]} → %{customdata[2]}<br>"
                "Date : %{customdata[3]}<br>"
                "Émissions : %{x:,.2f} tCO2e<extra></extra>"
            ),
            customdata=list(
                zip(
                    d["ID_MISSION"].astype(str),
                    depart.tolist(),
                    dest.tolist(),
                    date_str.tolist(),
                )
            ),
        )
    )
    title = "Top 10 missions les plus émettrices (tCO2e)"
    if filter_note:
        title += f" — {filter_note}"
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=title,
        height=78 * n + 120,
        bargap=0.42,
        margin=dict(l=left_margin, r=40, t=56, b=48),
        xaxis_title="tCO2e",
        yaxis_title="",
    )
    fig.update_xaxes(range=[0, xmax * 1.08])
    fig.update_yaxes(
        tickmode="array",
        tickvals=y_pos,
        ticktext=ticktext,
        tickfont=dict(size=10),
        automargin=True,
    )
    return fig


def _fig_transport_emissions(by_transport: pd.DataFrame) -> go.Figure:
    """Barres par transport : échelle log car Avion domine très largement les autres."""
    if by_transport.empty:
        return _empty_fig()
    df = by_transport.copy()
    df = df[df["tCO2e"] > 0]
    if df.empty:
        return _empty_fig("Aucune émission calculée pour ces transports")

    order = [t for t in TRANSPORT_ORDER if t in df["TRANSPORT"].astype(str).values]
    order += [t for t in df["TRANSPORT"].astype(str).unique() if t not in order]
    df["TRANSPORT"] = pd.Categorical(df["TRANSPORT"].astype(str), categories=order, ordered=True)
    df = df.sort_values("TRANSPORT")
    df["label"] = df["tCO2e"].map(lambda v: f"{v:,.2f}")

    fig = px.bar(
        df,
        x="tCO2e",
        y="TRANSPORT",
        orientation="h",
        title="Émissions par type de transport (tCO2e, échelle log)",
        template=PLOTLY_TEMPLATE,
        text="label",
        log_x=True,
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(xaxis_title="tCO2e (log)", yaxis_title="", margin=dict(l=180))
    return fig


def _empty_fig(message: str = "Aucune donnée pour cette sélection") -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=message,
        xaxis={"visible": False},
        yaxis={"visible": False},
        annotations=[
            dict(text=message, xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font=dict(size=14))
        ],
        height=360,
    )
    return fig


def kpi_card(label: str, value: str, sub: str = "") -> html.Div:
    return html.Div(
        className="kpi-card",
        children=[
            html.Div(label, className="kpi-label"),
            html.Div(value, className="kpi-value"),
            html.Div(sub, className="kpi-sub") if sub else None,
        ],
    )


# ── Application Dash ─────────────────────────────────────────────────────────

def create_app() -> Dash:
    wh = load_warehouse()
    data = build_analytical_frames(wh)
    d0, d1 = data["date_min"].date(), data["date_max"].date()

    app = Dash(__name__, title="BGES — Empreinte carbone", suppress_callback_exceptions=True)
    app.layout = html.Div(
        className="app-container",
        style={"backgroundColor": "#111", "minHeight": "100vh", "color": "#eee", "fontFamily": "system-ui"},
        children=[
            html.Div(
                className="header",
                style={"padding": "16px 24px", "borderBottom": "1px solid #333"},
                children=[
                    html.H1("Dashboard empreinte carbone BGES", style={"margin": 0, "fontSize": "1.5rem"}),
                    html.P(
                        id="meta-label",
                        children=(
                            f"Dernière charge : {data['meta']['loaded_at']} — "
                            f"source : {data['meta']['source']} ({data['meta']['day_folders']} jours détectés)"
                        ),
                        style={"color": "#888", "fontSize": "0.85rem"},
                    ),
                    html.Div(
                        style={"display": "flex", "flexWrap": "wrap", "gap": "16px", "alignItems": "center", "marginTop": "12px"},
                        children=[
                            html.Label("Période globale :", style={"fontWeight": 600}),
                            dcc.DatePickerRange(
                                id="global-dates",
                                min_date_allowed=d0,
                                max_date_allowed=d1,
                                start_date=d0,
                                end_date=d1,
                                display_format="YYYY-MM-DD",
                            ),
                            html.Button("Actualiser les données", id="btn-refresh", n_clicks=0, style={"padding": "8px 16px"}),
                        ],
                    ),
                ],
            ),
            dcc.Store(id="refresh-token", data=0),
            dcc.Tabs(
                id="tabs",
                value="tab-global",
                colors={"border": "#333", "primary": "#4e9af1", "background": "#1a1a1a"},
                children=[
                    dcc.Tab(label="Vue globale", value="tab-global", children=[html.Div(id="tab-global-content", style={"padding": "16px"})]),
                    dcc.Tab(
                        label="Missions & déplacements",
                        value="tab-missions",
                        children=[
                            html.Div(
                                style={"padding": "16px"},
                                children=[
                                    html.Div(
                                        id="missions-filters",
                                        style={"display": "flex", "flexWrap": "wrap", "gap": "16px", "marginBottom": "16px"},
                                        children=[
                                            html.Div([
                                                html.Label("Site"),
                                                dcc.Dropdown(id="filtre-site", options=[], value=None, placeholder="Tous les sites", clearable=True),
                                            ], style={"minWidth": "200px"}),
                                            html.Div([
                                                html.Label("Type de mission"),
                                                dcc.Dropdown(id="filtre-type-mission", options=[], value=None, placeholder="Tous les types", clearable=True),
                                            ], style={"minWidth": "220px"}),
                                            html.Div([
                                                html.Label("Transport"),
                                                dcc.Dropdown(id="filtre-transport", options=[], value=None, placeholder="Tous les transports", clearable=True),
                                            ], style={"minWidth": "220px"}),
                                            html.Div([
                                                html.Label("Période missions"),
                                                dcc.DatePickerRange(
                                                    id="filtre-dates-missions",
                                                    start_date=d0,
                                                    end_date=d1,
                                                    display_format="YYYY-MM-DD",
                                                ),
                                            ]),
                                        ],
                                    ),
                                    html.Div(id="tab-missions-content"),
                                ],
                            ),
                        ],
                    ),
                    dcc.Tab(label="Matériel informatique", value="tab-materiel", children=[html.Div(id="tab-materiel-content", style={"padding": "16px"})]),
                    dcc.Tab(label="Secteur d'activité", value="tab-secteur", children=[html.Div(id="tab-secteur-content", style={"padding": "16px"})]),
                ],
            ),
        ],
    )

    # ── Callbacks ────────────────────────────────────────────────────────────

    @callback(
        Output("refresh-token", "data"),
        Output("global-dates", "min_date_allowed"),
        Output("global-dates", "max_date_allowed"),
        Output("meta-label", "children"),
        Input("btn-refresh", "n_clicks"),
        prevent_initial_call=False,
    )
    def on_refresh(n_clicks):
        if n_clicks:
            refresh_warehouse()
        d = build_analytical_frames(load_warehouse())
        meta_txt = (
            f"Dernière charge : {d['meta']['loaded_at']} — "
            f"source : {d['meta']['source']} ({d['meta']['day_folders']} jours détectés)"
        )
        return (
            n_clicks or 0,
            d["date_min"].date(),
            d["date_max"].date(),
            meta_txt,
        )

    @callback(
        Output("tab-global-content", "children"),
        Input("global-dates", "start_date"),
        Input("global-dates", "end_date"),
        Input("refresh-token", "data"),
    )
    def render_tab_global(start, end, _token):
        d = build_analytical_frames(load_warehouse())
        m = _ensure_id_site(filter_by_period(d["missions"], start, end))
        mat = _ensure_id_site(filter_by_period(d["materiel"], start, end))
        t_m = m["tCO2e"].sum()
        t_mat = mat["tCO2e"].sum()
        t_tot = t_m + t_mat

        # Bar par site
        site_m = _group_sum(m, "ID_SITE")
        site_mat = _group_sum(mat, "ID_SITE")
        site_all = site_m.add(site_mat, fill_value=0).sort_values(ascending=False).reset_index()
        site_all.columns = ["ID_SITE", "tCO2e"]

        if site_all.empty:
            fig_site = _empty_fig()
        else:
            fig_site = px.bar(site_all, x="ID_SITE", y="tCO2e", title="Émissions par site (tCO2e)", template=PLOTLY_TEMPLATE)
            fig_site.update_layout(yaxis_title="tCO2e")

        fig_pie = go.Figure(
            data=[
                go.Pie(
                    labels=["Missions", "Matériel informatique"],
                    values=[max(t_m, 0), max(t_mat, 0)],
                    hole=0.4,
                )
            ]
        )
        fig_pie.update_layout(title="Répartition missions vs matériel", template=PLOTLY_TEMPLATE)

        return html.Div(
            [
                html.Div(
                    style={"display": "flex", "gap": "16px", "flexWrap": "wrap", "marginBottom": "24px"},
                    children=[
                        kpi_card("Total global", f"{t_tot:,.2f}", "tCO2e"),
                        kpi_card("Missions", f"{t_m:,.2f}", "tCO2e"),
                        kpi_card("Matériel", f"{t_mat:,.2f}", "tCO2e"),
                    ],
                ),
                html.Div(
                    style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "16px"},
                    children=[
                        dcc.Graph(figure=fig_site),
                        dcc.Graph(figure=fig_pie),
                    ],
                ),
            ]
        )

    @callback(
        Output("filtre-site", "options"),
        Output("filtre-type-mission", "options"),
        Output("filtre-transport", "options"),
        Output("filtre-dates-missions", "min_date_allowed"),
        Output("filtre-dates-missions", "max_date_allowed"),
        Input("global-dates", "start_date"),
        Input("global-dates", "end_date"),
        Input("refresh-token", "data"),
    )
    def update_missions_filter_options(start, end, _token):
        d = build_analytical_frames(load_warehouse())
        m_all = _ensure_id_site(filter_by_period(d["missions"], start, end))
        sites = sorted(m_all["ID_SITE"].dropna().unique().astype(str)) if not m_all.empty else []
        types = (
            sorted(m_all["TYPE_MISSION"].dropna().unique().astype(str))
            if not m_all.empty and "TYPE_MISSION" in m_all.columns
            else []
        )
        transports = (
            sorted(m_all["TRANSPORT"].dropna().unique().astype(str))
            if not m_all.empty and "TRANSPORT" in m_all.columns
            else []
        )
        return (
            [{"label": s, "value": s} for s in sites],
            [{"label": t, "value": t} for t in types],
            [{"label": t, "value": t} for t in transports],
            d["date_min"].date(),
            d["date_max"].date(),
        )

    @callback(
        Output("tab-missions-content", "children"),
        Input("global-dates", "start_date"),
        Input("global-dates", "end_date"),
        Input("filtre-site", "value"),
        Input("filtre-type-mission", "value"),
        Input("filtre-transport", "value"),
        Input("filtre-dates-missions", "start_date"),
        Input("filtre-dates-missions", "end_date"),
        Input("refresh-token", "data"),
    )
    def render_tab_missions(g_start, g_end, site, typ, transport, m_start, m_end, _token):
        d = build_analytical_frames(load_warehouse())
        m_period = _ensure_id_site(filter_by_period(d["missions"], g_start, g_end))
        m_period = filter_by_period(m_period, m_start or g_start, m_end or g_end)
        m = _apply_mission_filters(m_period, site=site, typ=typ, transport=transport)
        m_by_transport = _apply_mission_filters(m_period, site=site, typ=typ)

        if m.empty and m_by_transport.empty:
            return html.Div([dcc.Graph(figure=_empty_fig()) for _ in range(3)])

        monthly = _group_sum(m, "MOIS").reset_index(name="tCO2e").sort_values("MOIS")
        line_title = "Évolution mensuelle — missions (tCO2e)"
        filter_bits = [x for x in (typ, transport) if x]
        if filter_bits:
            line_title += " — " + ", ".join(filter_bits)
        fig_line = (
            px.line(monthly, x="MOIS", y="tCO2e", title=line_title, markers=True, template=PLOTLY_TEMPLATE)
            if not monthly.empty
            else _empty_fig("Aucune donnée pour les filtres sélectionnés")
        )
        by_transport = _group_sum(m_by_transport, "TRANSPORT").reset_index(name="tCO2e")
        fig_transport = _fig_transport_emissions(by_transport)
        filter_note = " · ".join(x for x in (site, typ, transport) if x)
        fig_top_missions = _fig_top_missions_chart(m, top_n=10, filter_note=filter_note)
        return html.Div(
            style={"display": "grid", "gridTemplateColumns": "1fr", "gap": "16px"},
            children=[
                dcc.Graph(figure=fig_line),
                dcc.Graph(figure=fig_transport),
                dcc.Graph(figure=fig_top_missions, style={"width": "100%"}),
            ],
        )

    @callback(
        Output("tab-materiel-content", "children"),
        Input("global-dates", "start_date"),
        Input("global-dates", "end_date"),
        Input("refresh-token", "data"),
    )
    def render_tab_materiel(start, end, _token):
        d = build_analytical_frames(load_warehouse())
        mat = _ensure_id_site(filter_by_period(d["materiel"], start, end))
        missing = d["missing_materiel_pct"]

        if mat.empty:
            return html.Div([
                html.Div(
                    className="kpi-card",
                    style={"marginBottom": "16px", "padding": "12px", "background": "#222", "borderRadius": "8px"},
                    children=f"Données IMPACT manquantes (référentiel matériel) : {missing:.1f} %",
                ),
                dcc.Graph(figure=_empty_fig()),
            ])

        by_type = _group_sum(mat, "TYPE").reset_index(name="tCO2e")
        fig_type = _fig_bar_chart(
            by_type, "TYPE", "tCO2e", "Émissions par type de matériel (tCO2e)", horizontal=True
        )

        monthly = _group_sum(mat, "MOIS").reset_index(name="tCO2e").sort_values("MOIS")
        fig_month = (
            px.line(monthly, x="MOIS", y="tCO2e", title="Évolution mensuelle des achats (tCO2e)", markers=True, template=PLOTLY_TEMPLATE)
            if not monthly.empty
            else _empty_fig()
        )

        by_site = _group_sum(mat, "ID_SITE").reset_index(name="tCO2e")
        fig_site = _fig_bar_chart(by_site, "ID_SITE", "tCO2e", "Émissions matériel par site (tCO2e)")

        alert_color = "#e74c3c" if missing > 5 else "#2ecc71"
        indicator = html.Div(
            style={
                "padding": "16px",
                "marginBottom": "16px",
                "borderRadius": "8px",
                "background": "#222",
                "borderLeft": f"6px solid {alert_color}",
            },
            
        )

        return html.Div([
            indicator,
            html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "16px"}, children=[
                dcc.Graph(figure=fig_type),
                dcc.Graph(figure=fig_month),
            ]),
            dcc.Graph(figure=fig_site),
        ])

    @callback(
        Output("tab-secteur-content", "children"),
        Input("global-dates", "start_date"),
        Input("global-dates", "end_date"),
        Input("refresh-token", "data"),
    )
    def render_tab_secteur(start, end, _token):
        d = build_analytical_frames(load_warehouse())
        m = _ensure_id_site(filter_by_period(d["missions"], start, end))
        mat = _ensure_id_site(filter_by_period(d["materiel"], start, end))

        em_m = _group_sum(m, "FONCTION_PERSONNEL")
        em_mat = _group_sum(mat, "FONCTION_PERSONNEL")
        em = em_m.add(em_mat, fill_value=0).reindex(SECTEURS_ORDRE).fillna(0).reset_index()
        em.columns = ["FONCTION_PERSONNEL", "tCO2e"]
        em = em[em["tCO2e"] > 0]

        if em.empty:
            return dcc.Graph(figure=_empty_fig("Aucune émission par secteur sur la période"))

        fig_sect = px.bar(em, x="FONCTION_PERSONNEL", y="tCO2e", title="Émissions par secteur d'activité (tCO2e)", template=PLOTLY_TEMPLATE)

        # Heatmap secteur × site
        frames = []
        if not m.empty:
            frames.append(m.assign(source="mission"))
        if not mat.empty:
            frames.append(mat.assign(source="materiel"))
        if frames:
            comb = _ensure_id_site(pd.concat(frames, ignore_index=True))
            heat = comb.pivot_table(
                index="FONCTION_PERSONNEL", columns="ID_SITE", values="tCO2e", aggfunc="sum", fill_value=0
            ) if "ID_SITE" in comb.columns else pd.DataFrame()
            if not heat.empty:
                heat = heat.reindex(SECTEURS_ORDRE).dropna(how="all")
                fig_heat = px.imshow(
                    heat, labels=dict(x="Site", y="Secteur", color="tCO2e"),
                    title="Secteur × site (tCO2e)", template=PLOTLY_TEMPLATE, aspect="auto",
                )
            else:
                fig_heat = _empty_fig("ID_SITE indisponible pour la heatmap")
        else:
            fig_heat = _empty_fig()

        emp_m = _group_sum(m, "ID_PERSONNEL")
        emp_mat = _group_sum(mat, "ID_PERSONNEL")
        emp = emp_m.add(emp_mat, fill_value=0).nlargest(10).reset_index(name="tCO2e")
        fig_emp = (
            px.bar(emp.sort_values("tCO2e"), x="tCO2e", y="ID_PERSONNEL", orientation="h",
                   title="Top 10 émetteurs (ID anonymisé)", template=PLOTLY_TEMPLATE)
            if not emp.empty
            else _empty_fig()
        )

        return html.Div([
            dcc.Graph(figure=fig_sect),
            html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "16px"}, children=[
                dcc.Graph(figure=fig_heat),
                dcc.Graph(figure=fig_emp),
            ]),
        ])

    return app


# ── Styles KPI injectés ──────────────────────────────────────────────────────

KPI_STYLE = """
.kpi-card { background: #1e1e1e; border-radius: 8px; padding: 16px 24px; min-width: 160px; border: 1px solid #333; }
.kpi-label { font-size: 0.8rem; color: #aaa; text-transform: uppercase; }
.kpi-value { font-size: 1.8rem; font-weight: 700; color: #4e9af1; margin-top: 4px; }
.kpi-sub { font-size: 0.75rem; color: #666; }
"""


if __name__ == "__main__":
    app = create_app()
    app.index_string = app.index_string.replace("</head>", f"<style>{KPI_STYLE}</style></head>")
    print(f"Entrepôt : {WAREHOUSE_DIR}")
    print("Ouvrir http://127.0.0.1:8050")
    app.run(debug=True, host="127.0.0.1", port=8050)
