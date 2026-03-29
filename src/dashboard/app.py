import sys
from pathlib import Path
from html import escape

import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
import plotly.graph_objects as go
import streamlit.components.v1 as components
import joblib

# Ensure project root is on sys.path for "src" imports when run via streamlit
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from build_predicted_table import main as build_predicted_table_main
from src.utils.teams import normalize_team_name


DATA_DIR = Path("data")
PRED_TABLE_PATHS = [
    DATA_DIR / "predicted_table.csv",
    DATA_DIR / "table_2025_26_linkedin.csv",
]
CURRENT_TABLE_PATHS = [
    DATA_DIR / "current_table.csv",
    DATA_DIR / "current_table_2025_26.csv",
]


def _load_first_existing(paths):
    for path in paths:
        if path.exists():
            return path, pd.read_csv(path)
    return None, pd.DataFrame()


def _datetime_series_to_naive(series: pd.Series) -> pd.Series:
    """Force timezone-naive datetimes so sort/compare never mixes naive vs aware."""
    s = pd.to_datetime(series, errors="coerce", utc=True)
    return s.dt.tz_localize(None)


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    if "team" not in df.columns:
        for candidate in ["club", "name", "team_name"]:
            if candidate in df.columns:
                df = df.rename(columns={candidate: "team"})
                break
    if "points" not in df.columns:
        for candidate in ["pts"]:
            if candidate in df.columns:
                df = df.rename(columns={candidate: "points"})
                break
    if "gd" not in df.columns:
        for candidate in ["goal_diff", "goal_difference"]:
            if candidate in df.columns:
                df = df.rename(columns={candidate: "gd"})
                break
    if "position" not in df.columns and "pos" in df.columns:
        df = df.rename(columns={"pos": "position"})
    if "team" in df.columns:
        df["team"] = df["team"].astype(str).map(normalize_team_name)
    return df


def _ensure_position(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "position" in df.columns and df["position"].notna().any():
        return df
    sort_cols = [c for c in ["points", "gd"] if c in df.columns]
    if not sort_cols:
        return df
    df = df.sort_values(sort_cols, ascending=False).reset_index(drop=True)
    df["position"] = range(1, len(df) + 1)
    return df


def _zone_label(pos: int) -> str:
    if pos == 1:
        return "Champion"
    if pos <= 4:
        return "UCL"
    if pos == 5:
        return "UEL"
    if pos == 6:
        return "UECL"
    if pos >= 18:
        return "Relegation"
    return "Mid"


def _add_deltas(pred_df: pd.DataFrame, current_df: pd.DataFrame) -> pd.DataFrame:
    df = pred_df.copy()
    if current_df.empty or "team" not in current_df.columns:
        df["pos_change"] = 0
        return df
    current = current_df.copy()
    current = _ensure_position(_normalize_columns(current))
    current_map = dict(zip(current["team"], current["position"]))
    df["pos_change"] = df["team"].map(current_map).fillna(df["position"]) - df["position"]
    return df


def _build_form_map(raw_matches: pd.DataFrame, team_col: str = "team") -> dict:
    if raw_matches.empty:
        return {}
    raw = raw_matches.copy()
    if "home_team" in raw.columns:
        raw["home_team"] = raw["home_team"].astype(str).map(normalize_team_name)
    if "away_team" in raw.columns:
        raw["away_team"] = raw["away_team"].astype(str).map(normalize_team_name)
    if "date" in raw.columns:
        raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    raw = raw.sort_values("date")

    form_map = {}
    teams = pd.unique(pd.concat([raw["home_team"], raw["away_team"]], ignore_index=True))
    for team in teams:
        team_mask = (raw["home_team"] == team) | (raw["away_team"] == team)
        team_matches = raw[team_mask].tail(5)
        results = []
        points = 0
        for _, row in team_matches.iterrows():
            if pd.isna(row.get("home_score")) or pd.isna(row.get("away_score")):
                continue
            home = row["home_team"] == team
            if row["home_score"] == row["away_score"]:
                results.append("D")
                points += 1
            elif (row["home_score"] > row["away_score"] and home) or (
                row["away_score"] > row["home_score"] and not home
            ):
                results.append("W")
                points += 3
            else:
                results.append("L")
        form_map[team] = {"form": "".join(results), "points": points}
    return form_map


def _delta_label(delta: int) -> str:
    if delta > 0:
        return f"▲ {int(delta)}"
    if delta < 0:
        return f"▼ {abs(int(delta))}"
    return "—"


def _style_table(df: pd.DataFrame):
    def row_style(row):
        zone = row.get("zone", "")
        if zone == "Champion":
            return ["background-color: #fff4c2; color: #b8860b; font-weight: 700"] * len(row)
        if zone == "UCL":
            return ["background-color: #cfe8ff"] * len(row)
        if zone == "UEL":
            return ["background-color: #ffe6cc"] * len(row)
        if zone == "UECL":
            return ["background-color: #e3f7e3"] * len(row)
        if zone == "Relegation":
            return ["background-color: #ffd6d6"] * len(row)
        return [""] * len(row)

    styler = df.style.apply(row_style, axis=1)
    if "Form (last 5)" in df.columns:
        styler = styler.format({"Form (last 5)": lambda v: str(v).split("|", 1)[-1]})
    if "Δ (Change from current position)" in df.columns:
        styler = styler.format(
            {"Δ (Change from current position)": lambda v: str(v).split("|", 1)[-1]}
        )
    return styler


def _confidence_from_points(points: pd.Series) -> pd.Series:
    if points.empty:
        return pd.Series([], dtype=int)
    min_pts = points.min()
    max_pts = points.max()
    if max_pts == min_pts:
        return pd.Series([50] * len(points), index=points.index, dtype=int)
    scaled = 20 + (points - min_pts) * 80 / (max_pts - min_pts)
    return scaled.round().clip(5, 95).astype(int)


def _render_bloomberg_table(
    table_view: pd.DataFrame, metrics: dict, matches_simulated: int
) -> str:
    champ = table_view.iloc[0]["Team"] if not table_view.empty else "TBD"
    top4 = table_view.head(4)["Team"].tolist()
    relegation = table_view.tail(3)["Team"].tolist()
    conf_series = _confidence_from_points(table_view["Pts"])

    def form_pips(form_str: str) -> str:
        pips = []
        for ch in form_str:
            cls = "fp-w" if ch == "W" else "fp-d" if ch == "D" else "fp-l"
            pips.append(f'<div class="fp {cls}"></div>')
        return "".join(pips)

    def mv_arrow(diff: int) -> str:
        if diff > 0:
            return f'<div class="c-mv mv-u">▲{diff}</div>'
        if diff < 0:
            return f'<div class="c-mv mv-d">▼{abs(diff)}</div>'
        return '<div class="c-mv mv-eq">—</div>'

    def gd_class(val: int) -> str:
        return "gd-pos" if val > 0 else "gd-neg" if val < 0 else "gd-zero"

    zone_breaks = {2: "champ", 5: "uel", 7: "conf", 18: "rel"}
    zone_tags = {
        "champ": "Champion",
        "uel": "UEL Entry",
        "conf": "UECL Entry",
        "rel": "Relegation Zone",
    }

    rows_html = []
    pos_change_map = pred_df.set_index("team")["pos_change"].to_dict()
    for idx, row in table_view.iterrows():
        pos = int(row["Pos"])
        delta_val = int(pos_change_map.get(row["Team"], 0))
        zone = str(row.get("Zone", "Mid")).lower()
        zone_class = (
            "ucl" if zone == "ucl" else "uel" if zone == "uel" else "conf" if zone == "uecl" else "rel" if zone == "relegation" else "mid"
        )
        if pos in zone_breaks:
            zb = zone_breaks[pos]
            rows_html.append(
                f'<div class="zone-sep"><div class="zone-tag zt-{zb}">{zone_tags[zb]}</div></div>'
            )
        gd_val = int(row["GD"])
        gd_str = f"+{gd_val}" if gd_val > 0 else f"{gd_val}"
        conf = int(conf_series.loc[idx]) if idx in conf_series.index else 50
        conf_color = "#1a9e5a" if conf >= 70 else "#d4920a" if conf >= 50 else "#e84040"
        form_raw = str(row.get("Form (last 5)", ""))
        form = form_raw.split("|", 1)[-1]
        form_pts = 0
        if "|" in form_raw:
            try:
                form_pts = int(form_raw.split("|", 1)[0])
            except ValueError:
                form_pts = 0
        delta_cell = mv_arrow(delta_val)
        rows_html.append(
            f"""
            <div class="t-row z-{zone_class} {'champ' if pos == 1 else ''}"
                 data-pos="{pos}"
                 data-delta="{delta_val}"
                 data-club="{escape(str(row['Team']))}"
                 data-form="{form_pts}"
                 data-pl="{int(row['Pl'])}"
                 data-w="{int(row['W'])}"
                 data-d="{int(row['D'])}"
                 data-l="{int(row['L'])}"
                 data-gd="{gd_val}"
                 data-pts="{int(row['Pts'])}"
                 data-conf="{conf}">
              <div class="c-pos">{pos}</div>
              {delta_cell}
              <div class="c-team">{escape(str(row['Team']))}</div>
              <div class="c-num">{int(row['Pl'])}</div>
              <div class="c-num">{int(row['W'])}</div>
              <div class="c-num">{int(row['D'])}</div>
              <div class="c-num">{int(row['L'])}</div>
              <div class="c-gd {gd_class(gd_val)}">{gd_str}</div>
              <div class="c-pts">{int(row['Pts'])}</div>
              <div class="form-row">{form_pips(form)}</div>
              <div class="conf-cell">
                <div class="conf-bar-bg"><div class="conf-bar-fg" style="width:{conf}%;background:{conf_color};"></div></div>
                <div class="conf-val">{conf}%</div>
              </div>
            </div>
            """
        )

    accuracy = metrics.get("Accuracy", "—")
    macro_f1 = metrics.get("Macro F1", "—")
    log_loss = metrics.get("Log Loss", "—")

    html = f"""
    <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    .wrap {{
      background: #0d1117;
      border: 1px solid #1e2a38;
      border-radius: 10px;
      overflow: hidden;
      font-family: 'Courier New', monospace;
    }}
    .topbar {{
      background: #111820;
      border-bottom: 1px solid #1e2a38;
      padding: 8px 16px;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }}
    .tb-left {{ display: flex; align-items: center; gap: 10px; }}
    .tb-path {{ font-size: 11px; color: #4a6070; letter-spacing: 0.06em; }}
    .tb-right {{ display: flex; gap: 16px; align-items: center; }}
    .tb-stat {{ text-align: right; }}
    .tb-stat-label {{ font-size: 9px; color: #3a5060; letter-spacing: 0.12em; text-transform: uppercase; }}
    .tb-stat-val {{ font-size: 13px; font-weight: 600; letter-spacing: 0.04em; }}
    .v-amber {{ color: #d4920a; }}
    .v-muted {{ color: #3a5060; }}

    .headline {{
      padding: 14px 18px 10px;
      border-bottom: 1px solid #1a2530;
      display: flex;
      align-items: flex-end;
      justify-content: space-between;
    }}
    .hl-title {{
      font-family: 'Georgia', serif;
      font-size: 20px;
      font-weight: bold;
      color: #c8d8e8;
      letter-spacing: -0.01em;
    }}
    .hl-sub {{
      font-size: 10px;
      color: #d4920a;
      letter-spacing: 0.1em;
      text-transform: uppercase;
      margin-top: 3px;
      font-family: 'Courier New', monospace;
    }}
    .hl-meta {{ display: flex; gap: 20px; }}
    .meta-blk {{ text-align: right; }}
    .meta-label {{ font-size: 9px; color: #3a5060; letter-spacing: 0.1em; text-transform: uppercase; }}
    .meta-val {{ font-size: 18px; color: #c8d8e8; font-weight: bold; }}
    .meta-val span {{ font-size: 10px; color: #4a6070; margin-left: 2px; }}

    .summary-strip {{
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      border-bottom: 1px solid #1a2530;
    }}
    .ss-card {{
      padding: 10px 18px;
      border-right: 1px solid #1a2530;
    }}
    .ss-card:last-child {{ border-right: none; }}
    .ss-label {{ font-size: 9px; color: #3a5060; letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 4px; }}
    .ss-teams {{ display: flex; gap: 6px; flex-wrap: wrap; }}
    .ss-team {{
      font-size: 11px;
      font-weight: 600;
      padding: 2px 7px;
      border-radius: 2px;
      letter-spacing: 0.04em;
    }}
    .badge-champ {{ background: #1a2808; color: #6bc422; border: 1px solid #2a4a0a; }}
    .badge-ucl   {{ background: #08162a; color: #4a9eff; border: 1px solid #0a2248; }}
    .badge-rel   {{ background: #1a0808; color: #e84040; border: 1px solid #4a0808; }}

    .zone-bar {{
      display: grid;
      grid-template-columns: 36px 120px 1fr 38px 34px 34px 34px 46px 54px 80px 72px;
      padding: 6px 18px;
      border-bottom: 1px solid #1a2530;
      background: #111820;
    }}
    .zh {{
      font-size: 9px;
      color: #3a5060;
      letter-spacing: 0.1em;
      text-transform: uppercase;
      text-align: center;
    }}
    .zh.l {{ text-align: left; }}
    .zone-bar .zh[data-sort] {{ cursor: pointer; }}
    .zone-bar .zh[data-sort]:hover {{ color: #9bb3c8; }}

    .zone-sep {{ height: 1px; background: #1e2a38; position: relative; margin: 2px 0; }}
    .zone-tag {{
      position: absolute;
      right: 18px;
      top: -8px;
      font-size: 9px;
      letter-spacing: 0.1em;
      text-transform: uppercase;
      padding: 1px 6px;
      border-radius: 2px;
      border: 1px solid;
      font-family: 'Courier New', monospace;
    }}
    .zt-champ{{ color: #6bc422; border-color: #2a4a0a; background: #1a2808; }}
    .zt-ucl  {{ color: #4a9eff; border-color: #0a2248; background: #060e1a; }}
    .zt-uel  {{ color: #d4920a; border-color: #4a2a08; background: #100800; }}
    .zt-conf {{ color: #9b59b6; border-color: #3a1060; background: #0e0618; }}
    .zt-rel  {{ color: #e84040; border-color: #4a0808; background: #100404; }}

    .t-row {{
      display: grid;
      grid-template-columns: 36px 120px 1fr 38px 34px 34px 34px 46px 54px 80px 72px;
      padding: 5px 18px;
      border-bottom: 1px solid #0f1318;
      align-items: center;
      transition: background 0.1s;
      cursor: default;
    }}
    .t-row:hover {{ background: #131c26; }}
    .t-row.z-ucl  {{ border-left: 2px solid #4a9eff; padding-left: 16px; }}
    .t-row.z-uel  {{ border-left: 2px solid #d4920a; padding-left: 16px; }}
    .t-row.z-conf {{ border-left: 2px solid #9b59b6; padding-left: 16px; }}
    .t-row.z-rel  {{ border-left: 2px solid #e84040; padding-left: 16px; }}
    .t-row.z-mid  {{ border-left: 2px solid transparent; padding-left: 16px; }}
    .t-row.champ {{ background: #121a12; }}

    .c-pos {{ font-size: 13px; font-weight: 600; color: #4a7090; text-align: center; }}
    .c-mv  {{ font-size: 11px; text-align: center; color: #4a6070; }}
    .mv-u  {{ color: #1a9e5a; }}
    .mv-d  {{ color: #e84040; }}
    .mv-eq {{ color: #2a3a4a; }}

    .c-team {{ font-size: 13px; color: #b0c4d8; font-weight: 500; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
    .c-num  {{ font-size: 12px; color: #7090a8; text-align: center; font-family: 'Courier New', monospace; }}
    .c-pts  {{ font-size: 14px; font-weight: 700; color: #c8d8e8; text-align: center; font-family: 'Courier New', monospace; }}
    .c-gd   {{ font-size: 12px; text-align: center; font-family: 'Courier New', monospace; }}
    .gd-pos {{ color: #1a9e5a; }}
    .gd-neg {{ color: #e84040; }}
    .gd-zero{{ color: #4a6070; }}

    .form-row {{ display: flex; gap: 2px; align-items: center; justify-content: center; }}
    .fp {{
      width: 12px; height: 12px; border-radius: 1px;
      display: flex; align-items: center; justify-content: center;
      font-size: 7px; font-weight: 700; color: #0d1117;
    }}
    .fp-w {{ background: #1a9e5a; }}
    .fp-d {{ background: #5a7090; }}
    .fp-l {{ background: #e84040; }}

    .conf-cell {{ display: flex; align-items: center; gap: 5px; }}
    .conf-bar-bg {{ flex: 1; height: 4px; background: #1a2530; border-radius: 1px; }}
    .conf-bar-fg {{ height: 4px; border-radius: 1px; }}
    .conf-val {{ font-size: 10px; color: #4a6070; font-family: 'Courier New', monospace; min-width: 30px; text-align: right; }}

    .footer-strip {{
      padding: 8px 18px;
      background: #111820;
      border-top: 1px solid #1a2530;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }}
    .fs-left {{ font-size: 10px; color: #2a3a4a; letter-spacing: 0.06em; }}
    .fs-right {{ display: flex; gap: 14px; }}
    .legend-item {{ display: flex; align-items: center; gap: 4px; font-size: 9px; color: #3a5060; letter-spacing: 0.06em; text-transform: uppercase; }}
    .l-pip {{ width: 8px; height: 8px; border-radius: 1px; }}
    .lp-ucl{{background:#4a9eff;}} .lp-uel{{background:#d4920a;}} .lp-conf{{background:#9b59b6;}} .lp-rel{{background:#e84040;}}
    </style>

    <div class="wrap">
      <div class="topbar">
        <div class="tb-left">
          <span class="tb-path">pl-predictor / table / 2025-26 / xgb_calibrated</span>
        </div>
        <div class="tb-right">
          <div class="tb-stat"><div class="tb-stat-label">Accuracy</div><div class="tb-stat-val v-amber">{escape(str(accuracy))}</div></div>
          <div class="tb-stat"><div class="tb-stat-label">Macro F1</div><div class="tb-stat-val v-amber">{escape(str(macro_f1))}</div></div>
          <div class="tb-stat"><div class="tb-stat-label">Log Loss</div><div class="tb-stat-val v-muted">{escape(str(log_loss))}</div></div>
        </div>
      </div>
      <div class="headline">
        <div>
          <div class="hl-title">Premier League 2025/26 — Predicted Final Table</div>
          <div class="hl-sub">XGBoost · Calibrated · GW38 projection</div>
        </div>
        <div class="hl-meta">
          <div class="meta-blk">
            <div class="meta-label">Matches simulated</div>
            <div class="meta-val">{matches_simulated} <span>matches</span></div>
          </div>
          <div class="meta-blk">
            <div class="meta-label">Model</div>
            <div class="meta-val" style="font-size:13px;margin-top:3px;color:#d4920a;">XGB</div>
          </div>
        </div>
      </div>
      <div class="zone-bar">
        <div class="zh" data-sort="pos">Pos</div>
        <div class="zh" data-sort="delta">Change vs Current</div>
        <div class="zh l" data-sort="club">Club</div>
        <div class="zh" data-sort="pl">Pl</div>
        <div class="zh" data-sort="w">W</div>
        <div class="zh" data-sort="d">D</div>
        <div class="zh" data-sort="l">L</div>
        <div class="zh" data-sort="gd">GD</div>
        <div class="zh" data-sort="pts">Pts</div>
        <div class="zh" data-sort="form">Form</div>
        <div class="zh" data-sort="conf">Conf%</div>
      </div>
      <div class="t-body">{''.join(rows_html)}</div>
      <div class="footer-strip">
        <div class="fs-left">LIVE MODEL · XGBOOST CALIBRATED · UPDATED</div>
        <div class="fs-right">
          <div class="legend-item"><div class="l-pip lp-ucl"></div>UCL</div>
          <div class="legend-item"><div class="l-pip lp-uel"></div>UEL</div>
          <div class="legend-item"><div class="l-pip lp-conf"></div>UECL</div>
          <div class="legend-item"><div class="l-pip lp-rel"></div>REL</div>
        </div>
      </div>
    </div>
    <script>
      const initTableSort = () => {{
        const bodyEl = document.querySelector('.t-body');
        const headers = document.querySelectorAll('.zone-bar .zh[data-sort]');
        if (!bodyEl || !headers.length) return;
        let sortState = {{}};
        headers.forEach(h => {{
          h.addEventListener('click', () => {{
            const key = h.dataset.sort;
            const rows = Array.from(bodyEl.querySelectorAll('.t-row'));
            sortState[key] = !sortState[key];
            const dir = sortState[key] ? 1 : -1;
            rows.sort((a, b) => {{
              if (key === 'club') {{
                const av = (a.dataset[key] || '').toLowerCase();
                const bv = (b.dataset[key] || '').toLowerCase();
                return av.localeCompare(bv) * dir;
              }}
              const av = parseFloat(a.dataset[key] || 0);
              const bv = parseFloat(b.dataset[key] || 0);
              return (av - bv) * dir;
            }});
            bodyEl.innerHTML = '';
            rows.forEach(r => bodyEl.appendChild(r));
          }});
        }});
      }};
      if (document.readyState === 'loading') {{
        document.addEventListener('DOMContentLoaded', initTableSort);
      }} else {{
        initTableSort();
      }}
    </script>
    """
    return html


_TRAJECTORY_COLORS = {
    "Arsenal": "#ef0107",
    "Manchester City": "#6cabdd",
    "Aston Villa": "#95bfe5",
    "Chelsea": "#034694",
    "Manchester United": "#da291c",
    "Liverpool": "#c8102e",
    "Brentford": "#e30613",
    "Brighton": "#0057b8",
    "Everton": "#003399",
    "Newcastle": "#241f20",
    "Tottenham": "#132257",
    "Fulham": "#000000",
    "West Ham": "#7a263a",
    "Wolves": "#fdb913",
    "Nottingham Forest": "#dd0000",
    "Bournemouth": "#da291c",
    "Crystal Palace": "#1b458f",
    "Sunderland": "#eb172b",
    "Leeds": "#ffcd00",
    "Burnley": "#6c1d45",
}


def _build_points_trajectory_frame(
    season_df: pd.DataFrame,
    preds_df: pd.DataFrame,
    teams: list,
    cur_points: dict,
    predicted_pts: dict,
    actual_end_date: pd.Timestamp,
) -> pd.DataFrame:
    """
    Actual segment: per-match cumulative from raw results (reconciled to cur_points).
    Projected segment: linear steps from cur_points → predicted_pts across remaining
    fixture dates so the dashed line ends exactly on the model's integer prediction.
    """
    team_set = set(teams)
    records = []
    played = season_df[
        season_df["home_score"].notna()
        & season_df["away_score"].notna()
        & (season_df["date"] <= actual_end_date)
    ].copy()
    for _, row in played.iterrows():
        h, a = row["home_team_n"], row["away_team_n"]
        if h not in team_set and a not in team_set:
            continue
        hg, ag = float(row["home_score"]), float(row["away_score"])
        hp, ap = (3, 0) if hg > ag else (1, 1) if hg == ag else (0, 3)
        d = row["date"]
        if h in team_set:
            records.append({"team": h, "date": d, "pts": hp, "type": "actual"})
        if a in team_set:
            records.append({"team": a, "date": d, "pts": ap, "type": "actual"})
    actual_df = (
        pd.DataFrame(records)
        if records
        else pd.DataFrame(columns=["team", "date", "pts", "type"])
    )
    if cur_points:
        adj_rows = []
        for t in teams:
            sub = actual_df[actual_df["team"] == t]
            derived = float(sub["pts"].sum()) if not sub.empty else 0.0
            target = float(cur_points.get(t, derived))
            delta = target - derived
            if abs(delta) > 1e-6:
                adj_rows.append(
                    {
                        "team": t,
                        "date": actual_end_date + pd.Timedelta(seconds=1),
                        "pts": delta,
                        "type": "actual",
                    }
                )
        if adj_rows:
            actual_df = pd.concat(
                [actual_df, pd.DataFrame(adj_rows)], ignore_index=True
            )
    future = preds_df[preds_df["Date"] > actual_end_date].copy()
    proj_records = []
    for team in teams:
        start_pts = float(cur_points.get(team, 0))
        end_pts = float(predicted_pts.get(team, start_pts))
        pts_to_add = end_pts - start_pts

        team_future = future[
            (future["Home Team"] == team) | (future["Away Team"] == team)
        ].sort_values("Date")

        if team_future.empty:
            continue

        n = len(team_future)
        pts_per_fixture = pts_to_add / n

        for i, (_, row) in enumerate(team_future.iterrows()):
            if i < n - 1:
                step_pts = float(round(pts_per_fixture))
            else:
                step_pts = float(
                    end_pts - start_pts - round(pts_per_fixture) * (n - 1)
                )
            proj_records.append(
                {
                    "team": team,
                    "date": row["Date"],
                    "pts": step_pts,
                    "type": "projected",
                }
            )
    proj_df = (
        pd.DataFrame(proj_records)
        if proj_records
        else pd.DataFrame(columns=["team", "date", "pts", "type"])
    )
    if actual_df.empty and proj_df.empty:
        return pd.DataFrame()
    combined = pd.concat([actual_df, proj_df], ignore_index=True)
    combined["_ord"] = combined["type"].map({"actual": 0, "projected": 1})
    combined = combined.sort_values(["team", "date", "_ord"]).reset_index(drop=True)
    combined["cumulative_pts"] = combined.groupby("team")["pts"].cumsum()
    combined = combined.drop(columns=["_ord"])
    return combined


def _plot_points_trajectory_figure(
    combined_df: pd.DataFrame, selected_teams: list
) -> go.Figure:
    fig = go.Figure()
    for team in selected_teams:
        color = _TRAJECTORY_COLORS.get(team, "#9bb3c8")
        r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
        tdata = combined_df[combined_df["team"] == team].sort_values("date")
        actual = tdata[tdata["type"] == "actual"].copy()
        projected = tdata[tdata["type"] == "projected"].copy()
        if not actual.empty and not projected.empty:
            last_actual = actual.iloc[-1]
            bridge = pd.DataFrame(
                [
                    {
                        "team": team,
                        "date": last_actual["date"],
                        "pts": 0.0,
                        "type": "projected",
                        "cumulative_pts": last_actual["cumulative_pts"],
                    }
                ]
            )
            projected = pd.concat([bridge, projected], ignore_index=True)
        if not projected.empty and len(projected) > 1:
            upper = projected["cumulative_pts"] * 1.04
            lower = projected["cumulative_pts"] * 0.96
            band_x = list(projected["date"]) + list(projected["date"])[::-1]
            band_y = list(upper) + list(lower)[::-1]
            fig.add_trace(
                go.Scatter(
                    x=band_x,
                    y=band_y,
                    mode="lines",
                    fill="toself",
                    fillcolor=f"rgba({r},{g},{b},0.10)",
                    line=dict(width=0, color="rgba(0,0,0,0)"),
                    showlegend=False,
                    hoverinfo="skip",
                    legendgroup=team,
                    name=f"{team} band",
                )
            )
        if not actual.empty:
            fig.add_trace(
                go.Scatter(
                    x=actual["date"],
                    y=actual["cumulative_pts"],
                    mode="lines",
                    name=team,
                    line=dict(color=color, width=2.5, dash="solid"),
                    legendgroup=team,
                )
            )
        if not projected.empty:
            fig.add_trace(
                go.Scatter(
                    x=projected["date"],
                    y=projected["cumulative_pts"],
                    mode="lines",
                    name=f"{team} (proj)",
                    line=dict(color=color, width=1.8, dash="dash"),
                    legendgroup=team,
                    showlegend=False,
                )
            )
    fig.update_layout(
        height=420,
        margin=dict(l=20, r=20, t=10, b=20),
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        font=dict(family="Courier New, monospace", color="#7090a8", size=11),
        xaxis=dict(
            gridcolor="#1a2530",
            linecolor="#1e2a38",
            tickfont=dict(size=10),
            title=None,
            range=["2025-08-01", "2026-06-01"],
        ),
        yaxis=dict(
            gridcolor="#1a2530",
            linecolor="#1e2a38",
            tickfont=dict(size=10),
            title=dict(text="Pts", font=dict(size=11)),
            rangemode="tozero",
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="left",
            x=0,
            font=dict(size=10),
            bgcolor="rgba(0,0,0,0)",
        ),
        hovermode="x unified",
    )
    return fig


def _render_fixtures_section(preds_path: Path) -> None:
    st.markdown("### Remaining Fixture Predictions")
    if preds_path.exists():
        preds_df = pd.read_csv(preds_path)
        if "matchweek" in preds_df.columns:
            preds_df["GW"] = (
                preds_df["matchweek"]
                .astype(str)
                .str.extract(r"'gameweek':\s*([0-9]+)")[0]
            )
            preds_df = preds_df.drop(columns=["matchweek"])
        if "date" in preds_df.columns:
            preds_df["Date Sort"] = pd.to_datetime(preds_df["date"], errors="coerce")
            preds_df = preds_df.sort_values("Date Sort").drop(columns=["Date Sort"])
        rename_map = {
            "fixture_id": "Fixture ID",
            "date": "Date",
            "home_team": "Home Team",
            "away_team": "Away Team",
            "predicted_outcome": "Predicted Outcome",
            "home_win_prob": "Home Win %",
            "draw_prob": "Draw %",
            "away_win_prob": "Away Win %",
        }
        preds_df = preds_df.rename(columns=rename_map)
        for col in ["Home Win %", "Draw %", "Away Win %"]:
            if col in preds_df.columns:
                preds_df[col] = (preds_df[col] * 100).round(1)
        ordered_cols = [
            "GW",
            "Date",
            "Home Team",
            "Away Team",
            "Predicted Outcome",
            "Home Win %",
            "Draw %",
            "Away Win %",
        ]
        preds_df = preds_df[[c for c in ordered_cols if c in preds_df.columns]]
        st.markdown('<div class="section tight">', unsafe_allow_html=True)
        st.dataframe(preds_df, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info(
            "Run generate_remaining_predictions.py to create remaining fixture predictions."
        )


def _render_model_section(pred_df: pd.DataFrame, cur_df: pd.DataFrame) -> None:
    st.markdown("### Points Trajectory")
    st.caption(
        "Solid = cumulative points from results (reconciled to current table). "
        "Dashed = linear steps to the model's predicted final points across remaining fixtures."
    )
    raw_path = DATA_DIR / "raw_matches.csv"
    preds_path = DATA_DIR / "remaining_predictions.csv"
    if raw_path.exists() and preds_path.exists() and not cur_df.empty:
        raw_df = pd.read_csv(raw_path)
        try:
            raw_df["date"] = pd.to_datetime(
                raw_df["date"], format="mixed", errors="coerce"
            )
        except (TypeError, ValueError):
            raw_df["date"] = pd.to_datetime(raw_df["date"], errors="coerce")
        bad_dates = int(raw_df["date"].isna().sum())
        if bad_dates > 0:
            st.caption(f"⚠ {bad_dates} raw match rows dropped (unparseable date).")
            raw_df = raw_df.dropna(subset=["date"])
        raw_df["date"] = _datetime_series_to_naive(raw_df["date"])
        raw_df["home_team_n"] = raw_df["home_team"].map(normalize_team_name)
        raw_df["away_team_n"] = raw_df["away_team"].map(normalize_team_name)
        if "season" in raw_df.columns and raw_df["season"].notna().any():
            latest_season = raw_df["season"].dropna().sort_values().iloc[-1]
            season_df = raw_df[raw_df["season"] == latest_season].copy()
        else:
            season_df = raw_df.copy()
        season_df = season_df.sort_values("date")
        teams = sorted(pred_df["team"].dropna().unique().tolist())

        preds_df = pd.read_csv(preds_path)
        if "date" in preds_df.columns:
            preds_df["Date"] = (
                preds_df["date"]
                .astype(str)
                .str.replace(r"\s+(BST|GMT|UTC)$", "", regex=True)
            )
            preds_df["Date"] = pd.to_datetime(
                preds_df["Date"], format="%a %d %b %Y, %H:%M", errors="coerce"
            )
        else:
            preds_df["Date"] = pd.NaT
        preds_df["Date"] = _datetime_series_to_naive(preds_df["Date"])
        preds_df = preds_df.dropna(subset=["Date"]).sort_values("Date")
        preds_df["Home Team"] = preds_df["home_team"].map(normalize_team_name)
        preds_df["Away Team"] = preds_df["away_team"].map(normalize_team_name)

        scored = season_df["home_score"].notna() & season_df["away_score"].notna()
        last_res = season_df.loc[scored, "date"].max()
        snap_cap = pd.Timestamp.now().normalize()
        if getattr(last_res, "tzinfo", None) is not None:
            last_res = pd.Timestamp(last_res).tz_localize(None)
        if getattr(snap_cap, "tzinfo", None) is not None:
            snap_cap = snap_cap.tz_localize(None)
        actual_end_date = min(last_res, snap_cap) if pd.notna(last_res) else snap_cap

        cur_norm = _normalize_columns(cur_df)
        if "team" in cur_norm.columns:
            cur_norm["team"] = cur_norm["team"].map(normalize_team_name)
        cur_points: dict = {}
        if "pts" in cur_norm.columns and "team" in cur_norm.columns:
            raw_vals = cur_norm.set_index("team")["pts"].to_dict()
            vals = list(raw_vals.values())
            if len(set(vals)) > 1 and any(v > 0 for v in vals):
                cur_points = {k: float(v) for k, v in raw_vals.items()}

        if not cur_points:
            derived_pts: dict = {}
            scored_rows = season_df[
                season_df["home_score"].notna()
                & season_df["away_score"].notna()
                & (season_df["date"] <= actual_end_date)
            ]
            for _, row in scored_rows.iterrows():
                h, a = row["home_team_n"], row["away_team_n"]
                hg, ag = float(row["home_score"]), float(row["away_score"])
                hp, ap = (3, 0) if hg > ag else (1, 1) if hg == ag else (0, 3)
                derived_pts[h] = derived_pts.get(h, 0) + hp
                derived_pts[a] = derived_pts.get(a, 0) + ap
            cur_points = derived_pts

        predicted_pts: dict = {}
        if "points" in pred_df.columns and "team" in pred_df.columns:
            predicted_pts = {
                str(t): int(round(float(v)))
                for t, v in pred_df.set_index("team")["points"].to_dict().items()
            }

        combined = _build_points_trajectory_frame(
            season_df, preds_df, teams, cur_points, predicted_pts, actual_end_date
        )
        if combined.empty:
            st.info("No trajectory rows after filtering.")
        else:
            act_out = combined[combined["type"] == "actual"].copy()
            act_out["GW"] = act_out.groupby("team").cumcount() + 1
            act_out.rename(
                columns={"team": "Team", "date": "Date", "cumulative_pts": "Points"},
                inplace=True,
            )
            act_out.to_csv(DATA_DIR / "points_trajectory_actual.csv", index=False)
            proj_out = combined[combined["type"] == "projected"].copy()
            proj_out.rename(
                columns={"team": "Team", "date": "Date", "cumulative_pts": "Points"},
                inplace=True,
            )
            proj_out["Points"] = proj_out["Points"].round(0).astype(int)
            proj_out.to_csv(DATA_DIR / "points_trajectory_predicted.csv", index=False)

            default_teams = pred_df.sort_values("position").head(6)["team"].tolist()
            selected = st.multiselect("Teams", teams, default=default_teams)
            fig = _plot_points_trajectory_figure(combined, selected)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Points trajectory needs raw matches and remaining predictions.")

    st.markdown("### Head-to-head Probability Heatmap")
    st.caption(
        "Each cell shows the model's home-win probability for that matchup "
        "(row = home team, column = away team)."
    )
    model_path = Path("models") / "best_model.pkl"
    features_path = DATA_DIR / "features.csv"
    if model_path.exists() and features_path.exists():
        model_data = joblib.load(model_path)
        model = model_data.get("model") or model_data.get("calibrated_model")
        scaler = model_data.get("scaler")
        feature_cols = model_data.get("feature_columns", [])
        feat_df = pd.read_csv(features_path)
        feat_df["date"] = pd.to_datetime(feat_df["date"], errors="coerce")
        feat_df["home_team_n"] = feat_df["home_team"].map(normalize_team_name)
        feat_df["away_team_n"] = feat_df["away_team"].map(normalize_team_name)
        feat_df = feat_df.sort_values("date")
        teams = sorted(pred_df["team"].dropna().unique().tolist())
        home_latest = (
            feat_df[feat_df["home_team_n"].isin(teams)]
            .groupby("home_team_n")
            .tail(1)
            .set_index("home_team_n")
        )
        away_latest = (
            feat_df[feat_df["away_team_n"].isin(teams)]
            .groupby("away_team_n")
            .tail(1)
            .set_index("away_team_n")
        )
        z = []
        for home in teams:
            row = []
            hrow = home_latest.loc[home] if home in home_latest.index else None
            for away in teams:
                arow = away_latest.loc[away] if away in away_latest.index else None
                if hrow is None or arow is None:
                    row.append(0.0)
                    continue
                feats = {}
                for col in feature_cols:
                    if col.startswith("home_"):
                        feats[col] = hrow.get(col, 0)
                    elif col.startswith("away_"):
                        feats[col] = arow.get(col, 0)
                    else:
                        feats[col] = hrow.get(col, 0)
                X = scaler.transform([[feats.get(c, 0.0) for c in feature_cols]])
                probs = model.predict_proba(X)[0]
                home_win = float(probs[2]) if len(probs) > 2 else float(probs[0])
                row.append(round(home_win, 4))
            z.append(row)
        heat = go.Figure(
            data=go.Heatmap(
                z=z,
                x=teams,
                y=teams,
                zmin=0,
                zmax=1,
                colorscale=[[0, "#0d1117"], [1, "#d4920a"]],
                colorbar=dict(title="Home Win %"),
                hovertemplate="Home: %{y}<br>Away: %{x}<br>Home Win: %{z:.0%}<extra></extra>",
            )
        )
        heat.update_layout(
            height=520,
            margin=dict(l=20, r=20, t=30, b=20),
            paper_bgcolor="#0d1117",
            plot_bgcolor="#0d1117",
            font=dict(color="#c8d8e8"),
            xaxis=dict(title="Away Team"),
            yaxis=dict(title="Home Team"),
        )
        st.plotly_chart(heat, use_container_width=True)
    else:
        st.info("Heatmap needs model file and features.csv.")

    def _pretty_feature_name(name: str) -> str:
        swaps = {
            "xg": "xG",
            "gd": "GD",
            "ppg": "PPG",
            "sot": "SoT",
            "avg": "Avg",
            "home": "Home",
            "away": "Away",
            "form": "Form",
        }
        parts = name.replace("_", " ").split()
        pretty = []
        for p in parts:
            key = p.lower()
            pretty.append(swaps.get(key, p.title()))
        return " ".join(pretty)

    st.markdown("### Feature Importance")
    st.caption("Top 15 XGBoost features by gain (higher = more influence).")
    if model_path.exists():
        model_data = joblib.load(model_path)
        model = model_data.get("model") or model_data.get("calibrated_model")
        feature_cols = model_data.get("feature_columns", [])
        importances = {}
        if hasattr(model, "get_booster"):
            raw = model.get_booster().get_score(importance_type="gain")
            # Map f0..fN to actual feature names when available
            importances = {
                (feature_cols[int(k[1:])] if k.startswith("f") and k[1:].isdigit() and int(k[1:]) < len(feature_cols) else k): v
                for k, v in raw.items()
            }
        elif hasattr(model, "feature_importances_"):
            vals = model.feature_importances_.tolist()
            importances = {c: v for c, v in zip(feature_cols, vals)}
        if importances:
            imp_series = (
                pd.Series(importances)
                .sort_values(ascending=False)
                .head(15)
                .iloc[::-1]
            )
            imp_series.index = [ _pretty_feature_name(i) for i in imp_series.index ]
            fig_imp = go.Figure(
                go.Bar(
                    x=imp_series.values,
                    y=imp_series.index,
                    orientation="h",
                    marker=dict(color="#f59e0b"),
                )
            )
            fig_imp.update_layout(
                height=420,
                margin=dict(l=120, r=20, t=30, b=20),
                paper_bgcolor="#0d1117",
                plot_bgcolor="#0d1117",
                font=dict(color="#c8d8e8"),
                xaxis=dict(gridcolor="#1e2a38"),
            )
            st.plotly_chart(fig_imp, use_container_width=True)
        else:
            st.info("Feature importance is not available for this model.")

    st.markdown("### Strongest Finishers (Last 5 Form)")
    form_points = (
        pred_df.set_index("team")["form_points_5"].to_dict()
        if "form_points_5" in pred_df.columns
        else {}
    )
    position_map = pred_df.set_index("team")["position"].to_dict()
    form_chart = (
        pred_df[["team"]]
        .assign(form_points=lambda d: d["team"].map(form_points).fillna(0))
        .sort_values("form_points", ascending=False)
        .head(10)
        .assign(Position=lambda d: d["team"].map(position_map))
        .rename(columns={"team": "Team", "form_points": "Form Points (5)"})
    )
    chart = (
        alt.Chart(form_chart)
        .mark_bar(color="#f59e0b", cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
        .encode(
            x=alt.X("Team:N", sort=alt.SortField("Form Points (5)", order="descending")),
            y=alt.Y("Form Points (5):Q"),
            tooltip=["Team", "Form Points (5)"],
        )
        .properties(height=280)
    )
    st.altair_chart(chart, use_container_width=True)
    show_form_table = st.toggle("Show table view", value=False)
    if show_form_table:
        form_html_rows = []
        for _, r in form_chart.iterrows():
            form_html_rows.append(
                f"<tr><td>{int(r['Position'])}</td><td>{escape(str(r['Team']))}</td><td>{int(r['Form Points (5)'])}</td></tr>"
            )
        form_table_html = f"""
        <style>
          .mini-wrap {{
            background: #0d1117;
            border: 1px solid #1e2a38;
            border-radius: 10px;
            overflow: hidden;
            font-family: 'Courier New', monospace;
            margin-top: 10px;
          }}
          .mini-head {{
            padding: 8px 14px;
            border-bottom: 1px solid #1a2530;
            color: #9bb3c8;
            font-size: 10px;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            background: #111820;
          }}
          .mini-table {{
            width: 100%;
            border-collapse: collapse;
          }}
          .mini-table th, .mini-table td {{
            padding: 6px 14px;
            border-bottom: 1px solid #0f1318;
            font-size: 12px;
            color: #c8d8e8;
          }}
          .mini-table th {{
            color: #3a5060;
            font-size: 9px;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            text-align: left;
          }}
        </style>
        <div class="mini-wrap">
          <div class="mini-head">Top 10 Finishers by Form</div>
          <table class="mini-table">
            <thead><tr><th>Pos</th><th>Club</th><th>Form Pts</th></tr></thead>
            <tbody>{''.join(form_html_rows)}</tbody>
          </table>
        </div>
        """
        components.html(form_table_html, height=320, scrolling=False)

    st.markdown(
        """
        <div class="card about-app-card">
          <h4>About This App</h4>
          <p>This dashboard predicts remaining fixtures using a calibrated XGBoost model
          built on recent form (5 &amp; 10 game windows), rolling goals for/against,
          rest days, squad value, manager PPG and trophies, and weather signals,
          then projects the final table by rolling those outcomes through to the end of the season.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.set_page_config(page_title="Premier League Predictions", layout="wide")
alt.renderers.set_embed_options(actions=False)

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700&display=swap');
      html, body, [class*="css"]  {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      }
      .app-header {
        background: #0d1117;
        border: 1px solid #1e2a38;
        border-radius: 12px;
        padding: 12px 16px;
        margin-bottom: 10px;
      }
      .app-title {
        font-family: 'Courier New', monospace;
        font-size: 13px;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        color: #c8d8e8;
      }
      .live-dot {
        color: #d4920a;
        margin-left: 8px;
      }
      .section {
        background: #0f141b;
        border: 1px solid #1e2a38;
        border-radius: 14px;
        padding: 14px 16px;
        margin-top: 8px;
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.25);
        color: #d7e3ef;
      }
      .spacer {
        height: 16px;
      }
      .tight {
        margin-top: 4px;
      }
      /* Bloomberg table iframe: trim dead space; pull Model Metrics up */
      [data-testid="stHtml"] {
        margin-bottom: 0 !important;
      }
      div[data-testid="element-container"]:has([data-testid="stHtml"])
        + div[data-testid="element-container"] {
        margin-top: -1.25rem !important;
      }
      /* Same ### styling as everywhere else; only tighten vertical spacing */
      div[data-testid="element-container"]:has([data-testid="stHtml"])
        + div[data-testid="element-container"] h3 {
        margin-top: 0.25rem !important;
        margin-bottom: 0.5rem !important;
      }
      .card {
        background: linear-gradient(135deg, #121823 0%, #0f141b 100%);
        border: 1px solid #1e2a38;
        border-radius: 12px;
        padding: 12px 14px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.35);
        color: #d7e3ef;
      }
      .card h4 {
        margin: 0 0 6px 0;
        font-size: 13px;
        color: #a9bfd4;
      }
      .card p {
        margin: 0;
        font-size: 13px;
        color: #e6eef7;
        line-height: 1.3;
      }
      .about-app-card h4 {
        font-family: 'Courier New', monospace;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        font-size: 11px;
      }
      .about-app-card p {
        font-family: 'Courier New', monospace;
        font-size: 12px;
        line-height: 1.45;
        color: #c8d8e8;
      }
      [data-testid="stToolbar"], #MainMenu, footer {
        visibility: hidden;
        height: 0px;
      }
      [data-testid="stSidebar"] > div:first-child {
        padding-top: 0.5rem;
      }
      .gold-text {
        color: #d4af37;
        font-weight: 700;
      }
      .chips {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
      }
      .chip {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 999px;
        font-size: 11px;
        background: #eef2ff;
        color: #4338ca;
      }
      .pill {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 999px;
        background: #f2f6ff;
        color: #234;
        font-size: 12px;
        margin-right: 6px;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="app-header">
      <div class="app-title">PREMIER LEAGUE 2025/26 · PREDICTED FINAL TABLE <span class="live-dot">● LIVE</span></div>
    </div>
    """,
    unsafe_allow_html=True,
)

pred_path, pred_df = _load_first_existing(PRED_TABLE_PATHS)
cur_path, cur_df = _load_first_existing(CURRENT_TABLE_PATHS)

uploaded_cur = st.sidebar.file_uploader("Upload current table CSV", type=["csv"])
if uploaded_cur is not None:
    cur_df = pd.read_csv(uploaded_cur)
    cur_path = None
    # persist so prediction builder can use it
    DATA_DIR.mkdir(exist_ok=True)
    cur_df.to_csv(DATA_DIR / "current_table.csv", index=False)
    st.session_state["needs_rebuild"] = True

if "needs_rebuild" not in st.session_state:
    st.session_state["needs_rebuild"] = False

if st.session_state["needs_rebuild"] or pred_df.empty:
    with st.spinner("Building predictions from current table..."):
        build_predicted_table_main()
    pred_path, pred_df = _load_first_existing(PRED_TABLE_PATHS)
    st.session_state["needs_rebuild"] = False

if pred_df.empty:
    st.warning("No predicted table found. Place a CSV in data/ or upload one.")
    st.stop()

pred_df = _normalize_columns(pred_df)
pred_df = _ensure_position(pred_df)
pred_df = _add_deltas(pred_df, cur_df)

raw_matches_path = DATA_DIR / "raw_matches.csv"
raw_matches_df = pd.read_csv(raw_matches_path) if raw_matches_path.exists() else pd.DataFrame()
form_map = _build_form_map(raw_matches_df)
pred_df["form_last5"] = pred_df["team"].map(lambda t: form_map.get(t, {}).get("form", "")).fillna("")
pred_df["form_points_5"] = pred_df["team"].map(lambda t: form_map.get(t, {}).get("points", 0)).fillna(0)

pred_df["zone"] = pred_df["position"].apply(lambda p: _zone_label(int(p)))
pred_df["delta"] = pred_df["pos_change"].apply(_delta_label)

show_cols = [
    c
    for c in [
        "position",
        "team",
        "pl",
        "w",
        "d",
        "l",
        "gf",
        "ga",
        "gd",
        "points",
        "zone",
        "delta",
        "form_last5",
    ]
    if c in pred_df.columns
]
table_df = pred_df[show_cols].copy()
table_df = table_df.rename(
    columns={
        "position": "Pos",
        "team": "Team",
        "pl": "Pl",
        "w": "W",
        "d": "D",
        "l": "L",
        "gf": "GF",
        "ga": "GA",
        "points": "Pts",
        "gd": "GD",
        "zone": "Zone",
        "delta": "Δ (Change from current position)",
        "form_last5": "Form (last 5)",
    }
)

# Encode form points into Form column for correct sorting (hidden by formatter)
if "Form (last 5)" in table_df.columns:
    form_points_map = (
        pred_df.set_index("team")["form_points_5"].to_dict()
        if "form_points_5" in pred_df.columns
        else {}
    )
    form_points_series = (
        table_df["Team"].map(form_points_map).fillna(0).astype(int)
    )
    form_text = table_df["Form (last 5)"].fillna("").astype(str)
    table_df["Form (last 5)"] = form_points_series.map(lambda p: f"{p:02d}") + "|" + form_text
if "Δ (Change from current position)" in table_df.columns:
    pos_change_map = pred_df.set_index("team")["pos_change"].to_dict()
    delta_series = table_df["Team"].map(pos_change_map).fillna(0).astype(int)
    delta_text = table_df["Δ (Change from current position)"].fillna("").astype(str)
    # Prefix sorts by numeric value on a number line
    def _delta_prefix(val: int) -> str:
        return f"{val + 100:03d}"

    table_df["Δ (Change from current position)"] = (
        delta_series.map(_delta_prefix) + "|" + delta_text
    )

metrics_path = DATA_DIR / "performance_metrics.txt"
metrics = {}
if metrics_path.exists():
    for line in metrics_path.read_text().splitlines():
        if ":" in line:
            key, val = line.split(":", 1)
            metrics[key.strip()] = val.strip()

matches_simulated = 0
preds_path = DATA_DIR / "remaining_predictions.csv"
if preds_path.exists():
    try:
        matches_simulated = len(pd.read_csv(preds_path))
    except Exception:
        matches_simulated = 0

main_sort_by = "Default (Position)"
main_sort_order = "Ascending"
main_ascending = main_sort_order == "Ascending"

table_view = table_df.copy()
zone_order = ["Relegation", "Mid", "UECL", "UEL", "UCL", "Champion"]
if "Zone" in table_view.columns:
    table_view["Zone"] = pd.Categorical(
        table_view["Zone"], categories=zone_order, ordered=True
    )
form_points_map = (
    pred_df.set_index("team")["form_points_5"].to_dict()
    if "form_points_5" in pred_df.columns
    else {}
)
pos_change_map = pred_df.set_index("team")["pos_change"].to_dict()
table_view["Form Points (5)"] = table_view["Team"].map(form_points_map).fillna(0)
table_view["Pos Change"] = table_view["Team"].map(pos_change_map).fillna(0)
table_view["Change Sign"] = table_view["Pos Change"].apply(
    lambda v: 1 if v > 0 else 0 if v == 0 else -1
)
table_view["Change Abs"] = table_view["Pos Change"].abs()

if main_sort_by == "Form points (last 5)":
    table_view = table_view.sort_values(
        ["Form Points (5)", "Pos Change"], ascending=[main_ascending, False]
    )
elif main_sort_by == "Change from current position":
    if main_ascending:
        table_view = table_view.sort_values(
            ["Change Sign", "Change Abs"], ascending=[True, False]
        )
    else:
        table_view = table_view.sort_values(
            ["Change Sign", "Change Abs"], ascending=[False, False]
        )

table_view = table_view.drop(
    columns=["Form Points (5)", "Pos Change", "Change Sign", "Change Abs"]
)
table_html = _render_bloomberg_table(table_view, metrics, matches_simulated)
_n_rows = len(table_view)
# Fit iframe to table (chrome + ~26px/row + footer); trims empty band vs fixed 760px
_bloomberg_iframe_h = int(178 + _n_rows * 26 + 36)
_bloomberg_iframe_h = max(460, min(_bloomberg_iframe_h, 1200))
components.html(table_html, height=_bloomberg_iframe_h, scrolling=True)

st.markdown("### Model Metrics")
if not metrics:
    st.info("Run test_predictions.py to generate performance metrics.")

metric_items = [
    ("Accuracy", metrics.get("Accuracy")),
    ("Macro F1", metrics.get("Macro F1")),
    ("Weighted F1", metrics.get("Weighted F1")),
    ("Precision", metrics.get("Precision (macro)")),
    ("Recall", metrics.get("Recall (macro)")),
    ("Log Loss", metrics.get("Log Loss")),
]

cols = st.columns(3)
for idx, (label, value) in enumerate(metric_items):
    if value is None:
        continue
    cols[idx % 3].markdown(
        f"""
        <div class="card">
          <h4>{label}</h4>
          <p>{value}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)
_render_fixtures_section(preds_path)

st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)
_render_model_section(pred_df, cur_df)

st.sidebar.markdown("### Filters")
zone_order = ["Relegation", "Mid", "UECL", "UEL", "UCL", "Champion"]
zones = [z for z in zone_order if z in table_df["Zone"].unique().tolist()]
if not zones:
    zones = sorted(table_df["Zone"].dropna().astype(str).unique().tolist())
selected_zones = st.sidebar.multiselect("Zones", options=zones, default=zones)
filtered = table_df[table_df["Zone"].isin(selected_zones)].copy()
if "Zone" in filtered.columns and not filtered.empty:
    _z_present = filtered["Zone"].dropna().unique().tolist()
    _z_cats = [c for c in zone_order if c in _z_present] + [
        c for c in _z_present if c not in zone_order
    ]
    filtered["Zone"] = pd.Categorical(
        filtered["Zone"], categories=_z_cats, ordered=True
    )
form_points = (
    pred_df.set_index("team")["form_points_5"].to_dict()
    if "form_points_5" in pred_df.columns
    else {}
)
pos_change_map = pred_df.set_index("team")["pos_change"].to_dict()
filtered["Form Points (5)"] = filtered["Team"].map(form_points).fillna(0)
filtered["Pos Change"] = filtered["Team"].map(pos_change_map).fillna(0)
filtered["Change Sign"] = filtered["Pos Change"].apply(
    lambda v: 1 if v > 0 else 0 if v == 0 else -1
)
filtered["Change Abs"] = filtered["Pos Change"].abs()

sort_by = st.sidebar.selectbox(
    "Sort by",
    [
        "Default (Position)",
        "Form points (last 5)",
        "Change from current position",
    ],
)
sort_order = st.sidebar.selectbox("Order", ["Ascending", "Descending"])
ascending = sort_order == "Ascending"

if sort_by == "Default (Position)" and "Pos" in filtered.columns:
    filtered = filtered.sort_values("Pos", ascending=ascending)
elif sort_by == "Form points (last 5)":
    filtered = filtered.sort_values(
        ["Form Points (5)", "Pos Change"], ascending=[ascending, False]
    )
elif sort_by == "Change from current position":
    if ascending:
        filtered = filtered.sort_values(
            ["Change Sign", "Change Abs"], ascending=[True, False]
        )
    else:
        filtered = filtered.sort_values(
            ["Change Sign", "Change Abs"], ascending=[False, False]
        )

filtered = filtered.drop(columns=["Form Points (5)", "Pos Change", "Change Sign", "Change Abs"])
if "Form (last 5)" in filtered.columns:
    filtered["Form (last 5)"] = filtered["Form (last 5)"].astype(str).str.split("|").str[-1]
if "Δ (Change from current position)" in filtered.columns:
    filtered["Δ (Change from current position)"] = (
        filtered["Δ (Change from current position)"].astype(str).str.split("|").str[-1]
    )
st.sidebar.markdown("### Filtered view")
if filtered.empty:
    st.sidebar.caption("No rows match the current filters.")
else:
    st.sidebar.dataframe(filtered, use_container_width=True, hide_index=True)
