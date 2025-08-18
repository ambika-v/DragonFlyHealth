"""
Dragonfly Health — AI Scheduling Demo (Streamlit)
=================================================

Quickstart
----------
1) Create a virtual env (recommended), then install deps:
   pip install streamlit pandas numpy scikit-learn altair python-dateutil ortools geopy

2) Run the app:
   streamlit run app.py

Notes
-----
- This is a turnkey demo that works *without* any proprietary data. It can also accept a CSV export
  from the Order Coordination system. Columns are auto-detected; see the `EXPECTED_COLUMNS` section
  for mappings and fallbacks.
- The "AI" portion includes a lightweight logistic-regression model to predict appointment adjustment
  probability, plus a slot scorer that balances SLA risk, resource load, patient preferences, and travel distance.
- New in this version: **Hospital → Patient Profile → Equipment → Slot & Route** flow, a **VRP‑lite** routing tab,
  **date+time pickers** (no `st.datetime_input`), and a **blue+green Dragonfly theme** with logo auto-detect.
- The code is organized for clarity rather than micro-optimizations; swap the model and scoring functions
  with your production counterparts later.

Branding
--------
- Place logos in `assets/dragonfly_logo.(png|jpg|jpeg)` and `assets/dragonfly_mark.(png|jpg|jpeg)`.
- Colors live in `PRIMARY_COLOR` (blue) and `ACCENT_COLOR` (green).

"""

from __future__ import annotations
import os
import io
import math
import random
from dataclasses import dataclass
from datetime import datetime, date, time, timedelta

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support

# Optional: OR-Tools for routing (VRP/TSP). Falls back to greedy if not present.
try:
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2
    HAS_ORTOOLS = True
except Exception:
    HAS_ORTOOLS = False

# ---------------------------
# Theming & Branding
# ---------------------------
PRIMARY_COLOR = "#114E7A"   # Dragonfly blue
ACCENT_COLOR  = "#14B58A"   # Dragonfly green
WARN_COLOR    = "#8FD3C8"   # muted teal
ALERT_COLOR   = "#2E8B57"   # deep green
LIGHT_BG      = "#F4FAF8"   # very light blue‑green

# Optional logos — add files to your repo under assets/ and they will render automatically
from pathlib import Path
import base64
BASE_DIR = Path(__file__).resolve().parent

# Prefer the filenames you provided
LOGO_PATH = "assets/dragonflyhealth_logo.jpg"         # full logo
LOGO_MARK_PATH = "assets/dragonflyhealth_mark.jpg"    # compact mark

# Robust auto-detect fallback: look for .png/.jpg/.jpeg in ./assets and ../assets
if not os.path.exists(LOGO_PATH) or not os.path.exists(LOGO_MARK_PATH):
    _logo = LOGO_PATH if os.path.exists(LOGO_PATH) else None
    _mark = LOGO_MARK_PATH if os.path.exists(LOGO_MARK_PATH) else None
    for folder in [BASE_DIR / "assets", BASE_DIR.parent / "assets", Path("assets")]:
        for ext in ["png", "jpg", "jpeg", "JPG", "JPEG", "PNG"]:
            if not _logo:
                cand = folder / f"dragonflyhealth_logo.{ext}"
                if cand.exists():
                    _logo = str(cand)
            if not _mark:
                cand2 = folder / f"dragonflyhealth_mark.{ext}"
                if cand2.exists():
                    _mark = str(cand2)
    LOGO_PATH = _logo or LOGO_PATH
    LOGO_MARK_PATH = _mark or LOGO_MARK_PATH

st.set_page_config(
    page_title="Dragonfly Health — AI Scheduling Demo",
    page_icon=LOGO_MARK_PATH if os.path.exists(LOGO_MARK_PATH) else "🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    f"""
    <style>
      .main {{ background: {LIGHT_BG}; }}
      .stApp header {{ background: white; border-bottom: 1px solid #e6f0ec; }}
      .metric-row {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }}
      .pill {{ display:inline-block; padding:4px 10px; border-radius:20px; background:{PRIMARY_COLOR}; color:white; font-size:12px; }}
      .brand {{ color: {PRIMARY_COLOR}; }}
      .accent {{ color: {ACCENT_COLOR}; }}
      .warn {{ color: {WARN_COLOR}; }}
      .alert {{ color: {ALERT_COLOR}; }}  .card {{ background:white; padding:16px; border-radius:16px; box-shadow: 0 2px 10px rgba(0,0,0,0.05); }}}}t}}{{ color:#4d6672; font-size:13px; }}  }}di}}e}}{ }}ght:1px; background:#e6f0ec; margin:8px 0 16px; }}
}}   }}But}}s }}   }}div.stButton>button:first-child {{ background:{PRIMARY_COLOR}; color:white; border-radius:10px; }}
 }}  /*}}ider}}/
 }}  .s}}ider [data-baseweb="slider"]>div>div {{ background:{ACCENT_COLOR}22; }}
  }} /* S}}llbar}}blue}}/
   }}*::-webkit-scrollbar {{ width: 10px; height: 10px; }}
   }}*::-we}}t-scro}}ar-th}} {{ ba}}round: {PRIMARY_COLOR}; border-radius: 8px; }}
    }}e6f0ec; }}
     }}c; scrol}}r-wid}}}}n; }}
}}   }}Br}}ed top }}bar */
  }}}d}} #114E7}}%, #0F}}6 55%, #14B58A 100%); color: white; padding: 10px 14px; border-radius: }}x; disp}}:flex; align-items:center; gap:12px; margin-bottom: 10px; }}
      .df-nav .df-mark {{ height: 40px; width:auto; border-radius:8px; background:#ffffff22; padding:4px; }}
      .df-nav .df-title {{ font-weight:700; font-size:18px; letter-spacing:0.3px; }}
    </style>
    """,
    unsafe_allow_html=True,
)
# Normalize any accidental triple braces from earlier edits (prevent f-string errors)
# (No-op if not present)


    unsafe_allow_html=True,
)
# Normalize any accidental triple braces from earlier edits (prevent f-string errors)
# (No-op if not present)


    unsafe_allow_html=True,
)
# Normalize any accidental triple braces from earlier edits (prevent f-string errors)
# (No-op if not present)


    unsafe_allow_html=True,
)
# Normalize any accidental triple braces from earlier edits (prevent f-string errors)
# (No-op if not present)


    unsafe_allow_html=True,
)
# Normalize any accidental triple braces from earlier edits (prevent f-string errors)
# (No-op if not present)


    unsafe_allow_html=True,
)
# Normalize any accidental triple braces from earlier edits (prevent f-string errors)
# (No-op if not present)


    unsafe_allow_html=True,
)

# ---------------------------
# Compact Landing Styles (add-on)
# ---------------------------
st.markdown(
    f"""
    <style>
      /* compact KPI tiles */
      .df-tiles {{ display:grid; grid-template-columns: repeat(6, minmax(0,1fr)); gap:10px; }}
      .df-tile {{ background:white; border-radius:14px; padding:12px; box-shadow:0 1px 6px rgba(0,0,0,.05); }}
      .df-t-h {{ font-size:12px; color:#56707a; }}
      .df-t-v {{ font-size:22px; font-weight:700; color:{PRIMARY_COLOR}; margin-top:2px; }}
      .df-t-spark {{ height:48px; }}
      /* section header */
      .df-section {{ display:flex; align-items:center; justify-content:space-between; margin:2px 2px 8px; }}
      .df-section h3 {{ margin:0; color:{PRIMARY_COLOR}; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------
# Demo Data & Column Expectations
# ---------------------------
EXPECTED_COLUMNS = {
    "order_id": ["order_id", "Order ID", "OrderID"],
    "priority": ["priority", "Priority", "Urgency"],                 # STAT / Urgent / Routine
    "reason_code": ["reason_code", "Reason", "AdjustmentReason"],
    "requested_at": ["requested_at", "RequestedAt", "OrderCreated"],
    "due_by": ["due_by", "SLA_DueBy", "DueBy"],
    "scheduled_start": ["scheduled_start", "ApptStart", "ScheduledStart"],
    "scheduled_end": ["scheduled_end", "ApptEnd", "ScheduledEnd"],
    "window_len_hrs": ["window_len_hrs", "WindowHours", "WindowLenHrs"],
    "channel": ["channel", "SchedulingChannel", "Source"],            # app / call_center / fax etc.
    "adjusted": ["adjusted", "WasAdjusted", "Adjusted"],
    "distance_km": ["distance_km", "DistanceKM", "Distance"],
    "patient_pref": ["patient_pref", "PatientPref", "PrefWindow"],    # am / pm / eve / none
    "resource_load": ["resource_load", "TechLoad", "RouteLoad"],      # 0..1 utilization estimate
    # New fields for routing + workflow
    "hospital_id": ["hospital_id", "FacilityID", "Site"],
    "patient_id": ["patient_id", "PatientID", "MRN"],
    "equipment_type": ["equipment_type", "Equipment", "DME"],
    "patient_lat": ["patient_lat", "PatientLat", "Lat"],
    "patient_lon": ["patient_lon", "PatientLon", "Lon"],
    "hospital_lat": ["hospital_lat", "HospitalLat", "SiteLat"],
    "hospital_lon": ["hospital_lon", "HospitalLon", "SiteLon"],
    "tech_skill": ["tech_skill", "TechSkill", "Capability"],
}

PRIORITY_ORDER = ["STAT", "Urgent", "Routine"]
PREF_ORDER = ["am", "pm", "eve", "none"]
CHANNELS = ["app", "call_center", "fax", "portal", "other"]
EQUIPMENT = ["oxygen_concentrator", "cpap", "bp_monitor", "walker", "wheelchair", "nebulizer"]
TECH_SKILLS = ["general", "respiratory", "mobility"]

# Simple equipment catalog with prep times, stock and required skill
EQUIPMENT_CATALOG = {
    "oxygen_concentrator": {"prep_min": 20, "stock": 14, "skill": "respiratory"},
    "cpap": {"prep_min": 25, "stock": 9, "skill": "respiratory"},
    "bp_monitor": {"prep_min": 10, "stock": 22, "skill": "general"},
    "walker": {"prep_min": 8, "stock": 18, "skill": "mobility"},
    "wheelchair": {"prep_min": 15, "stock": 11, "skill": "mobility"},
    "nebulizer": {"prep_min": 15, "stock": 12, "skill": "respiratory"},
}

# Demo technician roster
TECHNICIANS = [
    {"id": "T-101", "name": "Alex R.", "skill": "respiratory", "lat": 42.355, "lon": -71.065, "active_jobs": 3},
    {"id": "T-102", "name": "Brianna K.", "skill": "mobility", "lat": 42.381, "lon": -71.030, "active_jobs": 1},
    {"id": "T-103", "name": "Chris D.", "skill": "general", "lat": 42.340, "lon": -71.120, "active_jobs": 2},
    {"id": "T-104", "name": "Deepa S.", "skill": "respiratory", "lat": 42.470, "lon": -70.990, "active_jobs": 0},
]

HOSPITALS = [
    {"hospital_id": "DF-BOS-1", "name": "Dragonfly Boston Main", "lat": 42.361, "lon": -71.057},
    {"hospital_id": "DF-BOS-2", "name": "Dragonfly Boston West", "lat": 42.349, "lon": -71.120},
    {"hospital_id": "DF-BOS-3", "name": "Dragonfly North Shore", "lat": 42.480, "lon": -70.940},
]

np.random.seed(7)
random.seed(7)


def _coalesce(df: pd.DataFrame, name: str, candidates: list[str]):
    for c in candidates:
        if c in df.columns:
            return df[c]
    return None


def _jitter(base: float, spread: float = 0.06):
    return base + np.random.uniform(-spread, spread)


def make_synthetic(n_orders: int = 1500, start_date: datetime | None = None) -> pd.DataFrame:
    """Create a realistic-ish synthetic dataset for the demo, including hospital, patient, equipment, and locations."""
    if start_date is None:
        start_date = datetime.now() - timedelta(days=60)

    rows = []
    for i in range(n_orders):
        requested = start_date + timedelta(hours=np.random.randint(0, 24 * 60))
        priority = np.random.choice(PRIORITY_ORDER, p=[0.12, 0.28, 0.60])
        sla_hours = {"STAT": 4, "Urgent": 24, "Routine": 72}[priority]
        due_by = requested + timedelta(hours=sla_hours)

        start_offset = np.random.randint(1, min(72, sla_hours + 12))
        scheduled_start = requested + timedelta(hours=start_offset)
        window_len = int(np.random.choice([2, 4, 6], p=[0.45, 0.45, 0.10]))
        scheduled_end = scheduled_start + timedelta(hours=int(window_len))

        channel = np.random.choice(CHANNELS, p=[0.55, 0.30, 0.02, 0.10, 0.03])
        distance_km = np.random.gamma(3, 7)
        patient_pref = np.random.choice(PREF_ORDER, p=[0.35, 0.45, 0.12, 0.08])
        resource_load = max(0, min(1, np.random.normal(0.6, 0.2)))

        hosp = random.choice(HOSPITALS)
        equipment = np.random.choice(EQUIPMENT)
        tech_skill = np.random.choice(TECH_SKILLS)
        # place patients near greater Boston
        plat, plon = _jitter(42.36, 0.2), _jitter(-71.05, 0.2)

        base_risk = 0.15
        base_risk += 0.12 if window_len == 2 else (-0.05 if window_len == 6 else 0)
        base_risk += 0.08 if priority == "Routine" else (-0.05 if priority == "STAT" else 0)
        base_risk += 0.10 if resource_load > 0.75 else 0
        base_risk += 0.07 if distance_km > 40 else 0
        base_risk += 0.06 if channel == "call_center" else 0
        adjusted = np.random.binomial(1, max(0.01, min(0.95, base_risk)))

        rows.append({
            "order_id": f"DME-{100000 + i}",
            "priority": priority,
            "reason_code": np.random.choice(["missing_info", "patient_req", "routing_opt", "provider_sched", "insurance_hold", "none"], p=[0.15,0.30,0.22,0.18,0.05,0.10]),
            "requested_at": requested,
            "due_by": due_by,
            "scheduled_start": scheduled_start,
            "scheduled_end": scheduled_end,
            "window_len_hrs": window_len,
            "channel": channel,
            "adjusted": adjusted,
            "distance_km": distance_km,
            "patient_pref": patient_pref,
            "resource_load": resource_load,
            # new fields
            "hospital_id": hosp["hospital_id"],
            "patient_id": f"P-{600000 + i}",
            "equipment_type": equipment,
            "patient_lat": plat,
            "patient_lon": plon,
            "hospital_lat": hosp["lat"],
            "hospital_lon": hosp["lon"],
            "tech_skill": tech_skill,
        })
    df = pd.DataFrame(rows)
    return df


# ---------------------------
# Modeling & Scoring
# ---------------------------

def build_model(df: pd.DataFrame):
    features = [
        "priority", "reason_code", "channel", "patient_pref",
        "window_len_hrs", "distance_km", "resource_load",
        # Time deltas (numeric)
        "hours_to_due", "hours_from_request",
    ]

    df = df.copy()
    df["hours_to_due"] = (pd.to_datetime(df["due_by"]) - pd.to_datetime(df["scheduled_start"])) / pd.Timedelta(hours=1)
    df["hours_from_request"] = (pd.to_datetime(df["scheduled_start"]) - pd.to_datetime(df["requested_at"])) / pd.Timedelta(hours=1)

    X = df[features]
    y = df["adjusted"].astype(int)

    num_feats = ["window_len_hrs", "distance_km", "resource_load", "hours_to_due", "hours_from_request"]
    cat_feats = ["priority", "reason_code", "channel", "patient_pref"]

    preproc = ColumnTransformer([
        ("onehot", OneHotEncoder(handle_unknown="ignore"), cat_feats),
        ("pass", "passthrough", num_feats),
    ])

    model = Pipeline([
        ("prep", preproc),
        ("clf", LogisticRegression(max_iter=1000)),
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)

    y_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
    y_pred = (y_proba >= 0.5).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary")

    metrics = {"AUC": auc, "Precision": prec, "Recall": rec, "F1": f1}
    return model, metrics


def score_slots(order_row: pd.Series, candidate_slots: list[tuple[datetime, datetime]]):
    scored = []
    w_sla = 0.35; w_load = 0.20; w_dist = 0.15; w_pref = 0.15; w_risk = 0.15

    due_by = pd.to_datetime(order_row.get("due_by", datetime.now()))
    distance = float(order_row.get("distance_km", 20.0))
    resource_load = float(order_row.get("resource_load", 0.6))
    pref = str(order_row.get("patient_pref", "none"))

    def pref_bucket(dt: datetime):
        h = dt.hour
        if 8 <= h < 12: return "am"
        if 12 <= h < 17: return "pm"
        if 17 <= h <= 20: return "eve"
        return "none"

    for start, end in candidate_slots:
        hours_to_due = (due_by - start).total_seconds() / 3600.0
        sla_score = max(0.0, min(1.0, hours_to_due / 72))
        load_score = 1.0 - resource_load
        dist_score = max(0.0, min(1.0, 1 - (distance / 80.0)))
        pref_match = 1.0 if pref_bucket(start) == pref else (0.5 if pref == "none" else 0.0)
        window_len = (end - start).total_seconds() / 3600.0
        risk_penalty = (0.25 if hours_to_due < 12 else 0.0) + (0.15 if window_len <= 2 else 0.0)
        total = w_sla*sla_score + w_load*load_score + w_dist*dist_score + w_pref*pref_match - w_risk*risk_penalty
        scored.append({
            "slot_start": start, "slot_end": end, "score": round(total, 3),
            "explain": f"SLA:{sla_score:.2f} Load:{load_score:.2f} Dist:{dist_score:.2f} Pref:{pref_match:.2f} RiskPen:{risk_penalty:.2f}",
        })

    return sorted(scored, key=lambda x: x["score"], reverse=True)


# ---------------------------
# Landing helpers (sparklines + layout)
# ---------------------------

def _sparkline(values: pd.Series) -> alt.Chart:
    s = pd.DataFrame({"x": range(len(values)), "y": values})
    return (
        alt.Chart(s).mark_line()
        .encode(x=alt.X("x:Q", axis=None), y=alt.Y("y:Q", axis=None))
        .properties(height=48)
    )


def _pct(x):
    try:
        return f"{100*float(x):.1f}%"
    except Exception:
        return "–"


def render_landing(df: pd.DataFrame, model_metrics: dict):
    if df is None or len(df)==0:
        st.info("No data to summarize yet.")
        return
    adj_rate = float(np.clip(df["adjusted"].astype(int).mean(), 0, 1)) if "adjusted" in df else 0.0
    on_time = 1 - adj_rate
    stat_share = float((df["priority"]=="STAT").mean()) if "priority" in df else 0.0
    avg_dist = float(df.get("distance_km", pd.Series([0])).mean())
    med_window = float(pd.to_numeric(df.get("window_len_hrs", pd.Series([0]))).median())
    med_eta = int(np.median(np.clip(df.get("distance_km", pd.Series([18])).values/38*60 + 15, 10, 240)))

    st.markdown("<div class='df-section'><h3>Today at a glance</h3></div>", unsafe_allow_html=True)
    labels = ["On-time rate","Adj. rate","% STAT","Avg distance","Median ETA","Model AUC"]
    values = [
        _pct(on_time),
        _pct(adj_rate),
        _pct(stat_share),
        f"{avg_dist:.1f} km",
        f"{med_eta} min",
        f"{model_metrics.get('AUC',0):.2f}" if model_metrics else "—",
    ]

    def _trend(series, take=100):
        s = pd.to_numeric(series, errors="coerce").dropna().tail(take).reset_index(drop=True)
        if s.empty:
            s = pd.Series(np.random.normal(0,1,50)).cumsum()
        return s.rolling(5, min_periods=1).mean()

    trends = [
        _trend(1-df.get("adjusted", pd.Series(np.zeros(len(df)))).astype(float)),
        _trend(df.get("adjusted", pd.Series(np.zeros(len(df)))).astype(float)),
        _trend((df.get("priority", pd.Series(["Routine"]).repeat(len(df)))=="STAT").astype(float)),
        _trend(df.get("distance_km", pd.Series(np.random.normal(20,5,120)))),
        _trend(df.get("distance_km", pd.Series(np.random.normal(20,5,120)))/38*60+15),
        _trend(pd.Series(np.random.normal(model_metrics.get("AUC",0.7), 0.02, 120))),
    ]

    st.markdown("<div class='df-tiles'>", unsafe_allow_html=True)
    for label, value in zip(labels, values):
        st.markdown(
            f"<div class='df-tile'><div class='df-t-h'>{label}</div><div class='df-t-v'>{value}</div></div>",
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    cols = st.columns(6)
    for c, tr in zip(cols, trends):
        with c:
            st.altair_chart(_sparkline(tr), use_container_width=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    if "reason_code" in df.columns:
        top = df["reason_code"].value_counts().head(6).rename_axis("reason").reset_index(name="count")
        bar = (
            alt.Chart(top)
            .mark_bar()
            .encode(
                x=alt.X("reason:N", sort="-y", title="Reason"),
                y=alt.Y("count:Q", title="Adjustments"),
                tooltip=["reason","count"],
            )
            .properties(height=240)
        )
        st.altair_chart(bar, use_container_width=True)

# --- Executive Overview (always on top) ---
try:
    _df0 = raw_df if 'raw_df' in locals() else make_synthetic(800)
    _m0 = metrics if 'metrics' in locals() else {"AUC":0.70}
    render_landing(_df0, _m0)
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
except Exception as _e:
    pass

# ---------------------------
# Simple geo utilities
# ---------------------------

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c


def build_distance_matrix(points: list[tuple[float,float]]):
    n = len(points)
    M = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i==j: M[i][j]=0
            else:
                M[i][j] = int(haversine_km(points[i][0], points[i][1], points[j][0], points[j][1]) * 1000)  # meters for ortools
    return M


def solve_route(distance_matrix: list[list[int]]):
    n = len(distance_matrix)
    if not HAS_ORTOOLS or n <= 2:
        # Greedy fallback
        unvisited = list(range(1, n))
        route = [0]
        while unvisited:
            last = route[-1]
            nxt = min(unvisited, key=lambda j: distance_matrix[last][j])
            route.append(nxt)
            unvisited.remove(nxt)
        return route

    manager = pywrapcp.RoutingIndexManager(n, 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    def distance_cb(from_index, to_index):
        f = manager.IndexToNode(from_index)
        t = manager.IndexToNode(to_index)
        return distance_matrix[f][t]

    transit_cb_index = routing.RegisterTransitCallback(distance_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_cb_index)
    search_params = pywrapcp.DefaultRoutingSearchParameters()
    search_params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    solution = routing.SolveWithParameters(search_params)

    if solution:
        idx = routing.Start(0)
        route = []
        while not routing.IsEnd(idx):
            route.append(manager.IndexToNode(idx))
            idx = solution.Value(routing.NextVar(idx))
        route.append(manager.IndexToNode(idx))
        return route
    else:
        # Fallback greedy if solver fails
        unvisited = list(range(1, n))
        route = [0]
        while unvisited:
            last = route[-1]
            nxt = min(unvisited, key=lambda j: distance_matrix[last][j])
            route.append(nxt)
            unvisited.remove(nxt)
        return route


# ---------------------------
# Data Loader
# ---------------------------

def load_input_df(upload: io.BytesIO | None) -> pd.DataFrame:
    if upload is None:
        return make_synthetic()
    df = pd.read_csv(upload)

    mapped = {}
    for k, candidates in EXPECTED_COLUMNS.items():
        series = _coalesce(df, k, candidates)
        if series is None:
            if k in ("distance_km", "resource_load"):
                mapped[k] = np.random.uniform(5, 45, size=len(df)) if k == "distance_km" else np.random.uniform(0.4, 0.9, size=len(df))
            elif k == "window_len_hrs":
                mapped[k] = np.random.choice([2,4,6], size=len(df))
            elif k == "channel":
                mapped[k] = np.random.choice(CHANNELS, size=len(df))
            elif k == "patient_pref":
                mapped[k] = np.random.choice(PREF_ORDER, size=len(df))
            elif k in ("requested_at", "due_by", "scheduled_start", "scheduled_end"):
                now = datetime.now()
                mapped[k] = [now + timedelta(hours=int(i)%48) for i in range(len(df))]
            elif k == "priority":
                mapped[k] = np.random.choice(PRIORITY_ORDER, size=len(df))
            elif k == "adjusted":
                mapped[k] = np.random.binomial(1, 0.2, size=len(df))
            elif k in ("patient_lat","patient_lon"):
                mapped[k] = np.random.uniform(42.2, 42.5, size=len(df)) if k=="patient_lat" else np.random.uniform(-71.3, -70.9, size=len(df))
            elif k in ("hospital_lat","hospital_lon"):
                h = random.choice(HOSPITALS)
                mapped[k] = [h["lat"]]*len(df) if k=="hospital_lat" else [h["lon"]]*len(df)
            elif k == "hospital_id":
                mapped[k] = [random.choice(HOSPITALS)["hospital_id"] for _ in range(len(df))]
            elif k == "patient_id":
                mapped[k] = [f"P-{700000+i}" for i in range(len(df))]
            elif k == "equipment_type":
                mapped[k] = np.random.choice(EQUIPMENT, size=len(df))
            elif k == "tech_skill":
                mapped[k] = np.random.choice(TECH_SKILLS, size=len(df))
            else:
                mapped[k] = [f"UNK-{i}" for i in range(len(df))]
        else:
            mapped[k] = series

    out = pd.DataFrame(mapped)
    for c in ["requested_at", "due_by", "scheduled_start", "scheduled_end"]:
        out[c] = pd.to_datetime(out[c])
    return out


# ---------------------------
# ETA & Technician assignment helpers (AI-ish demo)
# ---------------------------

def estimate_eta_minutes(distance_km: float, prep_min: int, traffic: float, jobs_queue: int) -> int:
    """Rough ETA estimator (can be replaced by a trained regressor).
    travel speed baseline ~ 38 km/h, traffic multiplier ∈ [0.8, 1.6].
    """
    base_speed_kmh = 38.0 / max(0.5, min(2.0, traffic))
    travel_min = (distance_km / max(5.0, base_speed_kmh)) * 60.0
    queue_min = max(0, jobs_queue - 1) * 12  # ~12 min overhead per queued job
    return int(round(prep_min + travel_min + queue_min))


def best_technicians(patient_lat: float, patient_lon: float, required_skill: str, topn: int = 3# ---------------------------
# UI Helpers
# ---------------------------

def _ensure_state():
    for k, v in {
        "demo_running": False,
        "demo_step": 0,
        "report_payload": {},
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v

_ensure_state()

def _img_b64(path: str) -> str | None:
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception:
        return None


def render_navbar():
    mark_b64 = _img_b64(LOGO_MARK_PATH) if os.path.exists(LOGO_MARK_PATH) else None
    title_html = "Dragonfly Health — AI Scheduling & Order Coordination"
    if mark_b64:
        st.markdown(
            f"""
            <div class='df-nav'>IANS:
  # ---------------------------
# UI Helpers
# ---------------------------

def # ---# ---------------------------
# UI Helpers
# ---------------------------

def _ensure_state():
    for k, v in {
        "demo_running": False,
        "demo_step": 0,
        "report_payload": {},
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v

_ensure_state()

def _img_b64(path: str) -> str | None:
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception:
        return None


def render_navbar():
    mark_b64 = _img_b64(LOGO_MARK_PATH) if os.path.exists(LOGO_MARK_PATH) else None
    title_html = "Dragonfly Health — AI Scheduling & Order Coordination"
    if mark_b64:
        st.markdown(
            f"""
            <div class='df-nav'>
              <img class='df-mark' src='data:image/jpeg;base64,{mark_b64}' />
              <div class='df-title'>{title_html}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class='df-nav'>
              <div class='df-title'>{title_html}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def kpi(label: str, value: str, helptext: str = "-------------------------

def _ensure_state():
    for k, v in {
        "demo_running": False,
        "demo_step": 0,
        "report_payload": {},
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v

_ensure_state()

def _img_b64(path: str) -> str | None:
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception:
        return None


def render_navbar():
    mark_b64 = _img_b64(LOGO_MARK_PATH) if os.path.exists(LOGO_MARK_PATH) else None
    title_html = "Dragonfly Health — AI Scheduling & Order Coordination"
    if mark_b64:
        st.markdown(
            f"""
            <div class='df-nav'>
              <img class='df-mark' src='data:image/jpeg;base64,{mark_b64}' />
              <div class='df-title'>{title_html}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class='df-nav'>
              <div class='df-title'>{title_html}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def kpi(label: str, value: str, helptext: str = ""):
    st.markdown(
        f"""
        <div class="card">
          <div style="font-size:13px;color:#5f6b7a">{label}</div>
          <div style="font-size:28px;font-weight:700;margin-top:4px" class="brand">{value}</div>
          <div class="subtle" style="margin-top:4px">{helptext}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def datetime_picker(label_prefix: str, default_dt: datetime) -> datetime:
    """Streamlit-compatible date+time inputs (since st.datetime_input isn't universal)."""
    c1, c2 = st.columns(2)
    with c1:
        d = st.date_input(f"{label_prefix} — Date", value=default_dt.date())
    with c2:
        t = st.time_input(f"{label_prefix} — Time", value=default_dt.time())
    return datetime.combine(d, t)


# -------- Demo runner & report builder ---------

def run_demo():
    """Walk through intake → recommendations → ETA/tech, with animated status."""
    st.session_state.demo_running = True
    st.session_state.demo_step = 0
    with st.status("Playing demo…", expanded=True) as status:
        status.update(label="Loading synthetic data", state="running")
        st.session_state.demo_step = 1
        time.sleep(0.5)

        status.update(label="Training lightweight risk model", state="running")
        st.session_state.demo_step = 2
        time.sleep(0.6)

        status.update(label="Generating recommended slots", state="running")
        st.session_state.demo_step = 3
        time.sleep(0.6)

        status.update(label="Scoring ETA and recommending technicians", state="running")
        st.session_state.demo_step = 4
        time.sleep(0.6)

        status.update(label="Done — explore tabs 2 & 6 to see the prefilled scenario", state="complete")
    st.toast("Demo ready: open 'Order Intake + Recommender' then 'Equipment • ETA • Technicians'", icon="✅")


def build_report_html(payload: dict) -> bytes:
    """Create a lightweight HTML report that can be saved as PDF via browser print."""
    def esc(x):
        try:
            return str(x).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        except Exception:
            return str(x)

    css = f"""
    <style>
      body {{ font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 20px; }}
      h1 {{ color:{PRIMARY_COLOR}; }}
      .k {{ color:{ACCENT_COLOR}; font-weight:700; }}
      .card {{ border:1px solid #e6f0ec; border-radius:12px; padding:12px; margin:10px 0; }}
      .muted {{ color:#47606b; font-size:13px; }}
    </style>
    """
    html = [css, f"<h1>Dragonfly Health — Scheduling Run</h1>"]
    sec = payload
    html.append("<div class='card'><div class='muted'>Order</div>" \
                f"<div><span class='k'>Order ID:</span> {esc(sec.get('order_id',''))}</div>" \
                f"<div><span class='k'>Priority:</span> {esc(sec.get('priority',''))} • <span class='k'>Equip:</span> {esc(sec.get('equipment',''))}</div>" \
                f"<div><span class='k'>Due by:</span> {esc(sec.get('due_by',''))}</div>" \
                f"<div><span class='k'>Window:</span> {esc(sec.get('window_len',''))} hrs</div></div>")

    html.append("<div class='card'><div class='muted'>Recommended Slot</div>" \
                f"<div><span class='k'>Start:</span> {esc(sec.get('slot_start',''))} → <span class='k'>End:</span> {esc(sec.get('slot_end',''))}</div>" \
                f"<div><span class='k'>Score:</span> {esc(sec.get('score',''))} • <span class='k'>Risk:</span> {esc(sec.get( "demo_running": False,
        "demo_step": 0,
        "report_payload": {},
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v

_ensure_state()

def _img_b64(path: str) -> str | None:
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception:
        return None


def render_navbar():
    mark_b64 = _img_b64(LOGO_MARK_PATH) if os.path.exists(LOGO_MARK_PATH) else None
    title_html = "Dragonfly Health — AI Scheduling & Order Coordination"
    if mark_b64:
        st.markdown(
            f"""
            <div class='df-nav'>d_skill) or (required_skill == "general")
        dist = haversine_km(patient_lat, pat# ---# ---------------------------
# UI Helpers
# ---------------------------

def _ensure_state():
    for k, v in {
        "demo_running": False,
        "demo_step": 0,
        "report_payload": {},
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v

_ensure_state()

def _img_b64(path: str) -> str | None:
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception:
        return None


def render_navbar():
    mark_b64 = _img_b64(LOGO_MARK_PATH) if os.path.exists(LOGO_MARK_PATH) else None
    title_html = "Dragonfly Health — AI Scheduling & Order Coordination"
    if mark_b64:
        st.markdown(
            f"""
            <div class='df-nav'>
              <img class='df-mark' src='data:image/jpeg;base64,{mark_b64}' />
              <div class='df-title'>{title_html}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class='df-nav'>
              <div class='df-title'>{title_html}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def kpi(label: str, value: str, helptext: str = "-------------------------

def _ensure_state():
    for k, v in {
        "demo_running": False,
        "demo_step": 0,
        "report_payload": {},
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v

_ensure_state()

def _img_b64(path: str) -> str | None:
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception:
        return None


def render_navbar():
    mark_b64 = _img_b64(LOGO_MARK_PATH) if os.path.exists(LOGO_MARK_PATH) else None
    title_html = "Dragonfly Health — AI Scheduling & Order Coordination"
    if mark_b64:
        st.markdown(
            f"""
            <div class='df-nav'>
              <img class='df-mark' src='data:image/jpeg;base64,{mark_b64}' />
              <div class='df-title'>{title_html}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class='df-nav'>
              <div class='df-title'>{title_html}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def kpi(label: str, value: str, helptext: str = ""):
    st.markdown(
        f"""
        <div class="card">
          <div style="font-size:13px;color:#5f6b7a">{label}</div>
          <div style="font-size:28px;font-weight:700;margin-top:4px" class="brand">{value}</div>
          <div class="subtle" style="margin-top:4px">{helptext}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def datetime_picker(label_prefix: str, default_dt: datetime) -> datetime:
    """Streamlit-compatible date+time inputs (since st.datetime_input isn't universal)."""
    c1, c2 = st.columns(2)
    with c1:
        d = st.date_input(f"{label_prefix} — Date", value=default_dt.date())
    with c2:
        t = st.time_input(f"{label_prefix} — Time", value=default_dt.time())
    return datetime.combine(d, t)


# -------- Demo runner & report builder ---------

def run_demo():
    """Walk through intake → recommendations → ETA/tech, with animated status."""
    st.session_state.demo_running = True
    st.session_state.demo_step = 0
    with st.status("Playing demo…", expanded=True) as status:
        status.update(label="Loading synthetic data", state="running")
        st.session_state.demo_step = 1
        time.sleep(0.5)

        status.update(label="Training lightweight risk model", state="running")
        st.session_state.demo_step = 2
        time.sleep(0.6)

        status.update(label="Generating recommended slots", state="running")
        st.session_state.demo_step = 3
        time.sleep(0.6)

        status.update(label="Scoring ETA and recommending technicians", state="running")
        st.session_state.demo_step = 4
        time.sleep(0.6)

        status.update(label="Done — explore tabs 2 & 6 to see the prefilled scenario", state="complete")
    st.toast("Demo ready: open 'Order Intake + Recommender' then 'Equipment • ETA • Technicians'", icon="✅")


def build_report_html(payload: dict) -> bytes:
    """Create a lightweight HTML report that can be saved as PDF via browser print."""
    def esc(x):
        try:
            return str(x).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        except Exception:
            return str(x)

    css = f"""
    <style>
      body {{ font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 20px; }}
      h1 {{ color:{PRIMARY_COLOR}; }}
      .k {{ color:{ACCENT_COLOR}; font-weight:700; }}
      .card {{ border:1px solid #e6f0ec; border-radius:12px; padding:12px; margin:10px 0; }}
      .muted {{ color:#47606b; font-size:13px; }}
    </style>
    """
    html = [css, f"<h1>Dragonfly Health — Scheduling Run</h1>"]
    sec = payload
    html.append("<div class='card'><div class='muted'>Order</div>" \
                f"<div><span class='k'>Order ID:</span> {esc(sec.get('order_id',''))}</div>" \
                f"<div><span class='k'>Priority:</span> {esc(sec.get('priority',''))} • <span class='k'>Equip:</span> {esc(sec.get('equipment',''))}</div>" \
                f"<div><span class='k'>Due by:</span> {esc(sec.get('due_by',''))}</div>" \
                f"<div><span classs
# ---------------------------

def _ensure_state():
    for k, v in {
        "demo_running": False,
        "demo_step": 0,
        "report_payload": {},
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v

_ensure_state()

def _img_b64(path: str) -> str | None:
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception:
        return None


def render_navbar():
    mark_b64 = _img_b64(LOGO_MARK_PATH) if os.path.exists(LOGO_MARK_PATH) else None
    title_html = "Dragonfly Health — AI Scheduling & Order Coordination"
    if mark_b64:
        st.markdown(
            f"""
            <div class='df-nav'>
              <img class='df-mark' src='data:image/jpeg;base64,{mark_b64}' />
              <div class='df-title'>{title_html}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class='df-nav'>
              <div class='df-title'>{title_html}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def kpi(label: str, value: str, helptext: str = ""):
    st.markdown(
        f"""
        <div class="card">
          <div style="font-size:13px;color:#5f6b7a">{label}</div>
          <div style="font-size:28px;font-weight:700;margin-top:4px" class="brand">{value}</div>
          <div class="subtle" style="margin-top:4px">{helptext}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def datetime_picker(label_prefix: str, default_dt: datetime) -> datetime:
    """Streamlit-compatible date+time inputs (since st.datetime_input isn't universal)."""
    c1, c2 = st.columns(2)
    with c1:
        d = st.date_input(f"{label_prefix} — Date", value=default_dt.date())
    with c2:
        t = st.time_input(f"{label_prefix} — Time", value=default_dt.time())
    return datetime.combine(d, t)


# -------- Demo runner & report builder ---------

def run_demo():
    """Walk through intake → recommendations → ETA/tech, with animated status."""
    st.session_state.demo_running = True
    st.session_state.demo_step = 0
    with st.status("Playing demo…", expanded=True) as status:
        status.update(label="Loading synthetic data", state="running")
        st.session_state.demo_step = 1
        time.sleep(0.5)

        status.update(label="Training lightweight risk model", state="running")
        st.session_state.demo_step = 2
        time.sleep(0.6)

        status.update(label="Generating recommended slots", state="running")
        st.session_state.demo_step = 3
        time.sleep(0.6)

        status.update(label="Scoring ETA and recommending technicians", state="running")
        st.session_state.demo_step = 4
        time.sleep(0.6)

        status.update(label="Done — explore tabs 2 & 6 to see the prefilled scenario", state="complete")
    st.toast("Demo ready: open 'Order Intake + Recommender' then 'Equipment • ETA • Technicians'", icon="✅")


def build_report_html(payload: dict) -> bytes:
    """Create a lightweight HTML report that can be saved as PDF via browser print."""
    def esc(x):
        try:
            return str(x).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        except Exception:
            return str(x)

    css = f"""
    <style>
      body {{ font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 20px; }}
      h1 {{ color:{PRIMARY_COLOR}; }}
      .k {{ color:{ACCENT_COLOR}; font-weight:700; }}
      .card {{ border:1px solid #e6f0ec; border-radius:12px; padding:12px; margin:10px 0; }}
      .muted {{ color:#47606b; font-size:13px; }}
    </style>
    """
    html = [css, f"<h1>Dragonfly Health — Scheduling Run</h1>"]
    sec = payload
    html.append("<div class='card'><div class='muted'>Order</div>" \
                f"<div><span class='k'>Order ID:</span> {esc(sec.get('order_id',''))}</div>" \
                f"<div><span class='k'>Priority:</span> {esc(sec.get('priority',''))} • <span class='k'>Equip:</span> {esc(sec.get('equipment',''))}</div>" \
                f"<div><span class='k'>Due by:</span> {esc(sec.get('due_by',''))}</div>" \
                f"<div><span class='k'>Window:</span> {esc(sec.get('window_len',''))} hrs</div></div>")

    html.append("<div class='card'><div class='muted'>Recommended Slot</div>" \
                f"<div><span class='k'>Start:</span> {esc(sec.get('slot_start',''))} → <span class='k'>End:</span> {esc(sec.get('slot_end',''))}</div>" \
                f"<div><span class='k'>Score:</span> {esc(sec.get('score',''))} • <span class='k
# ---------------------------

def _ensure_state():
    for k, v in {
        "demo_running": False,
        "demo_step": 0,
        "report_payload": {},
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v

_ensure_state()

def _img_b64(path: str) -> str | None:
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception:
        return None


def render_navbar():
    mark_b64 = _img_b64(LOGO_MARK_PATH) if os.path.exists(LOGO_MARK_PATH) else None
    title_html = "Dragonfly Health — AI Scheduling & Order Coordination"
    if mark_b64:
        st.markdown(
            f"""
            <div class='df-nav'> load_penalty = tech["active_jobs"] * 0.8
     # ---------------------------
# UI Help# ---# ---------------------------
# UI Helpers
# ---------------------------
