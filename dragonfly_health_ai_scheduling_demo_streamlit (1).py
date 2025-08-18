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
- New in this version: **Hospital → Patient Profile → Equipment → Slot & Route** flow, and a **VRP‑lite** routing tab.
- The code is organized for clarity rather than micro-optimizations; swap the model and scoring functions
  with your production counterparts later.

Branding
--------
- Edit `PRIMARY_COLOR`, `ACCENT_COLOR`, logos, and titles to match Dragonfly branding.

"""

from __future__ import annotations
import os
import io
import math
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from dateutil import tz

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
PRIMARY_COLOR = "#1D3557"  # Deep blue
ACCENT_COLOR = "#06D6A0"    # Green
WARN_COLOR = "#FFD166"      # Amber
ALERT_COLOR = "#EF476F"     # Pink/Red
LIGHT_BG = "#F7FAFC"

st.set_page_config(
    page_title="Dragonfly Health — AI Scheduling Demo",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    f"""
    <style>
      .main {{ background: {LIGHT_BG}; }}
      .stApp header {{ background: white; border-bottom: 1px solid #eee; }}
      .metric-row {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }}
      .pill {{ display:inline-block; padding:4px 10px; border-radius:20px; background:{PRIMARY_COLOR}; color:white; font-size:12px; }}
      .brand {{ color: {PRIMARY_COLOR}; }}
      .accent {{ color: {ACCENT_COLOR}; }}
      .warn {{ color: {WARN_COLOR}; }}
      .alert {{ color: {ALERT_COLOR}; }}
      .card {{ background:white; padding:16px; border-radius:16px; box-shadow: 0 2px 10px rgba(0,0,0,0.05); }}
      .subtle {{ color:#5f6b7a; font-size:13px; }}
      .divider {{ height:1px; background:#eee; margin:8px 0 16px; }}
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
        start_date = datetime.now(tz.tzlocal()) - timedelta(days=60)

    rows = []
    for i in range(n_orders):
        requested = start_date + timedelta(hours=np.random.randint(0, 24 * 60))
        priority = np.random.choice(PRIORITY_ORDER, p=[0.12, 0.28, 0.60])
        sla_hours = {"STAT": 4, "Urgent": 24, "Routine": 72}[priority]
        due_by = requested + timedelta(hours=sla_hours)

        start_offset = np.random.randint(1, min(72, sla_hours + 12))
        scheduled_start = requested + timedelta(hours=start_offset)
        window_len = np.random.choice([2, 4, 6], p=[0.45, 0.45, 0.10])
        scheduled_end = scheduled_start + timedelta(hours=window_len)

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
# UI Helpers
# ---------------------------

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


# ---------------------------
# Sidebar — Controls
# ---------------------------
with st.sidebar:
    st.markdown(f"# 🧭 <span class='brand'>AI Scheduling</span>", unsafe_allow_html=True)
    st.caption("Demo: Automated order coordination for DME deliveries")

    data_mode = st.radio("Data Source", ["Demo (synthetic)", "Upload CSV"], index=0)
    upload_buf = None
    if data_mode == "Upload CSV":
        upload_buf = st.file_uploader("Upload CSV export", type=["csv"])  

    st.markdown("---")
    st.subheader("Model Settings")
    threshold = st.slider("Adjustment risk threshold (flag if ≥)", 0.05, 0.95, 0.5, 0.05)
    top_k = st.slider("# recommended slots", 1, 5, 3)
    st.markdown("---")
    st.subheader("Window Defaults")
    default_windows = st.select_slider("Allowed window lengths (hrs)", options=[2,4,6], value=(2,6))


# ---------------------------
# Main — Data & Model
# ---------------------------
raw_df = load_input_df(upload_buf)
model, metrics = build_model(raw_df)

st.markdown("""
# 🚑 Dragonfly Health — AI Scheduling & Order Coordination

A lightweight showcase of how AI can reduce *reschedules*, protect *SLA compliance*, and improve *route efficiency* for DME deliveries.
""")

with st.container():
    cols = st.columns(4)
    cols[0].markdown("### Metrics")
    cols[0].markdown("Model AUC")
    kpi("Model AUC", f"{metrics['AUC']:.2f}", "Probability of distinguishing adjusted vs not-adjusted")
    kpi("Precision", f"{metrics['Precision']:.2f}")
    kpi("Recall", f"{metrics['Recall']:.2f}")
    kpi("F1", f"{metrics['F1']:.2f}")

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# ---------------------------
# Tabs
# ---------------------------
_tab1, _tab2, _tab3, _tab4, _tab5 = st.tabs([
    "📊 Portfolio & Risk Heatmap", "📝 Order Intake + Recommender", "⚙️ What‑If Simulator", "📈 Ops Dashboard", "🚚 Routing (VRP‑lite)",
])

# ---------------------------
# Tab 1: Portfolio View & Risk Heatmap
# ---------------------------
with _tab1:
    st.subheader("Risk Distribution & Drivers")

    df = raw_df.copy()
    feats_df = df[[
        "priority", "reason_code", "channel", "patient_pref",
        "window_len_hrs", "distance_km", "resource_load",
    ]].copy()
    feats_df["hours_to_due"] = (df["due_by"] - df["scheduled_start"]) / pd.Timedelta(hours=1)
    feats_df["hours_from_request"] = (df["scheduled_start"] - df["requested_at"]) / pd.Timedelta(hours=1)
    df["risk"] = model.predict_proba(feats_df)[:,1]
    df["flagged"] = (df["risk"] >= threshold).astype(int)

    left, right = st.columns([2,1])
    with left:
        hist = alt.Chart(df).mark_bar().encode(
            x=alt.X("risk:Q", bin=alt.Bin(maxbins=30), title="Predicted adjustment probability"),
            y=alt.Y("count():Q", title="Orders"),
            tooltip=["count()"]
        ).properties(height=280)
        st.altair_chart(hist, use_container_width=True)

    with right:
        kpi("Orders (total)", f"{len(df):,}")
        kpi("Flagged ≥ threshold", f"{df['flagged'].sum():,}")
        kpi("Avg distance (km)", f"{df['distance_km'].mean():.1f}")
        kpi("Avg window (hrs)", f"{df['window_len_hrs'].mean():.1f}")

    st.markdown("### Heatmap by Priority × Window Length")
    heat = df.groupby(["priority", "window_len_hrs"]).agg(risk=("risk","mean")).reset_index()
    heatmap = alt.Chart(heat).mark_rect().encode(
        x=alt.X("window_len_hrs:O", title="Window length (hrs)"),
        y=alt.Y("priority:N", sort=PRIORITY_ORDER),
        color=alt.Color("risk:Q", scale=alt.Scale(scheme="redyellowgreen", domain=[1,0])),
        tooltip=["priority","window_len_hrs","risk"]
    ).properties(height=220)
    st.altair_chart(heatmap, use_container_width=True)


# ---------------------------
# Tab 2: Order Intake + Recommender
# ---------------------------
with _tab2:
    st.subheader("Create Order & Get Recommended Slots")

    sample_ids = st.multiselect("Prefill from order IDs (optional)", options=list(raw_df["order_id"].head(50)), default=[])
    if sample_ids:
        base_order = raw_df[raw_df.order_id.isin(sample_ids)].iloc[0]
    else:
        base_order = raw_df.sample(1).iloc[0]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        order_id = st.text_input("Order ID", value=str(base_order["order_id"]))
        priority = st.selectbox("Priority", PRIORITY_ORDER, index=PRIORITY_ORDER.index(base_order["priority"]))
        patient_pref = st.selectbox("Patient preference", PREF_ORDER, index=PREF_ORDER.index(base_order["patient_pref"]))
        equipment_type = st.selectbox("Equipment", EQUIPMENT, index=EQUIPMENT.index(base_order["equipment_type"]))
    with col2:
        distance_km = st.number_input("Distance (km)", min_value=0.0, max_value=300.0, value=float(base_order["distance_km"]))
        resource_load = st.slider("Resource load (0..1)", 0.0, 1.0, float(base_order["resource_load"]))
        channel = st.selectbox("Channel", CHANNELS, index=CHANNELS.index(base_order["channel"]))
        tech_skill = st.selectbox("Required skill", TECH_SKILLS, index=TECH_SKILLS.index(base_order["tech_skill"]))
    with col3:
        requested_at = st.datetime_input("Requested at", value=pd.to_datetime(base_order["requested_at"]).to_pydatetime())
        due_by = st.datetime_input("SLA due by", value=pd.to_datetime(base_order["due_by"]).to_pydatetime())
        window_len_hrs = st.select_slider("Window length (hrs)", options=[2,4,6], value=int(base_order["window_len_hrs"]))
    with col4:
        hospital_id = st.selectbox("Facility", [h["hospital_id"]+" — "+h["name"] for h in HOSPITALS], index=0)
        _, hname = hospital_id.split(" — ")
        reason_code = st.selectbox("Reason code", ["none","patient_req","routing_opt","provider_sched","missing_info","insurance_hold"], index=0)

    st.markdown("#### Patient Location (for routing)")
    colp1, colp2 = st.columns(2)
    with colp1:
        patient_lat = st.number_input("Patient lat", value=float(base_order["patient_lat"]))
    with colp2:
        patient_lon = st.number_input("Patient lon", value=float(base_order["patient_lon"]))

    start_base = datetime.now(tz.tzlocal()).replace(minute=0, second=0, microsecond=0) + timedelta(hours=2)
    starts = [start_base + timedelta(hours=h) for h in range(0, 72, window_len_hrs)]
    candidates = [(s, s + timedelta(hours=window_len_hrs)) for s in starts]

    order_row = pd.Series({
        "due_by": due_by,
        "distance_km": distance_km,
        "resource_load": resource_load,
        "patient_pref": patient_pref,
    })

    scored = score_slots(order_row, candidates)
    top = scored[:top_k]

    st.markdown("### Top Recommended Slots")
for i, rec in enumerate(top, 1):
    good = "✅" if rec["score"] >= 0.6 else "👍"
    st.markdown(
        f"""
        <div class='card'>
           <div style='display:flex;justify-content:space-between;align-items:center;'>
             <div>
                <span class='pill'>Rank {i}</span>
                <b>{rec['slot_start'].strftime('%a %b %d, %I:%M %p')}</b> → {rec['slot_end'].strftime('%I:%M %p')}
             </div>
             <div style='font-weight:700;color:{ACCENT_COLOR}'>
                Score {rec['score']:.2f} {good}
             </div>
           </div>
           <div class='subtle' style='margin-top:6px'>
              Why: {rec['explain']}
           </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("### Risk for Selected Slot")
    idx = st.select_slider("Pick a recommended slot to simulate risk", options=list(range(1, len(top)+1)), value=1)
    chosen = top[idx-1]

    feat_row = pd.DataFrame([{
        "priority": priority,
        "reason_code": reason_code,
        "channel": channel,
        "patient_pref": patient_pref,
        "window_len_hrs": window_len_hrs,
        "distance_km": distance_km,
        "resource_load": resource_load,
        "hours_to_due": (due_by - chosen["slot_start"]).total_seconds()/3600.0,
        "hours_from_request": (chosen["slot_start"] - requested_at).total_seconds()/3600.0,
    }])
    risk = float(model.predict_proba(feat_row)[:,1])

    k1, k2, k3 = st.columns(3)
    k1.metric("Predicted adjustment risk", f"{risk:.2%}")
    k2.metric("SLA margin (hrs)", f"{(due_by - chosen['slot_start']).total_seconds()/3600:.1f}")
    k3.metric("Window length (hrs)", f"{window_len_hrs}")


# ---------------------------
# Tab 3: What‑If Simulator
# ---------------------------
with _tab3:
    st.subheader("Policy What‑Ifs (Windows & Thresholds)")

    colA, colB = st.columns(2)
    with colA:
        st.markdown("**Scenario A — Current policy**")
        a_win = st.select_slider("Window length (A)", options=[2,4,6], value=2)
        a_thresh = st.slider("Risk flag threshold (A)", 0.05, 0.95, 0.5, 0.05)
    with colB:
        st.markdown("**Scenario B — Proposed policy**")
        b_win = st.select_slider("Window length (B)", options=[2,4,6], value=4)
        b_thresh = st.slider("Risk flag threshold (B)", 0.05, 0.95, 0.4, 0.05)

    def simulate(df: pd.DataFrame, window_len: int, thresh: float):
        tmp = df.copy()
        tmp["window_len_hrs"] = window_len
        feats = tmp[[
            "priority", "reason_code", "channel", "patient_pref",
            "window_len_hrs", "distance_km", "resource_load",
        ]].copy()
        feats["hours_to_due"] = (tmp["due_by"] - tmp["scheduled_start"]) / pd.Timedelta(hours=1)
        feats["hours_from_request"] = (tmp["scheduled_start"] - tmp["requested_at"]) / pd.Timedelta(hours=1)
        tmp["risk"] = model.predict_proba(feats)[:,1]
        tmp["flagged"] = (tmp["risk"] >= thresh).astype(int)
        flagged = tmp["flagged"].mean()
        est_reschedules = (tmp["risk"] >= thresh).mean() * 0.6 + (tmp["risk"] < thresh).mean() * 0.2
        avg_margin = ((tmp["due_by"] - tmp["scheduled_start"]) / pd.Timedelta(hours=1)).mean()
        return {"flag_rate": flagged, "est_reschedules": est_reschedules, "avg_sla_margin": avg_margin}

    a = simulate(raw_df, a_win, a_thresh)
    b = simulate(raw_df, b_win, b_thresh)

    g1, g2, g3 = st.columns(3)
    g1.metric("Flagged share (A → B)", f"{a['flag_rate']:.1%} → {b['flag_rate']:.1%}")
    g2.metric("Est. reschedule pressure (A → B)", f"{a['est_reschedules']:.2f} → {b['est_reschedules']:.2f}")
    g3.metric("Avg SLA margin hrs (A → B)", f"{a['avg_sla_margin']:.1f} → {b['avg_sla_margin']:.1f}")

    change = pd.DataFrame([
        {"Metric":"Flagged share","Scenario A":a['flag_rate'],"Scenario B":b['flag_rate']},
        {"Metric":"Reschedule pressure","Scenario A":a['est_reschedules'],"Scenario B":b['est_reschedules']},
        {"Metric":"Avg SLA margin (hrs)","Scenario A":a['avg_sla_margin'],"Scenario B":b['avg_sla_margin']},
    ])

    bar = alt.Chart(change).transform_fold(
        ["Scenario A","Scenario B"], as_=["Scenario","Value"]
    ).mark_bar().encode(
        x=alt.X("Value:Q"),
        y=alt.Y("Metric:N"),
        color=alt.Color("Scenario:N"),
        tooltip=["Metric","Scenario","Value"]
    ).properties(height=220)
    st.altair_chart(bar, use_container_width=True)


# ---------------------------
# Tab 4: Ops Dashboard
# ---------------------------
with _tab4:
    st.subheader("Real‑time Ops (Demo)")
    df = raw_df.copy()
    df["day"] = pd.to_datetime(df["scheduled_start"]).dt.date
    agg = df.groupby("day").agg(
        orders=("order_id","count"),
        avg_risk=("adjusted","mean"),
        avg_distance=("distance_km","mean"),
    ).reset_index()

    line1 = alt.Chart(agg).mark_line(point=True).encode(
        x=alt.X("day:T", title="Day"),
        y=alt.Y("orders:Q", title="Orders"),
        tooltip=["day","orders"]
    ).properties(height=220)

    line2 = alt.Chart(agg).mark_line(point=True).encode(
        x="day:T",
        y=alt.Y("avg_distance:Q", title="Avg Distance (km)"),
        tooltip=["day","avg_distance"]
    ).properties(height=220)

    c1, c2 = st.columns(2)
    c1.altair_chart(line1, use_container_width=True)
    c2.altair_chart(line2, use_container_width=True)

    st.markdown("#### By Channel")
    by_ch = raw_df.groupby("channel").agg(orders=("order_id","count")).reset_index()
    ch_bar = alt.Chart(by_ch).mark_bar().encode(
        x=alt.X("orders:Q", title="Orders"),
        y=alt.Y("channel:N", title="Channel"),
        tooltip=["channel","orders"]
    ).properties(height=200)
    st.altair_chart(ch_bar, use_container_width=True)


# ---------------------------
# Tab 5: Routing (VRP‑lite)
# ---------------------------
with _tab5:
    st.subheader("Route planning from facility → patients (demo)")
    st.caption("Pick a facility and generate a mini route with the current order + nearby patients.")

    # Facility pick
    h_opts = {h["hospital_id"]: h for h in HOSPITALS}
    h_sel = st.selectbox("Facility", options=list(h_opts.keys()), format_func=lambda x: f"{x} — {h_opts[x]['name']}")
    h_lat, h_lon = h_opts[h_sel]["lat"], h_opts[h_sel]["lon"]

    # Choose a seed order near the facility
    dfh = raw_df[raw_df["hospital_id"] == h_sel].copy()
    seed = dfh.sample(1).iloc[0] if not dfh.empty else raw_df.sample(1).iloc[0]

    st.markdown(f"**Seed order:** {seed['order_id']} · equipment: `{seed['equipment_type']}` · priority: `{seed['priority']}`")

    # Build a small set of stops (seed + 4 neighbors)
    def nearest_neighbors(df, center_lat, center_lon, k=4):
        df = df.copy()
        df["dist"] = df.apply(lambda r: haversine_km(center_lat, center_lon, r["patient_lat"], r["patient_lon"]), axis=1)
        return df.nsmallest(k, "dist")

    neighbors = nearest_neighbors(dfh, float(seed["patient_lat"]), float(seed["patient_lon"]), k=4)
    stops = pd.concat([seed.to_frame().T, neighbors]).drop_duplicates("order_id").head(5)

    # Points: 0 = facility, others = patients
    pts = [(h_lat, h_lon)] + list(zip(stops["patient_lat"].astype(float), stops["patient_lon"].astype(float)))
    distM = build_distance_matrix(pts)
    route_idx = solve_route(distM)

    # Present route
    route_labels = ["Facility"] + [f"{row.order_id} ({row.equipment_type})" for _, row in stops.iterrows()]
    pretty_route = [route_labels[i] for i in route_idx]

    st.markdown("### Suggested visit order")
    for step, label in enumerate(pretty_route, 1):
        st.markdown(f"{step}. **{label}**")

    total_km = sum(haversine_km(pts[route_idx[i]][0], pts[route_idx[i]][1], pts[route_idx[i+1]][0], pts[route_idx[i+1]][1]) for i in range(len(route_idx)-1))
    kpi("Total distance (km)", f"{total_km:.1f}")

    st.markdown("#### Map (lat/lon preview)")
    map_df = pd.DataFrame({
        "lat": [p[0] for p in pts],
        "lon": [p[1] for p in pts],
        "label": ["Facility"] + list(stops["order_id"]),
    })
    st.map(map_df.rename(columns={"lon":"longitude","lat":"latitude"}))

    st.info("For production: replace VRP‑lite with full OR‑Tools VRPTW (time windows, skills, capacities) and integrate with live calendars.")


st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

st.caption("© 2025 Dragonfly Health — Demo. For illustrative use only; not for clinical routing decisions.")
