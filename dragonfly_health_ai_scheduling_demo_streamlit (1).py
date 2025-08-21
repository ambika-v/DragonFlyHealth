# Dragonfly Health — AI Scheduling Demo (Streamlit)
# PART 1 of 2 — imports, theme, CSS, helpers, EXEC OVERVIEW, data+model

from __future__ import annotations
import os, io, math, random, base64
from pathlib import Path
from datetime import datetime, timedelta


import pydeck as pdk
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

# Optional: OR-Tools — we’ll fall back to greedy if not available (used in Part 2)
try:
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2
    HAS_ORTOOLS = True
except Exception:
    HAS_ORTOOLS = False

# ---------------------------
# Branding & Theme
# ---------------------------
PRIMARY_COLOR = "#114E7A"   # Dragonfly blue
ACCENT_COLOR  = "#14B58A"   # Dragonfly green
WARN_COLOR    = "#8FD3C8"   # muted teal
ALERT_COLOR   = "#2E8B57"   # deep green
LIGHT_BG      = "#F4FAF8"   # very light blue‑green

BASE_DIR = Path(__file__).resolve().parent
LOGO_PATH = ".devcontainer/assets/dragonflyhealth_logo.jpg"
LOGO_MARK_PATH = ".devcontainer/assets/dragonflyhealth_mark.jpg"

# Fallback auto-detect
if not os.path.exists(LOGO_PATH) or not os.path.exists(LOGO_MARK_PATH):
    _logo = LOGO_PATH if os.path.exists(LOGO_PATH) else None
    _mark = LOGO_MARK_PATH if os.path.exists(LOGO_MARK_PATH) else None
    for folder in [BASE_DIR/"assets", BASE_DIR.parent/"assets", Path("assets")]:
        for ext in ["png","jpg","jpeg","PNG","JPG","JPEG"]:
            if not _logo:
                c = folder / f"dragonflyhealth_logo.{ext}"
                if c.exists(): _logo = str(c)
            if not _mark:
                c2 = folder / f"dragonflyhealth_mark.{ext}"
                if c2.exists(): _mark = str(c2)
    LOGO_PATH = _logo or LOGO_PATH
    LOGO_MARK_PATH = _mark or LOGO_MARK_PATH

st.set_page_config(
    page_title="Dragonfly Health — AI Scheduling Demo",
    page_icon=LOGO_MARK_PATH if os.path.exists(LOGO_MARK_PATH) else "🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)
import streamlit as st
from datetime import datetime

# ---------- SIMPLE LOGIN GATE ----------
# 1) Add your allowed users here. Keys must match exactly what users type.
ALLOWED_USERS = {
    "Urvashi Patel": {"role": "pm"},
    "Ambika Varma": {"role": "owner"},
    # add more like:
    # "John Doe": {"role": "viewer"},
}

def _check_passcode(name: str, entered: str) -> bool:
    """
    Compares entered passcode to what's in st.secrets for this user.
    Create secrets like:
    [[secrets]]
    PASSCODES = {"Urvashi Patel": "1234", "Ambika Varma": "abcd"}
    """
    pc_map = st.secrets.get("PASSCODES", {})
    expected = pc_map.get(name)
    return bool(expected) and (entered == expected)

def login_gate():
    if st.session_state.get("authenticated"):
        return True  # already logged in

    st.markdown("### 🔐 Sign in")
    name = st.text_input("Your name (e.g., Urvashi Patel)")
    passcode = st.text_input("Passcode", type="password")
    go = st.button("Sign in")

    if go:
        if name in ALLOWED_USERS and _check_passcode(name, passcode):
            st.session_state.authenticated = True
            st.session_state.user_name = name
            st.session_state.user_role = ALLOWED_USERS[name]["role"]
            st.session_state.login_time = datetime.now().strftime("%Y-%m-%d %H:%M")
            st.success(f"Welcome, {name}!")
            st.rerun()
        else:
            st.error("Invalid name or passcode.")

    # Stop the rest of the app until they sign in
    st.stop()

def show_user_chip():
    u = st.session_state.get("user_name", "Guest")
    r = st.session_state.get("user_role", "")
    st.markdown(
        f"<div style='padding:6px 10px;border-radius:999px;background:#e6f7e6;"
        f"border:1px solid #b3e6cc;display:inline-block;color:#114E7A;'>"
        f"👤 {u} &nbsp;•&nbsp; {r}</div>",
        unsafe_allow_html=True,
    )
    if st.button("Logout"):
        for k in ("authenticated","user_name","user_role","login_time"):
            st.session_state.pop(k, None)
        st.rerun()

# ------ USE THE GATE ------
login_gate()   # <-- call this BEFORE building the rest of your UI
# (after this line, the user is signed in)

# ---------------------------
# Global CSS (escaped braces for f-strings)
# ---------------------------
st.markdown(
    f"""
    <style>
      /* Global app background */
      .stApp {{
        background-color: #e6f7e6 !important;  /* soft green */
      }}

      /* Main container */
      .main {{
        background-color: #e6f7e6 !important;
      }}

      /* Sidebar */
      section[data-testid="stSidebar"] {{
        background-color: #c8f2c8 !important;
      }}

      /* Header */
      .stApp header {{
        background: #c8f2c8 !important;
        border-bottom: 1px solid #99d699;
      }}

      /* KPI cards / tiles */
      .card, .df-tile {{
        background:#c8f2c8 !important;
        border-radius:16px;
        padding:16px;
        box-shadow:0 2px 10px rgba(0,0,0,0.05);
      }}

      /* Subtle text */
      .subtle {{ color:#2e5939; font-size:13px; }}

      /* Dividers */
      .divider {{ height:1px; background:#99d699; margin:8px 0 16px; }}

      /* Buttons */
      div.stButton>button:first-child {{
        background:{PRIMARY_COLOR}; color:white; border-radius:10px;
      }}

      /* Slider track tint */
      [data-baseweb="slider"]>div>div {{
        background:{ACCENT_COLOR}33;
      }}

      /* Scrollbar */
      *::-webkit-scrollbar {{ width:10px; height:10px; }}
      *::-webkit-scrollbar-thumb {{ background:{PRIMARY_COLOR}; border-radius:8px; }}

      /* Navbar */
      .df-nav {{
        background: linear-gradient(90deg, #0F6E86 0%, #14B58A 100%);
        color: white; padding: 10px 14px; border-radius: 12px;
        display:flex; align-items:center; gap:12px; margin-bottom: 10px;
      }}
      .df-nav .df-mark {{
        height: 40px; width:auto; border-radius:8px; background:#ffffff22; padding:4px;
      }}
      .df-nav .df-title {{ font-weight:700; font-size:18px; letter-spacing:0.3px; }}

      /* KPI grid */
      .df-tiles {{
        display:grid; grid-template-columns: repeat(6, minmax(0,1fr));
        gap:10px;
      }}
      .df-t-h {{ font-size:12px; color:#2e5939; }}
      .df-t-v {{ font-size:22px; font-weight:700; color:{PRIMARY_COLOR}; margin-top:2px; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------
# UI Helpers
# ---------------------------

def _ensure_state():
    for k, v in {"report_payload": {}}.items():
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
        <div class=\"card\">
          <div style=\"font-size:13px;color:#5f6b7a\">{label}</div>
          <div style=\"font-size:28px;font-weight:700;margin-top:4px\" class=\"brand\">{value}</div>
          <div class=\"subtle\" style=\"margin-top:4px\">{helptext}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def datetime_picker(label_prefix: str, default_dt: datetime) -> datetime:
    c1, c2 = st.columns(2)
    with c1:
        d = st.date_input(f"{label_prefix} — Date", value=default_dt.date())
    with c2:
        t = st.time_input(f"{label_prefix} — Time", value=default_dt.time())
    return datetime.combine(d, t)


# ---------------------------
# Executive Overview (visual)
# ---------------------------

def render_landing(df: pd.DataFrame, model_metrics: dict):
    """Executive overview with compact tiles + rich visuals (donut, trends, heatmap, capacity bar, micro-map)."""
    if df is None or len(df) == 0:
        st.info("No data yet.")
        return

    # KPIs
    adj_col = df.get("adjusted", pd.Series(np.zeros(len(df)))).astype(int)
    adj_rate = float(np.clip(adj_col.mean(), 0, 1))
    on_time = 1 - adj_rate
    stat_share = float((df.get("priority", pd.Series(["Routine"]).repeat(len(df))) == "STAT").mean())
    avg_dist = float(df.get("distance_km", pd.Series([0])).mean())
    med_eta = int(np.median(np.clip(df.get("distance_km", pd.Series([18])).values / 38 * 60 + 15, 10, 240)))

    st.markdown("<div class='df-section'><h3>Today at a glance</h3></div>", unsafe_allow_html=True)
    labels = ["On-time rate", "Adj. rate", "% STAT", "Avg distance", "Median ETA", "Model AUC"]
    values = [f"{on_time*100:.1f}%", f"{adj_rate*100:.1f}%", f"{stat_share*100:.1f}%", f"{avg_dist:.1f} km", f"{med_eta} min", f"{model_metrics.get('AUC', 0):.2f}"]

    colA, colB, colC = st.columns(3)
    with colA:
        kpi("On-time rate", values[0])
        kpi("Adj. rate", values[1])
    with colB:
        kpi("% STAT", values[2])
        kpi("Avg distance", values[3])
    with colC:
        kpi("Median ETA", values[4])
        kpi("Model AUC", values[5])
   

    # Row 1: Donut (on-time vs adjusted) + Daily volume
    c1, c2 = st.columns([1, 2])
    with c1:
        pie_df = pd.DataFrame({"status": ["On-time", "Adjusted"], "count": [int((1 - adj_col).sum()), int(adj_col.sum())]})
        pie = (
            alt.Chart(pie_df)
            .mark_arc(innerRadius=55)
            .encode(
                theta=alt.Theta("count:Q"),
                color=alt.Color("status:N", legend=None, scale=alt.Scale(range=[ACCENT_COLOR, PRIMARY_COLOR])),
                tooltip=["status", "count"],
            )
            .properties(height=210)
        )
        st.altair_chart(pie, use_container_width=True)
    with c2:
        dfx = df.copy()
        ref_ts = pd.to_datetime(dfx.get("scheduled_start", dfx.get("requested_at", datetime.now())))
        dfx["day"] = ref_ts.dt.date
        grp = dfx.groupby("day").agg(volume=("order_id", "count")).reset_index()
        line = (
            alt.Chart(grp)
            .mark_line(point=True)
            .encode(
                x=alt.X("day:T", title="Day"),
                y=alt.Y("volume:Q", title="Orders"),
                tooltip=[alt.Tooltip("day:T", title="Day"), "volume"],
            )
            .properties(height=210)
        )
        st.altair_chart(line, use_container_width=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # Row 2: Heatmap (adj rate by weekday x hour) + Top reasons
    c3, c4 = st.columns(2)
    with c3:
        dft = df.copy()
        ref_time = pd.to_datetime(dft.get("scheduled_start", dft.get("requested_at", datetime.now())))
        dft["hour"] = ref_time.dt.hour
        dft["weekday"] = ref_time.dt.day_name().str.slice(0, 3)
        heat = (
            alt.Chart(dft)
            .mark_rect()
            .encode(
                x=alt.X("hour:O", title="Hour of Day"),
                y=alt.Y("weekday:O", sort=["Mon","Tue","Wed","Thu","Fri","Sat","Sun"], title="Weekday"),
                color=alt.Color("mean(adjusted):Q", title="Adj. rate", scale=alt.Scale(scheme="greenblue")),
                tooltip=["weekday", "hour", alt.Tooltip("mean(adjusted):Q", title="Adj. rate", format=".1%")],
            )
            .properties(height=260)
        )
        st.altair_chart(heat, use_container_width=True)
    with c4:
        if "reason_code" in df.columns:
            top = df["reason_code"].fillna("unknown").value_counts().head(8).rename_axis("reason").reset_index(name="count")
            bar = (
                alt.Chart(top)
                .mark_bar()
                .encode(
                    x=alt.X("reason:N", sort="-y", title="Reason"),
                    y=alt.Y("count:Q", title="Adjustments"),
                    tooltip=["reason", "count"],
                )
                .properties(height=260)
            )
            st.altair_chart(bar, use_container_width=True)
        else:
            st.info("No reason_code field available.")

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # Row 3: Capacity bar + Micro map
    c5, c6 = st.columns([1, 1])
    with c5:
        sched_hours = float(pd.to_numeric(df.get("window_len_hrs", pd.Series([4]*len(df)))).sum())
        crews = max(5, int(len(df) / 60))  # demo heuristic
        avail_hours = crews * 8
        util = min(1.8, sched_hours / max(1.0, avail_hours))
        util_df = pd.DataFrame({"metric":["Capacity Utilization"], "value":[min(util, 1.5)]})
        util_bar = (
            alt.Chart(util_df)
            .mark_bar()
            .encode(x=alt.X("value:Q", scale=alt.Scale(domain=[0,1.2]), title="Utilization"), y=alt.Y("metric:N", title=""))
            .properties(height=70)
        )
        rule = alt.Chart(pd.DataFrame({"x":[1.0]})).mark_rule(color=PRIMARY_COLOR).encode(x="x:Q")
        st.altair_chart(util_bar + rule, use_container_width=True)
        st.caption("Blue line = 100% capacity")
    with c6:
        try:
            ref_ts = pd.to_datetime(df.get("scheduled_start", df.get("requested_at", datetime.now())))
            recent = df.assign(ts=ref_ts).sort_values("ts", ascending=False).head(40)
            map_df = pd.DataFrame({
                "latitude": recent["patient_lat"].astype(float),
                "longitude": recent["patient_lon"].astype(float),
            })
            st.map(map_df)
        except Exception:
            st.info("Map preview unavailable for this dataset.")

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# ---------------------------
# Data generation & loading
# ---------------------------
EXPECTED_COLUMNS = {
    "order_id": ["order_id","Order ID","OrderID"],
    "priority": ["priority","Priority","Urgency"],
    "reason_code": ["reason_code","Reason","AdjustmentReason"],
    "requested_at": ["requested_at","RequestedAt","OrderCreated"],
    "due_by": ["due_by","SLA_DueBy","DueBy"],
    "scheduled_start": ["scheduled_start","ApptStart","ScheduledStart"],
    "scheduled_end": ["scheduled_end","ApptEnd","ScheduledEnd"],
    "window_len_hrs": ["window_len_hrs","WindowHours","WindowLenHrs"],
    "channel": ["channel","SchedulingChannel","Source"],
    "adjusted": ["adjusted","WasAdjusted","Adjusted"],
    "distance_km": ["distance_km","DistanceKM","Distance"],
    "patient_pref": ["patient_pref","PatientPref","PrefWindow"],
    "resource_load": ["resource_load","TechLoad","RouteLoad"],
    "hospital_id": ["hospital_id","FacilityID","Site"],
    "patient_id": ["patient_id","PatientID","MRN"],
    "equipment_type": ["equipment_type","Equipment","DME"],
    "patient_lat": ["patient_lat","PatientLat","Lat"],
    "patient_lon": ["patient_lon","PatientLon","Lon"],
    "hospital_lat": ["hospital_lat","HospitalLat","SiteLat"],
    "hospital_lon": ["hospital_lon","HospitalLon","SiteLon"],
    "tech_skill": ["tech_skill","TechSkill","Capability"],
}

PRIORITY_ORDER = ["STAT","Urgent","Routine"]
PREF_ORDER = ["am","pm","eve","none"]
CHANNELS = ["app","call_center","fax","portal","other"]
EQUIPMENT = ["oxygen_concentrator","cpap","bp_monitor","walker","wheelchair","nebulizer"]
TECH_SKILLS = ["general","respiratory","mobility"]

EQUIPMENT_CATALOG = {
    "oxygen_concentrator": {"prep_min": 20, "stock": 14, "skill": "respiratory"},
    "cpap": {"prep_min": 25, "stock": 9, "skill": "respiratory"},
    "bp_monitor": {"prep_min": 10, "stock": 22, "skill": "general"},
    "walker": {"prep_min": 8, "stock": 18, "skill": "mobility"},
    "wheelchair": {"prep_min": 15, "stock": 11, "skill": "mobility"},
    "nebulizer": {"prep_min": 15, "stock": 12, "skill": "respiratory"},
}

TECHNICIANS = [
    {"id":"T-101","name":"Alex R.","skill":"respiratory","lat":42.355,"lon":-71.065,"active_jobs":3},
    {"id":"T-102","name":"Brianna K.","skill":"mobility","lat":42.381,"lon":-71.030,"active_jobs":1},
    {"id":"T-103","name":"Chris D.","skill":"general","lat":42.340,"lon":-71.120,"active_jobs":2},
    {"id":"T-104","name":"Deepa S.","skill":"respiratory","lat":42.470,"lon":-70.990,"active_jobs":0},
]

HOSPITALS = [
    {"hospital_id":"DF-BOS-1","name":"Dragonfly Boston Main","lat":42.361,"lon":-71.057},
    {"hospital_id":"DF-BOS-2","name":"Dragonfly Boston West","lat":42.349,"lon":-71.120},
    {"hospital_id":"DF-BOS-3","name":"Dragonfly North Shore","lat":42.480,"lon":-70.940},
]

np.random.seed(7); random.seed(7)


def _coalesce(df: pd.DataFrame, name: str, candidates: list[str]):
    for c in candidates:
        if c in df.columns: return df[c]
    return None


def _jitter(base: float, spread: float = 0.06):
    return base + np.random.uniform(-spread, spread)


def make_synthetic(n_orders: int = 1500, start_date: datetime | None = None) -> pd.DataFrame:
    if start_date is None:
        start_date = datetime.now() - timedelta(days=60)
    rows = []
    for i in range(n_orders):
        requested = start_date + timedelta(hours=np.random.randint(0, 24*60))
        priority = np.random.choice(PRIORITY_ORDER, p=[0.12,0.28,0.60])
        sla_hours = {"STAT":4, "Urgent":24, "Routine":72}[priority]
        due_by = requested + timedelta(hours=sla_hours)
        start_offset = np.random.randint(1, min(72, sla_hours+12))
        scheduled_start = requested + timedelta(hours=start_offset)
        window_len = int(np.random.choice([2,4,6], p=[0.45,0.45,0.10]))
        scheduled_end = scheduled_start + timedelta(hours=int(window_len))
        channel = np.random.choice(CHANNELS, p=[0.55,0.30,0.02,0.10,0.03])
        distance_km = np.random.gamma(3, 7)
        patient_pref = np.random.choice(PREF_ORDER, p=[0.35,0.45,0.12,0.08])
        resource_load = max(0, min(1, np.random.normal(0.6,0.2)))
        hosp = random.choice(HOSPITALS)
        equipment = np.random.choice(EQUIPMENT)
        tech_skill = np.random.choice(TECH_SKILLS)
        plat, plon = _jitter(42.36,0.2), _jitter(-71.05,0.2)
        base_risk = 0.15
        base_risk += 0.12 if window_len == 2 else (-0.05 if window_len == 6 else 0)
        base_risk += 0.08 if priority == "Routine" else (-0.05 if priority == "STAT" else 0)
        base_risk += 0.10 if resource_load > 0.75 else 0
        base_risk += 0.07 if distance_km > 40 else 0
        base_risk += 0.06 if channel == "call_center" else 0
        adjusted = np.random.binomial(1, max(0.01, min(0.95, base_risk)))
        rows.append({
            "order_id": f"DME-{100000+i}",
            "priority": priority,
            "reason_code": np.random.choice(["missing_info","patient_req","routing_opt","provider_sched","insurance_hold","none"], p=[0.15,0.30,0.22,0.18,0.05,0.10]),
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
            "hospital_id": hosp["hospital_id"],
            "patient_id": f"P-{600000+i}",
            "equipment_type": equipment,
            "patient_lat": plat,
            "patient_lon": plon,
            "hospital_lat": hosp["lat"],
            "hospital_lon": hosp["lon"],
            "tech_skill": tech_skill,
        })
    return pd.DataFrame(rows)


def load_input_df(upload: io.BytesIO | None) -> pd.DataFrame:
    if upload is None:
        return make_synthetic()
    df = pd.read_csv(upload)
    mapped = {}
    for k, candidates in EXPECTED_COLUMNS.items():
        series = _coalesce(df, k, candidates)
        if series is None:
            if k in ("distance_km","resource_load"):
                mapped[k] = np.random.uniform(5,45,len(df)) if k=="distance_km" else np.random.uniform(0.4,0.9,len(df))
            elif k == "window_len_hrs":
                mapped[k] = np.random.choice([2,4,6], size=len(df))
            elif k == "channel":
                mapped[k] = np.random.choice(CHANNELS, size=len(df))
            elif k == "patient_pref":
                mapped[k] = np.random.choice(PREF_ORDER, size=len(df))
            elif k in ("requested_at","due_by","scheduled_start","scheduled_end"):
                now = datetime.now(); mapped[k] = [now + timedelta(hours=int(i)%48) for i in range(len(df))]
            elif k == "priority":
                mapped[k] = np.random.choice(PRIORITY_ORDER, size=len(df))
            elif k == "adjusted":
                mapped[k] = np.random.binomial(1, 0.2, size=len(df))
            elif k in ("patient_lat","patient_lon"):
                mapped[k] = np.random.uniform(42.2,42.5,len(df)) if k=="patient_lat" else np.random.uniform(-71.3,-70.9,len(df))
            elif k in ("hospital_lat","hospital_lon"):
                h = random.choice(HOSPITALS); mapped[k] = [h["lat"]]*len(df) if k=="hospital_lat" else [h["lon"]]*len(df)
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
    for c in ["requested_at","due_by","scheduled_start","scheduled_end"]:
        out[c] = pd.to_datetime(out[c])
    return out

# ---------------------------
# Modeling & scoring (more in Part 2)
# ---------------------------

def build_model(df: pd.DataFrame):
    feats = [
        "priority","reason_code","channel","patient_pref",
        "window_len_hrs","distance_km","resource_load",
        "hours_to_due","hours_from_request",
    ]
    df = df.copy()
    df["hours_to_due"] = (pd.to_datetime(df["due_by"]) - pd.to_datetime(df["scheduled_start"])) / pd.Timedelta(hours=1)
    df["hours_from_request"] = (pd.to_datetime(df["scheduled_start"]) - pd.to_datetime(df["requested_at"])) / pd.Timedelta(hours=1)
    X = df[feats]
    y = df["adjusted"].astype(int)
    num = ["window_len_hrs","distance_km","resource_load","hours_to_due","hours_from_request"]
    cat = ["priority","reason_code","channel","patient_pref"]
    pre = ColumnTransformer([("onehot", OneHotEncoder(handle_unknown="ignore"), cat), ("pass","passthrough", num)])
    model = Pipeline([("prep", pre), ("clf", LogisticRegression(max_iter=1000))])
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(Xtr, ytr)
    yprob = model.predict_proba(Xte)[:,1]
    auc = roc_auc_score(yte, yprob)
    yhat = (yprob>=0.5).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(yte, yhat, average="binary")
    return model, {"AUC":auc, "Precision":prec, "Recall":rec, "F1":f1}


def score_slots(order_row: pd.Series, candidate_slots: list[tuple[datetime, datetime]]):
    scored = []
    w_sla=0.35; w_load=0.20; w_dist=0.15; w_pref=0.15; w_risk=0.15
    due_by = pd.to_datetime(order_row.get("due_by", datetime.now()))
    distance = float(order_row.get("distance_km", 20.0))
    resource_load = float(order_row.get("resource_load", 0.6))
    pref = str(order_row.get("patient_pref","none"))
    def pref_bucket(dt: datetime):
        h=dt.hour
        if 8<=h<12: return "am"
        if 12<=h<17: return "pm"
        if 17<=h<=20: return "eve"
        return "none"
    for start,end in candidate_slots:
        hours_to_due = (due_by - start).total_seconds()/3600.0
        sla_score = max(0.0, min(1.0, hours_to_due/72))
        load_score = 1.0 - resource_load
        dist_score = max(0.0, min(1.0, 1 - (distance/80.0)))
        pref_match = 1.0 if pref_bucket(start)==pref else (0.5 if pref=="none" else 0.0)
        window_len = (end-start).total_seconds()/3600.0
        risk_penalty = (0.25 if hours_to_due<12 else 0.0) + (0.15 if window_len<=2 else 0.0)
        total = w_sla*sla_score + w_load*load_score + w_dist*dist_score + w_pref*pref_match - w_risk*risk_penalty
        scored.append({
            "slot_start":start, "slot_end":end, "score":round(total,3),
            "explain":f"SLA:{sla_score:.2f} Load:{load_score:.2f} Dist:{dist_score:.2f} Pref:{pref_match:.2f} RiskPen:{risk_penalty:.2f}",
        })
    return sorted(scored, key=lambda x: x["score"], reverse=True)
# Dragonfly Health — AI Scheduling Demo (Streamlit)
# PART 2 of 2 — geo/routing, ETA/tech, report/export, full UI layout

# If Part 1 ran in same file, these exist already; otherwise import them from Part 1
try:
    HAS_ORTOOLS
except NameError:
    HAS_ORTOOLS = False

# ---------------------------
# Geo & Routing
# ---------------------------

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2-lat1); dlon = math.radians(lon2-lon1)
    a = math.sin(dlat/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R*c


def build_distance_matrix(points: list[tuple[float,float]]):
    n = len(points); M = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i==j: M[i][j]=0
            else: M[i][j] = int(haversine_km(points[i][0], points[i][1], points[j][0], points[j][1]) * 1000)
    return M

# ---------------------------
# Multi-route planning (K vehicles)
# ---------------------------
from typing import List, Dict
 
def plan_multi_routes_with_constraints(
    depot: tuple[float, float],
    stops: list[dict],
    num_routes: int = 2,
    capacity_per_vehicle: int = 6,
    enforce_skills: bool = True,
    use_time_windows: bool = True,
    speed_kmh: float = 38.0,
     ) -> dict:
    """
    Returns:
      {'routes': [[idx,...],...], 'labels': [...], 'points': [(lat,lon),...],
       'dist_km': [...], 'total_km': float}
    Index mapping:
      0 = depot; 1..N = stops in input order.
    """
    pts = [depot] + [(float(s["patient_lat"]), float(s["patient_lon"])) for s in stops]
    labels = ["Facility"] + [f'{s["order_id"]} ({s["equipment_type"]})' for s in stops]
    M = build_distance_matrix(pts)  # meters

    # Choose vehicles from TECHNICIANS (fewest active jobs first)
    techs_sorted = sorted(TECHNICIANS, key=lambda t: t.get("active_jobs", 0))
    k = max(1, int(num_routes))
    vehicles = techs_sorted[:k]
    vehicle_skills = [v.get("skill", "general") for v in vehicles]

    routes = []

    if HAS_ORTOOLS and len(pts) > 2:
        from ortools.constraint_solver import pywrapcp, routing_enums_pb2

        n = len(pts)
        starts = [0] * k
        ends   = [0] * k
        manager = pywrapcp.RoutingIndexManager(n, k, starts, ends)
        routing = pywrapcp.RoutingModel(manager)

        # service time proxy from equipment prep_min (put at "from" node)
        service_min = [0] + [int(s.get("service_min", EQUIPMENT_CATALOG.get(s["equipment_type"], {}).get("prep_min", 10))) for s in stops]

        def travel_time_cb(fi, ti):
            f = manager.IndexToNode(fi); t = manager.IndexToNode(ti)
            dist_m = M[f][t]
            travel_min = (dist_m / 1000.0) / max(5.0, float(speed_kmh)) * 60.0
            return int(round(travel_min + service_min[f]))

        time_idx = routing.RegisterTransitCallback(travel_time_cb)
        routing.SetArcCostEvaluatorOfAllVehicles(time_idx)

        # Capacity: 1 unit per stop
        def demand_cb(index):
            node = manager.IndexToNode(index)
            return 0 if node == 0 else 1
        demand_idx = routing.RegisterUnaryTransitCallback(demand_cb)
        caps = [int(capacity_per_vehicle)] * k
        routing.AddDimensionWithVehicleCapacity(demand_idx, 0, caps, True, "Capacity")

        # Time dimension
        horizon = 48 * 60  # minutes
        routing.AddDimension(time_idx, 60*12, horizon, True, "Time")
        time_dim = routing.GetDimensionOrDie("Time")

        if use_time_windows:
            for i, s in enumerate(stops, start=1):
                # prefer explicit tw fields; else derive from priority/prefs
                ws = int(s.get("tw_start", 0))
                we = int(s.get("tw_end", 12*60))
                time_dim.CumulVar(manager.NodeToIndex(i)).SetRange(ws, max(ws+30, we))
        # depot starts at time 0
        for v in range(k):
            time_dim.CumulVar(routing.Start(v)).SetRange(0, 0)

        # Skills: restrict vehicles that can visit a stop
        if enforce_skills:
            for i, s in enumerate(stops, start=1):
                required = s.get("skill_req", EQUIPMENT_CATALOG.get(s["equipment_type"], {}).get("skill", "general"))
                allowed = [v for v, sk in enumerate(vehicle_skills) if sk == required or required == "general" or sk == "general"]
                if allowed:
                    routing.SetAllowedVehiclesForIndex(allowed, manager.NodeToIndex(i))

        params = pywrapcp.DefaultRoutingSearchParameters()
        params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        params.time_limit.seconds = 3

        sol = routing.SolveWithParameters(params)
        if sol:
            for v in range(k):
                idx = routing.Start(v)
                route = []
                while not routing.IsEnd(idx):
                    route.append(manager.IndexToNode(idx))
                    idx = sol.Value(routing.NextVar(idx))
                route.append(manager.IndexToNode(idx))  # end at depot
                if len(route) > 2:
                    routes.append(route)

    # Fallback if OR-Tools missing or no routes created
    if not routes:
        try:
            from sklearn.cluster import KMeans
            import numpy as np
            if len(pts) <= 2 or k == 1:
                routes = [solve_route(M)]
            else:
                arr = np.array(pts[1:])
                km = KMeans(n_clusters=min(k, len(arr)), n_init=10, random_state=42)
                labs = km.fit_predict(arr)
                routes = []
                for cl in range(labs.max()+1):
                    idxs = [i+1 for i, lab in enumerate(labs) if lab == cl]
                    if not idxs: continue
                    sub = [0] + idxs
                    subM = [[M[a][b] for b in sub] for a in sub]
                    sub_route = solve_route(subM)
                    routes.append([sub[i] for i in sub_route])
        except Exception:
            routes = [list(range(len(pts)))]

    def seq_km(route):
        km = 0.0
        for i in range(len(route)-1):
            a, b = route[i], route[i+1]
            km += M[a][b]/1000.0
        return km

    rkms = [seq_km(r) for r in routes]
    return {"routes": routes, "labels": labels, "points": pts, "dist_km": rkms, "total_km": float(sum(rkms))}


def plan_multi_routes_from_points(
    depot: tuple[float, float],
    stops: List[Dict],
    num_routes: int = 2,
) -> Dict:
    """
    Returns {'routes': [ [indices...], ... ], 'labels': [...], 'points': [(lat,lon),...], 'dist_km': [..], 'total_km': float}
    Index 0 is always the depot. Each route is a list of point indices in visit order (including depot start/end).
    """

    pts = [depot] + [(float(r["patient_lat"]), float(r["patient_lon"])) for r in stops]
    labels = ["Facility"] + [f'{r["order_id"]} ({r["equipment_type"]})' for r in stops]
    M = build_distance_matrix(pts)  # uses your existing helper (meters)

    # If OR-Tools available: solve VRP with K vehicles starting/ending at depot
    if HAS_ORTOOLS and len(pts) > 2 and num_routes >= 1:
        from ortools.constraint_solver import pywrapcp, routing_enums_pb2

        n = len(pts)
        manager = pywrapcp.RoutingIndexManager(n, num_routes, 0)           # all start at depot 0
        routing = pywrapcp.RoutingModel(manager)

        def cb(fi, ti):
            f = manager.IndexToNode(fi); t = manager.IndexToNode(ti)
            return M[f][t]

        transit_idx = routing.RegisterTransitCallback(cb)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_idx)

        # All vehicles start & end at depot
        # for v in range(num_routes):
        #     routing.SetStartDepot(v, 0)
        #     routing.SetEndDepot(v, 0)

        # Balance routes a bit by adding a soft maximum number of nodes per vehicle
        # (not strictly necessary; helps avoid empty routes on small instances)
        # You can also add time windows / capacities here later.

        params = pywrapcp.DefaultRoutingSearchParameters()
        params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        params.time_limit.seconds = 2  # keep it snappy for the demo

        sol = routing.SolveWithParameters(params)
        routes = []
        if sol:
            for v in range(num_routes):
                idx = routing.Start(v)
                route = []
                while not routing.IsEnd(idx):
                    route.append(manager.IndexToNode(idx))
                    idx = sol.Value(routing.NextVar(idx))
                route.append(manager.IndexToNode(idx))  # end
                # Avoid returning trivial [0,0] routes
                if len(route) > 2:
                    routes.append(route)
        else:
            routes = []

        if not routes:
            # Fallback if solver returned empties
            routes = [list(range(0, len(pts)))]  # one greedy route as last resort

    else:
        # Fallback: KMeans cluster patients into K clusters, then greedy TSP inside each
        from sklearn.cluster import KMeans
        import numpy as np

        k = max(1, int(num_routes))
        if len(pts) <= 2 or k == 1:
            # Just one greedy route including all stops
            route = solve_route(M)  # your single-route greedy/ORTools helper
            routes = [route]
        else:
            arr = np.array(pts[1:])  # patients only
            km = KMeans(n_clusters=min(k, len(arr)), n_init=10, random_state=42)
            labels_k = km.fit_predict(arr)

            routes = []
            for cl in range(labels_k.max()+1):
                idxs = [i+1 for i, lab in enumerate(labels_k) if lab == cl]  # +1 due to depot at 0
                if not idxs:
                    continue
                # Build sub-matrix for [0] + cluster points
                sub_map = [0] + idxs
                subM = [[M[a][b] for b in sub_map] for a in sub_map]
                sub_route = solve_route(subM)  # returns indices in sub_map-space
                # Map back to original indices
                full_route = [sub_map[i] for i in sub_route]
                routes.append(full_route)

    # Compute distances
    def seq_km(route):
        km = 0.0
        for i in range(len(route)-1):
            a, b = route[i], route[i+1]
            km += M[a][b] / 1000.0
        return km

    rkms = [seq_km(r) for r in routes]
    return {
        "routes": routes,
        "labels": labels,
        "points": pts,
        "dist_km": rkms,
        "total_km": float(sum(rkms)),
    }
   
def solve_route(distance_matrix: list[list[int]]):
    n = len(distance_matrix)
    if not HAS_ORTOOLS or n <= 2:
        unvisited = list(range(1,n)); route=[0]
        while unvisited:
            last = route[-1]; nxt = min(unvisited, key=lambda j: distance_matrix[last][j])
            route.append(nxt); unvisited.remove(nxt)
        return route
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2
    manager = pywrapcp.RoutingIndexManager(n, 1, 0)
    routing = pywrapcp.RoutingModel(manager)
    def cb(fi, ti):
        f=manager.IndexToNode(fi); t=manager.IndexToNode(ti); return distance_matrix[f][t]
    idx = routing.RegisterTransitCallback(cb)
    routing.SetArcCostEvaluatorOfAllVehicles(idx)
    p = pywrapcp.DefaultRoutingSearchParameters()
    p.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    sol = routing.SolveWithParameters(p)
    if not sol:
        unvisited = list(range(1,n)); route=[0]
        while unvisited:
            last = route[-1]; nxt = min(unvisited, key=lambda j: distance_matrix[last][j])
            route.append(nxt); unvisited.remove(nxt)
        return route
    i = routing.Start(0); route=[]
    while not routing.IsEnd(i):
        route.append(manager.IndexToNode(i)); i = sol.Value(routing.NextVar(i))
    route.append(manager.IndexToNode(i)); return route

# ---------------------------
# ETA & Technicians
# ---------------------------

def estimate_eta_minutes(distance_km: float, prep_min: int, traffic: float, jobs_queue: int) -> int:
    base_speed_kmh = 38.0 / max(0.5, min(2.0, traffic))
    travel_min = (distance_km / max(5.0, base_speed_kmh)) * 60.0
    queue_min = max(0, jobs_queue - 1) * 12
    return int(round(prep_min + travel_min + queue_min))


def best_technicians(patient_lat: float, patient_lon: float, required_skill: str, topn: int = 3):
    scored=[]
    for tech in TECHNICIANS:
        skill_ok = (tech["skill"] == required_skill) or (required_skill == "general")
        dist = haversine_km(patient_lat, patient_lon, tech["lat"], tech["lon"])
        load_penalty = tech["active_jobs"] * 0.8
        score = (1.0/(1.0+dist)) - (0.05*load_penalty) + (0.2 if tech["skill"]==required_skill else 0.0)
        scored.append({"tech":tech, "dist_km":dist, "score":score, "ok":skill_ok})
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:topn]

# ---------------------------
# Report builder & Demo button
# ---------------------------

def build_report_html(payload: dict) -> bytes:
    css = f"""
    <style>
      body {{ font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 20px; }}
      h1 {{ color:{PRIMARY_COLOR}; }}
      .k {{ color:{ACCENT_COLOR}; font-weight:700; }}
      .card {{ border:1px solid #e6f0ec; border-radius:12px; padding:12px; margin:10px 0; }}
      .muted {{ color:#47606b; font-size:13px; }}
    </style>
    """
    def esc(x):
        try: return str(x).replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
        except Exception: return str(x)
    sec = payload or {}
    html = [css, "<h1>Dragonfly Health — Scheduling Run</h1>"]
    html.append(
        "<div class='card'><div class='muted'>Order</div>"
        f"<div><span class='k'>Order ID:</span> {esc(sec.get('order_id',''))}</div>"
        f"<div><span class='k'>Priority:</span> {esc(sec.get('priority',''))} • <span class='k'>Equip:</span> {esc(sec.get('equipment',''))}</div>"
        f"<div><span class='k'>Due by:</span> {esc(sec.get('due_by',''))}</div>"
        f"<div><span class='k'>Window:</span> {esc(sec.get('window_len',''))} hrs</div></div>"
    )
    html.append(
        "<div class='card'><div class='muted'>Recommended Slot</div>"
        f"<div><span class='k'>Start:</span> {esc(sec.get('slot_start',''))} → <span class='k'>End:</span> {esc(sec.get('slot_end',''))}</div>"
        f"<div><span class='k'>Score:</span> {esc(sec.get('score',''))} • <span class='k'>Risk:</span> {esc(sec.get('risk',''))}</div></div>"
    )
    if sec.get('eta_min') is not None:
        html.append(
            "<div class='card'><div class='muted'>ETA & Technician</div>"
            f"<div><span class='k'>ETA:</span> {esc(sec.get('eta_min'))} min</div>"
            f"<div><span class='k'>Technician:</span> {esc(sec.get('technician',''))}</div></div>"
        )
    html.append(f"<div class='muted'>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</div>")
    return "\n".join(html).encode("utf-8")


def run_demo_status():
    with st.status("Playing demo…", expanded=True) as status:
        import time as _t
        status.update(label="Loading synthetic data", state="running"); _t.sleep(0.5)
        status.update(label="Training lightweight risk model", state="running"); _t.sleep(0.6)
        status.update(label="Generating recommended slots", state="running"); _t.sleep(0.6)
        status.update(label="Scoring ETA and recommending technicians", state="running"); _t.sleep(0.6)
        status.update(label="Done — open Tabs 2 & 4 to view", state="complete")
    st.toast("Demo ready: open 'Order Intake + Recommender' and 'Equipment • ETA • Technicians'", icon="✅")

# ---------------------------
# App Layout (wire these blocks to the Part 1 functions/vars)
# ---------------------------

render_navbar()

with st.sidebar:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, use_container_width=True)
        show_user_chip()
    st.markdown("---")
    if st.button("▶ Play demo", use_container_width=True):
        run_demo_status()

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# Data + Model
left, right = st.columns([2,1])
with left:
    st.subheader("Data source")
    up = st.file_uploader("Upload CSV (optional)", type=["csv"])
    raw_df = load_input_df(up)
    st.success(f"Loaded {len(raw_df):,} orders")
with right:
    st.subheader("Model")
    model, metrics = build_model(raw_df)
    st.write({k:f"{v:.3f}" for k,v in metrics.items()})

# Tabs
_tab1, _tab2, _tab3, _tab4 = st.tabs([
    "Executive Overview", "Order Intake + Recommender", "Routing (VRP‑lite)", " Equipment • ETA • Technicians",
])

# Executive Overview
with _tab1:
    render_landing(raw_df, metrics)

# Intake + Recommender
with _tab2:
    st.subheader("Order intake → recommended slot")
    base = raw_df.sample(1).iloc[0]
    c1,c2,c3 = st.columns(3)
    with c1:
        order_id = st.text_input("Order ID", value=str(base["order_id"]))
        priority = st.selectbox("Priority", PRIORITY_ORDER, index=PRIORITY_ORDER.index(str(base["priority"])) )
        equipment_type = st.selectbox("Equipment", EQUIPMENT, index=EQUIPMENT.index(str(base["equipment_type"])) )
    with c2:
        distance_km = st.number_input("Distance (km)", value=float(base.get("distance_km", 18.0)), key="intake_dist")
        resource_load = st.slider("Resource load", 0.0, 1.0, float(base.get("resource_load", 0.6)), 0.05, key="intake_load")
        patient_pref = st.selectbox("Patient pref", PREF_ORDER, index=PREF_ORDER.index(str(base.get("patient_pref","none"))))
    with c3:
        from datetime import datetime as _dt
        def _dtpick(lbl, default):
            c1, c2 = st.columns(2)
            with c1: d = st.date_input(f"{lbl} — Date", value=pd.to_datetime(default).date())
            with c2: t = st.time_input(f"{lbl} — Time", value=pd.to_datetime(default).time())
            return _dt.combine(d, t)
        requested_at = _dtpick("Requested at", base["requested_at"])  
        due_by       = _dtpick("Due by",       base["due_by"])        
        window_len_hrs = st.select_slider("Window length (hrs)", options=[2,4,6], value=int(base.get("window_len_hrs",4)))

    now = datetime.now().replace(minute=0, second=0, microsecond=0)
    cand = [(now + timedelta(hours=2*i), now + timedelta(hours=2*i+window_len_hrs)) for i in range(1,24)]

    feat_row = pd.DataFrame({
        "priority":[priority], "reason_code":[base["reason_code"]], "channel":[base["channel"]],
        "patient_pref":[patient_pref], "window_len_hrs":[window_len_hrs], "distance_km":[distance_km],
        "resource_load":[resource_load],
        "hours_to_due":[(pd.to_datetime(due_by)-pd.to_datetime(now)).total_seconds()/3600.0],
        "hours_from_request":[(pd.to_datetime(now)-pd.to_datetime(requested_at)).total_seconds()/3600.0],
    })
    risk = float(model.predict_proba(feat_row)[:,1])

    order_like = pd.Series({"due_by": due_by, "distance_km": distance_km, "resource_load": resource_load, "patient_pref": patient_pref})
    scored = score_slots(order_like, cand)
    top = scored[:5]

    st.session_state["report_payload"].update({
        "order_id": order_id,
        "priority": priority,
        "equipment": equipment_type,
        "due_by": due_by.strftime('%Y-%m-%d %H:%M'),
        "window_len": window_len_hrs,
        "slot_start": top[0]["slot_start"].strftime('%Y-%m-%d %H:%M'),
        "slot_end": top[0]["slot_end"].strftime('%Y-%m-%d %H:%M'),
        "score": f"{top[0]['score']:.2f}",
        "risk": f"{risk:.2%}",
    })

    k1,k2,k3 = st.columns(3)
    with k1: st.metric("Top slot score", f"{top[0]['score']:.2f}")
    with k2: st.metric("Predicted adjustment risk", f"{risk:.1%}")
    with k3: st.metric("Window length", f"{window_len_hrs}h")

    st.markdown("### Recommendations")
    for i, rec in enumerate(top, 1):
        good = "✅" if i==1 else ""
        st.markdown(
            f"""
            <div class='card'>
               <div style='display:flex;justify-content:space-between;align-items:center;'>
                 <div><span class='pill'>Rank {i}</span> <b>{rec['slot_start'].strftime('%a %b %d, %I:%M %p')}</b> → {rec['slot_end'].strftime('%I:%M %p')}</div>
                 <div style='font-weight:700;color:{ACCENT_COLOR}'>Score {rec['score']:.2f} {good}</div>
               </div>
               <div class='subtle' style='margin-top:6px'>Why: {rec['explain']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
def _minute_horizon(day_hours: int = 24) -> int:
    return day_hours * 60

def _derive_windows_from_row(r: pd.Series) -> tuple[int,int]:
    now = pd.Timestamp.now()
    for a, b in [("scheduled_start","scheduled_end"), ("RequestedStartDate","RequestedEndDate")]:
        if a in r and b in r and pd.notna(r[a]) and pd.notna(r[b]):
            ws = int(max(0, (pd.to_datetime(r[a]) - now).total_seconds() // 60))
            we = int(max(ws+30, (pd.to_datetime(r[b]) - now).total_seconds() // 60))
            return (ws, we)
    pri = str(r.get("priority", r.get("PriorityName","Routine"))).strip()
    if pri.upper() == "STAT": return (0, 240)
    if pri.lower() == "urgent": return (0, 1440)
    pref = str(r.get("patient_pref","none"))
    if pref == "am": return (8*60, 12*60)
    if pref == "pm": return (12*60, 17*60)
    if pref == "eve": return (17*60, 20*60)
    return (8*60, 20*60)

def _make_stops_from_df(rows: pd.DataFrame) -> list[dict]:
    out = []
    for _, r in rows.iterrows():
        equip = str(r.get("equipment_type","bp_monitor"))
        meta  = EQUIPMENT_CATALOG.get(equip, {"prep_min": 10, "skill": "general"})
        tws, twe = _derive_windows_from_row(r)
        out.append({
            "order_id": str(r.get("order_id","")),
            "equipment_type": equip,
            "lat": float(r.get("patient_lat", 0.0)),
            "lon": float(r.get("patient_lon", 0.0)),
            "skill_req": meta.get("skill","general"),
            "service_min": int(meta.get("prep_min", 10)),
            "tw_start": int(max(0, tws)),
            "tw_end": int(min(_minute_horizon(), max(tws+30, twe))),
        })
    return out
# Routing (VRP‑lite)
# Routing (multi-route VRP)
with _tab3:
    st.subheader("Route planning — time windows • skills • capacity")

    h_opts = {h["hospital_id"]: h for h in HOSPITALS}
    c1, c2, c3 = st.columns(3)
    with c1:
        h_sel = st.selectbox(
            "Facility",
            options=list(h_opts.keys()),
            format_func=lambda x: f"{x} — {h_opts[x]['name']}",
            key="facility_routing_constrained",
        )
    with c2:
        n_orders = st.slider("Number of orders", 4, 50, 12, 1)
    with c3:
        n_routes = st.slider("Number of routes (technicians)", 1, 8, 3, 1)

    c4, c5, c6, c7 = st.columns(4)
    with c4:
        cap = st.number_input("Capacity per route (stops)", min_value=1, max_value=50, value=6)
    with c5:
        enforce_sk = st.checkbox("Enforce technician skills", value=True)
    with c6:
        use_tw = st.checkbox("Use time windows (if available/derived)", value=True)
    with c7:
        speed = st.number_input("Avg speed (km/h)", min_value=10.0, max_value=90.0, value=38.0, step=1.0)

    # Build depot & choose nearest orders
    depot_lat, depot_lon = h_opts[h_sel]["lat"], h_opts[h_sel]["lon"]
    fac_df = raw_df[raw_df["hospital_id"] == h_sel].copy()
    if fac_df.empty: fac_df = raw_df.copy()

    # choose a pool near depot
    def _nearest(df, clat, clon, k):
        d = df.copy()
        d["dist"] = d.apply(lambda r: haversine_km(clat, clon, float(r["patient_lat"]), float(r["patient_lon"])), axis=1)
        return d.nsmallest(k, "dist")

    pool = _nearest(fac_df, depot_lat, depot_lon, min(len(fac_df), max(5, n_orders))).head(n_orders)
      # Ensure required columns exist; create missing ones with sensible defaults
    needed = ["order_id","equipment_type","patient_lat","patient_lon","priority","patient_pref",
    "scheduled_start","scheduled_end","RequestedStartDate","RequestedEndDate"]

    pool = pool.copy()
    for c in needed:
        if c not in pool.columns:
        # Default types: dates as NaT, others as NaN/empty
           if any(k in c.lower() for k in ["date","time","scheduled","start","end"]):
            pool[c] = pd.NaT
           elif c in ("order_id","equipment_type","priority","patient_pref"):
            pool[c] = ""
           else:
            pool[c] = np.nan

    # Now select in the intended order
    stops_df = pool[needed].copy()

    # shape stops
    shaped = _make_stops_from_df(stops_df.rename(columns={"patient_lat":"patient_lat", "patient_lon":"patient_lon"}))
    # rename keys expected by solver
    stops_payload = [
        {"order_id": s["order_id"], "equipment_type": s["equipment_type"],
         "patient_lat": s["lat"], "patient_lon": s["lon"],
         "skill_req": s["skill_req"], "service_min": s["service_min"],
         "tw_start": s["tw_start"], "tw_end": s["tw_end"]}
        for s in shaped
    ]

    # solve

    plan = plan_multi_routes_with_constraints(
    (depot_lat, depot_lon),
    stops_payload,
    num_routes=n_routes,
    capacity_per_vehicle=cap,
    enforce_skills=enforce_sk,
    use_time_windows=use_tw,
    speed_kmh=speed,
)
    # Show routes
    st.markdown("### Suggested routes")
    cols = st.columns(min(3, max(1, len(plan["routes"]))))
    for ridx, route in enumerate(plan["routes"]):
        with cols[ridx % len(cols)]:
            st.markdown(f"**Route {ridx+1}** — Distance: **{plan['dist_km'][ridx]:.1f} km**")
            for step, pidx in enumerate(route):
                st.markdown(f"{step+1}. {plan['labels'][pidx]}")

    kpi("Total distance (km)", f"{plan['total_km']:.1f}", "All routes combined")

    # Map
    map_rows = []
for r_i, route in enumerate(plan["routes"], 1):
    for pidx in route:
        lat, lon = plan["points"][pidx]
        map_rows.append({
            "latitude": lat,
            "longitude": lon,
            "Route": f"Route {r_i}",
            "label": plan["labels"][pidx]
        })

if map_rows:
    # ---------- Multi-color route map (pydeck) ----------
    st.markdown("#### Route map (colored paths)")

    routes = plan["routes"]
    points = plan["points"]   # [(lat, lon), ...] 0 = depot
    labels = plan["labels"]   # ["Facility", "ORD-...", ...]

    if routes and points:
        # Color palette (RGB)
        palette = [
            [230, 57, 70],    # red
            [29, 53, 87],     # blue
            [42, 157, 143],   # green/teal
            [244, 162, 97],   # orange
            [38, 70, 83],     # dark teal
            [168, 218, 220],  # light blue
            [233, 196, 106],  # yellow
            [90, 90, 200],    # purple-ish
        ]

        # Build line features per route (depot -> stops -> depot)
        line_features = []
        stop_features = []
        for r_i, route in enumerate(routes):
            color = palette[r_i % len(palette)]

            # Ensure depot start/end
            path_idx = route
            if route[0] != 0:
                path_idx = [0] + route
            if route[-1] != 0:
                path_idx = path_idx + [0]

            path = [[float(points[i][1]), float(points[i][0])] for i in path_idx]  # [lon, lat]

            line_features.append({
                "path": path,
                "color": color,
                "name": f"Route {r_i+1}",
            })

            # Stops (exclude depot index 0)
            for pidx in route:
                if pidx == 0:
                    continue
                lat, lon = float(points[pidx][0]), float(points[pidx][1])
                stop_features.append({
                    "position": [lon, lat],
                    "color": color,
                    "route": f"Route {r_i+1}",
                    "label": labels[pidx] if pidx < len(labels) else f"Stop {pidx}",
                })

        # Depot marker
        depot_lat, depot_lon = float(points[0][0]), float(points[0][1])
        depot_df = pd.DataFrame([{"position": [depot_lon, depot_lat], "label": "Facility"}])

        # DataFrames for pydeck
        lines_df = pd.DataFrame(line_features)
        stops_df = pd.DataFrame(stop_features)

        # Layers
        line_layer = pdk.Layer(
            "PathLayer",
            data=lines_df,
            get_path="path",
            get_color="color",
            width_scale=1,
            width_min_pixels=3,
            opacity=0.7,
            pickable=True,
        )

        stops_layer = pdk.Layer(
            "ScatterplotLayer",
            data=stops_df,
            get_position="position",
            get_color="color",
            get_radius=50,           # meters
            radius_min_pixels=4,
            pickable=True,
        )

        depot_layer = pdk.Layer(
            "ScatterplotLayer",
            data=depot_df,
            get_position="position",
            get_color=[0, 0, 0],
            get_radius=80,
            radius_min_pixels=5,
            pickable=True,
        )

        # View centered on depot
        view_state = pdk.ViewState(
            latitude=depot_lat,
            longitude=depot_lon,
            zoom=11,
            pitch=0,
            bearing=0,
        )

        r = pdk.Deck(
            layers=[line_layer, stops_layer, depot_layer],
            initial_view_state=view_state,
            tooltip={"text": "{label} {name}{route}"},
            map_style="mapbox://styles/mapbox/light-v9",  # default basemap
        )

        st.pydeck_chart(r, use_container_width=True)

    else:
        st.info("No routes to display with current selection.")
# with _tab3:
#     st.subheader("Route planning from facility → patients (demo)")
#     h_opts = {h["hospital_id"]:h for h in HOSPITALS}
#     h_sel = st.selectbox("Facility", options=list(h_opts.keys()), format_func=lambda x: f"{x} — {h_opts[x]['name']}", key="facility_routing")
#     h_lat, h_lon = h_opts[h_sel]["lat"], h_opts[h_sel]["lon"]
#     dfh = raw_df[raw_df["hospital_id"]==h_sel].copy()
#     seed = dfh.sample(1).iloc[0] if not dfh.empty else raw_df.sample(1).iloc[0]
#     st.markdown(f"**Seed order:** {seed['order_id']} · equipment: `{seed['equipment_type']}` · priority: `{seed['priority']}`")

#     def nearest_neighbors(df, center_lat, center_lon, k=4):
#         df = df.copy(); df["dist"] = df.apply(lambda r: haversine_km(center_lat, center_lon, r["patient_lat"], r["patient_lon"]), axis=1)
#         return df.nsmallest(k, "dist")

#     neighbors = nearest_neighbors(dfh, float(seed["patient_lat"]), float(seed["patient_lon"]), k=4)
#     stops = pd.concat([seed.to_frame().T, neighbors]).drop_duplicates("order_id").head(5)

#     pts = [(h_lat, h_lon)] + list(zip(stops["patient_lat"].astype(float), stops["patient_lon"].astype(float)))
#     distM = build_distance_matrix(pts)
#     route_idx = solve_route(distM)

#     route_labels = ["Facility"] + [f"{row.order_id} ({row.equipment_type})" for _,row in stops.iterrows()]
#     pretty_route = [route_labels[i] for i in route_idx]

#     st.markdown("### Suggested visit order")
#     for step, label in enumerate(pretty_route, 1):
#         st.markdown(f"{step}. **{label}**")

#     total_km = sum(haversine_km(pts[route_idx[i]][0], pts[route_idx[i]][1], pts[route_idx[i+1]][0], pts[route_idx[i+1]][1]) for i in range(len(route_idx)-1))
#     st.metric("Total distance (km)", f"{total_km:.1f}")

#     st.markdown("#### Map (lat/lon preview)")
#     map_df = pd.DataFrame({
#         "lat": [p[0] for p in pts],
#         "lon": [p[1] for p in pts],
#         "label": ["Facility"] + list(stops["order_id"]),
#     })
#     st.map(map_df.rename(columns={"lon":"longitude","lat":"latitude"}))

#     st.info("For production: upgrade to OR‑Tools VRPTW with time windows, skills, capacities.")

# Equipment • ETA • Technicians
with _tab4:
    st.subheader("Equipment selection, ETA prediction, technician assignment")
    base_row = raw_df.sample(1).iloc[0]

    c1,c2,c3 = st.columns(3)
    with c1:
        equip = st.selectbox("Equipment for delivery", options=EQUIPMENT, index=EQUIPMENT.index(str(base_row.get("equipment_type","cpap"))))
        equip_meta = EQUIPMENT_CATALOG.get(equip, {"prep_min":10, "stock":5, "skill":"general"})
        st.metric("Stock available", f"{equip_meta['stock']}")
        st.metric("Prep time (min)", f"{equip_meta['prep_min']}")
    with c2:
        traffic = st.slider("Traffic multiplier", 0.6, 1.8, 1.1, 0.1)
        jobs_q = st.slider("Active jobs in queue", 0, 6, 2)
        dist_km_eta = st.number_input("Distance to patient (km)", value=float(base_row.get("distance_km", 18.0)), key="eta_dist_km")
    with c3:
        plat = st.number_input("Patient lat", value=float(base_row.get("patient_lat", 42.36)), key="eta_patient_lat")
        plon = st.number_input("Patient lon", value=float(base_row.get("patient_lon", -71.05)), key="eta_patient_lon")
        req_skill = equip_meta.get("skill","general")
        st.metric("Required skill", req_skill)

    eta_min = estimate_eta_minutes(dist_km_eta, equip_meta["prep_min"], traffic, jobs_q)
    st.markdown("### ETA prediction")
    st.metric("Estimated time to arrive (min)", f"{eta_min}")

    st.markdown("### Technician recommendation")
    top_techs = best_technicians(plat, plon, req_skill, topn=3)
    for rank, item in enumerate(top_techs, 1):
        t = item["tech"]
        st.markdown(
            f"""
            <div class='card'>
               <div style='display:flex;justify-content:space-between;align-items:center;'>
                 <div>
                    <span class='pill'>Rank {rank}</span>
                    <b>{t['name']}</b> — skill: <b>{t['skill']}</b>
                    <div class='subtle'>Distance ≈ {item['dist_km']:.1f} km • Active jobs: {t['active_jobs']}</div>
                 </div>
                 <div style='font-weight:700;color:{ACCENT_COLOR}'>Score {item['score']:.3f}</div>
               </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Communications (templates)")
    tech_choice = st.selectbox("Assign technician", options=[x["tech"]["id"] + " — " + x["tech"]["name"] for x in top_techs], key="assign_tech")
    sel_name = tech_choice.split(" — ")[-1]
    appt_time = datetime.now() + timedelta(minutes=eta_min)
    msg_patient = (
        f"Hello! Your Dragonfly Health delivery for {equip.replace('_',' ')} is scheduled today. "
        f"Your technician {sel_name} is on the way. ETA: {appt_time.strftime('%I:%M %p')}. "
        f"Reply 1 to confirm or 2 to reschedule."
    )
    msg_tech = (
        f"Assignment: Deliver {equip.replace('_',' ')}. Patient coords: ({plat:.4f}, {plon:.4f}). "
        f"ETA {eta_min} min. Prep: {equip_meta['prep_min']} min."
    )
    msg_facility = (
        f"Dispatch notice: {sel_name} assigned to order (demo) for {equip.replace('_',' ')}. ETA ~{eta_min} min."
    )
    st.text_area("Patient SMS", value=msg_patient, height=90)
    st.text_area("Technician push note", value=msg_tech, height=90)
    st.text_area("Facility/Case note", value=msg_facility, height=90)

    # Export
    st.markdown("### Export")
    st.session_state["report_payload"].update({"eta_min": eta_min, "technician": sel_name})
    report_bytes = build_report_html(st.session_state["report_payload"])
    st.download_button(
        label="📄 Export run (HTML)",
        data=report_bytes,
        file_name=f"dragonfly_scheduling_run_{datetime.now().strftime('%Y%m%d_%H%M')}.html",
        mime="text/html",
        use_container_width=True,
    )
    st.caption("Open the HTML and use your browser's Print → Save as PDF.")

# Footer
st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
st.caption("© 2025 Dragonfly Health — Demo. For illustrative use only; not for clinical routing decisions.")
