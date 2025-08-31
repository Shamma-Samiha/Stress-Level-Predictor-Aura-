import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ===================== Load model + feature order =====================
@st.cache_resource
def load_artifacts():
    model = joblib.load("model_rf.joblib")
    with open("feature_order.json") as f:
        features = json.load(f)["features"]
    return model, features

model, FEATURE_ORDER = load_artifacts()

LABELS = {0: "Low", 1: "Medium", 2: "High"}

# ===================== Page config =====================
st.set_page_config(page_title="Stress Level Predictor", page_icon="🧠", layout="wide")
st.title("🧠 Stress Level Predictor")
st.caption("RandomForest model trained on psychological, social, and lifestyle features")

# ===================== Defaults & Session State =====================
default_values = {
    "anxiety_level": 10, "self_esteem": 15, "mental_health_history": 0, "depression": 8,
    "headache": 2, "blood_pressure": 2, "sleep_quality": 3, "breathing_problem": 1,
    "noise_level": 2, "living_conditions": 3, "safety": 3, "basic_needs": 4,
    "academic_performance": 3, "study_load": 3, "teacher_student_relationship": 3,
    "future_career_concerns": 2, "social_support": 2, "peer_pressure": 2,
    "extracurricular_activities": 3, "bullying": 1,
    "_go": False
}
for k, v in default_values.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ===================== Sidebar Inputs =====================
st.sidebar.header("💾 Input features")

anxiety_level = st.sidebar.slider("Anxiety level", 0, 21, st.session_state["anxiety_level"], key="anxiety_level")
self_esteem   = st.sidebar.slider("Self esteem", 0, 30, st.session_state["self_esteem"], key="self_esteem")
mental_health_history = st.sidebar.select_slider("Mental health history", options=[0, 1], value=st.session_state["mental_health_history"], key="mental_health_history")
depression    = st.sidebar.slider("Depression", 0, 27, st.session_state["depression"], key="depression")

headache  = st.sidebar.slider("Headache", 0, 5, st.session_state["headache"], key="headache")
blood_pressure = st.sidebar.slider("Blood pressure (category)", 1, 3, st.session_state["blood_pressure"], key="blood_pressure")
sleep_quality = st.sidebar.slider("Sleep quality", 0, 5, st.session_state["sleep_quality"], key="sleep_quality")
breathing_problem = st.sidebar.slider("Breathing problem", 0, 5, st.session_state["breathing_problem"], key="breathing_problem")
noise_level = st.sidebar.slider("Noise level", 0, 5, st.session_state["noise_level"], key="noise_level")
living_conditions = st.sidebar.slider("Living conditions", 0, 5, st.session_state["living_conditions"], key="living_conditions")
safety = st.sidebar.slider("Safety", 0, 5, st.session_state["safety"], key="safety")
basic_needs = st.sidebar.slider("Basic needs", 0, 5, st.session_state["basic_needs"], key="basic_needs")
academic_performance = st.sidebar.slider("Academic performance", 0, 5, st.session_state["academic_performance"], key="academic_performance")
study_load = st.sidebar.slider("Study load", 0, 5, st.session_state["study_load"], key="study_load")
tsr = st.sidebar.slider("Teacher-student relationship", 0, 5, st.session_state["teacher_student_relationship"], key="teacher_student_relationship")
career = st.sidebar.slider("Future career concerns", 0, 5, st.session_state["future_career_concerns"], key="future_career_concerns")
social_support = st.sidebar.slider("Social support", 0, 3, st.session_state["social_support"], key="social_support")
peer_pressure = st.sidebar.slider("Peer pressure", 0, 5, st.session_state["peer_pressure"], key="peer_pressure")
extracurricular = st.sidebar.slider("Extracurricular activities", 0, 5, st.session_state["extracurricular_activities"], key="extracurricular_activities")
bullying = st.sidebar.slider("Bullying", 0, 5, st.session_state["bullying"], key="bullying")

col_btn1, col_btn2 = st.sidebar.columns(2)
with col_btn1:
    if st.button("Predict", use_container_width=True):
        st.session_state["_go"] = True
with col_btn2:
    if st.button("Reset", use_container_width=True):
        for k, v in default_values.items():
            st.session_state[k] = v
        st.rerun()

# Build single-row DataFrame in correct feature order
row = pd.DataFrame([{
    "anxiety_level": anxiety_level,
    "self_esteem": self_esteem,
    "mental_health_history": mental_health_history,
    "depression": depression,
    "headache": headache,
    "blood_pressure": blood_pressure,
    "sleep_quality": sleep_quality,
    "breathing_problem": breathing_problem,
    "noise_level": noise_level,
    "living_conditions": living_conditions,
    "safety": safety,
    "basic_needs": basic_needs,
    "academic_performance": academic_performance,
    "study_load": study_load,
    "teacher_student_relationship": tsr,
    "future_career_concerns": career,
    "social_support": social_support,
    "peer_pressure": peer_pressure,
    "extracurricular_activities": extracurricular,
    "bullying": bullying,
}])[FEATURE_ORDER]

# ===================== Prediction =====================
st.subheader("🔎 Prediction")
if st.session_state.get("_go", False):
    proba = model.predict_proba(row)[0]
    pred  = int(np.argmax(proba))
    st.metric("Predicted stress level", LABELS[pred])

    prob_df = pd.DataFrame({"stress_level": [LABELS[i] for i in range(len(proba))],
                            "probability": proba}).set_index("stress_level")
    st.bar_chart(prob_df)

    st.subheader("Feature importance (global RF)")
    imp = pd.Series(model.feature_importances_, index=FEATURE_ORDER).sort_values(ascending=False)
    st.dataframe(imp.head(12).rename("importance"))
else:
    st.info("Set your inputs on the left and click **Predict**.")

# ===================== Scenario Testing / Optimiser =====================
st.divider()
st.subheader("🧪 Scenario testing — optimise my stress")

target_label_for_opt = st.selectbox(
    "Target stress level to optimise for",
    options=list(LABELS.values()),
    index=list(LABELS.values()).index("Low"),
)
target_class_for_opt = {v: k for k, v in LABELS.items()}[target_label_for_opt]

tweak_budget = st.slider("Maximum number of small tweaks", 1, 5, 3,
                         help="We try up to this many +/-1 adjustments.")

# Bounds by dataset
BOUNDS = {
    "anxiety_level": (0, 21), "self_esteem": (0, 30), "mental_health_history": (0, 1), "depression": (0, 27),
    "headache": (0, 5), "blood_pressure": (1, 3), "sleep_quality": (0, 5), "breathing_problem": (0, 5),
    "noise_level": (0, 5), "living_conditions": (0, 5), "safety": (0, 5), "basic_needs": (0, 5),
    "academic_performance": (0, 5), "study_load": (0, 5), "teacher_student_relationship": (0, 5),
    "future_career_concerns": (0, 5), "social_support": (0, 3), "peer_pressure": (0, 5),
    "extracurricular_activities": (0, 5), "bullying": (0, 5),
}
# +1 increases are helpful, -1 decreases are helpful, 0 = skip/immutable
DIRECTION = {
    "anxiety_level": -1, "self_esteem": +1, "mental_health_history": 0, "depression": -1,
    "headache": -1, "blood_pressure": -1, "sleep_quality": +1, "breathing_problem": -1,
    "noise_level": -1, "living_conditions": +1, "safety": +1, "basic_needs": +1,
    "academic_performance": +1, "study_load": -1, "teacher_student_relationship": +1,
    "future_career_concerns": -1, "social_support": +1, "peer_pressure": -1,
    "extracurricular_activities": +1, "bullying": -1,
}

def _clip(v, lo, hi): return max(lo, min(hi, v))
def _score(p, t): return float(p[t])
def _proba(df_row): return model.predict_proba(df_row[FEATURE_ORDER])[0]

def try_local_moves(base_row_df, current_probs, target_idx):
    best_delta, best_feat, best_val, best_probs = 0.0, None, None, current_probs
    base_vals = base_row_df.iloc[0]

    for feat in FEATURE_ORDER:
        if DIRECTION.get(feat, 0) == 0:
            continue
        step = DIRECTION[feat]
        for s in [step, -step]:  # prefer helpful direction first
            lo, hi = BOUNDS[feat]
            trial_val = _clip(int(base_vals[feat] + s), lo, hi)
            if trial_val == int(base_vals[feat]):  # no change
                continue
            trial_row = base_row_df.copy()
            trial_row.at[0, feat] = trial_val
            trial_probs = _proba(trial_row)
            delta = _score(trial_probs, target_idx) - _score(current_probs, target_idx)
            if delta > best_delta:
                best_delta, best_feat, best_val, best_probs = delta, feat, trial_val, trial_probs
    return best_delta, best_feat, best_val, best_probs

def greedy_optimize(base_row_df, target_idx, max_moves=3):
    history, cur_row = [], base_row_df.copy()
    cur_probs = _proba(cur_row)
    for _ in range(max_moves):
        delta, feat, new_val, new_probs = try_local_moves(cur_row, cur_probs, target_idx)
        if not feat or delta <= 0:
            break
        old_val = int(cur_row.at[0, feat])
        cur_row.at[0, feat] = int(new_val)
        history.append({"feature": feat, "old": old_val, "new": int(new_val), "delta_prob": float(delta)})
        cur_probs = new_probs
    return cur_row, cur_probs, history

# Run optimiser
if st.button("✨ Optimise my stress", use_container_width=True):
    start_probs = _proba(row)
    new_row, end_probs, changes = greedy_optimize(row.copy(), target_class_for_opt, tweak_budget)

    colA, colB = st.columns(2)
    with colA:
        st.markdown("**Before (current inputs)**")
        st.metric(f"P({target_label_for_opt})", f"{start_probs[target_class_for_opt]:.3f}")
        st.write(pd.DataFrame({
            "stress_level": [LABELS[i] for i in range(len(start_probs))],
            "probability": start_probs
        }).set_index("stress_level"))
    with colB:
        st.markdown("**After (suggested tweaks)**")
        st.metric(
            f"P({target_label_for_opt})",
            f"{end_probs[target_class_for_opt]:.3f}",
            delta=f"{(end_probs[target_class_for_opt] - start_probs[target_class_for_opt]):+.3f}"
        )
        st.write(pd.DataFrame({
            "stress_level": [LABELS[i] for i in range(len(end_probs))],
            "probability": end_probs
        }).set_index("stress_level"))

    if changes:
        st.markdown("**Proposed small changes**")
        st.dataframe(pd.DataFrame(changes))

        # Reset sliders to optimised values
        for feat in FEATURE_ORDER:
            st.session_state[feat] = int(new_row.iloc[0][feat])
        st.success("✅ Sliders reset to optimised values. Adjust further if you like.")
        # Removed st.rerun() to prevent UI refresh before showing results
    else:
        st.info("No beneficial tweaks found within the given budget and bounds.")
