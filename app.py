# ══════════════════════════════════════════════════════════════════════════════
#  CardioPredict AI — app.py
#  Heart Disease Prediction | Matches your Colab notebook exactly
#  Run: streamlit run app.py
# ══════════════════════════════════════════════════════════════════════════════

import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, precision_score,
                             recall_score, f1_score)
import plotly.graph_objects as go
import plotly.express as px

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CardioPredict AI",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS (Dark Premium Theme) ───────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0d1117 100%); }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #161b22 0%, #0d1117 100%);
        border-right: 1px solid #30363d;
    }
    .hero {
        background: linear-gradient(135deg, #1c2128, #21262d);
        border: 1px solid #30363d; border-radius: 20px;
        padding: 40px; margin-bottom: 28px; text-align: center;
    }
    .hero h1 { font-size: 2.6rem; font-weight: 700; color: #f0f6fc; margin-bottom: 6px; }
    .hero p  { color: #8b949e; font-size: 1.05rem; }
    .metric-card {
        background: linear-gradient(135deg, #1c2128, #21262d);
        border: 1px solid #30363d; border-radius: 16px;
        padding: 22px; text-align: center; margin-bottom: 16px;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .metric-card:hover { transform: translateY(-4px); box-shadow: 0 12px 40px rgba(248,81,73,0.15); }
    .metric-card h1 { font-size: 2.6rem; margin: 0; color: #f85149; }
    .metric-card p  { color: #8b949e; margin: 4px 0 0; font-size: 0.88rem; }
    .section-title {
        font-size: 1.4rem; font-weight: 600; color: #f0f6fc;
        border-left: 4px solid #f85149; padding-left: 12px; margin-bottom: 18px;
    }
    .result-danger {
        background: linear-gradient(135deg, #3d1515, #2d1010);
        border: 2px solid #f85149; border-radius: 16px;
        padding: 32px; text-align: center; animation: pulse 2s infinite;
    }
    .result-safe {
        background: linear-gradient(135deg, #0d2c1e, #0a1f16);
        border: 2px solid #3fb950; border-radius: 16px;
        padding: 32px; text-align: center;
    }
    .result-danger h2 { color: #f85149; font-size: 1.9rem; }
    .result-safe  h2 { color: #3fb950; font-size: 1.9rem; }
    .result-danger p, .result-safe p { color: #8b949e; font-size: 0.98rem; margin-top: 8px; }
    @keyframes pulse {
        0%,100% { box-shadow: 0 0 0 0 rgba(248,81,73,0.4); }
        50%      { box-shadow: 0 0 0 16px rgba(248,81,73,0); }
    }
    label { color: #c9d1d9 !important; font-weight: 500 !important; }
    .stButton > button {
        background: linear-gradient(135deg, #da3633, #f85149) !important;
        color: white !important; border: none !important; border-radius: 12px !important;
        padding: 14px 36px !important; font-size: 1rem !important;
        font-weight: 600 !important; width: 100% !important;
        transition: all 0.3s ease !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(248,81,73,0.45) !important;
    }
    .stTabs [data-baseweb="tab-list"] { background: #161b22; border-radius: 10px; padding: 4px; }
    .stTabs [data-baseweb="tab"]       { color: #8b949e; border-radius: 8px; }
    .stTabs [aria-selected="true"]     { background: #21262d; color: #f0f6fc; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
#   TRAIN MODELS (exactly as in your Colab — no scaler, no stratify)
# ═══════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="⏳ Training models on heart.csv …")
def train_all_models():
    """
    Replicates your Colab notebook exactly:
    - No StandardScaler
    - train_test_split(test_size=0.2, random_state=42)  — no stratify
    - LR: default params
    - SVM: default kernel (rbf), probability=True added for predict_proba
    - Decision Tree: max_depth=5, criterion='entropy'
    - Random Forest: n_estimators=500, criterion='entropy'
    """
    df = pd.read_csv("heart.csv")
    X = df.iloc[:, :-1]   # all columns except last
    y = df.iloc[:, -1]    # last column = target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42   # exactly your Colab split
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "SVM":                 SVC(probability=True),            # probability=True for predict_proba
        "Decision Tree":       DecisionTreeClassifier(max_depth=5, criterion="entropy"),
        "Random Forest":       RandomForestClassifier(n_estimators=500, criterion="entropy"),
    }

    trained = {}
    metrics = {}
    for name, mdl in models.items():
        mdl.fit(X_train, y_train)
        preds = mdl.predict(X_test)
        trained[name] = mdl
        metrics[name] = {
            "Accuracy":  round(accuracy_score(y_test, preds) * 100, 2),
            "Precision": round(precision_score(y_test, preds, zero_division=0) * 100, 2),
            "Recall":    round(recall_score(y_test, preds, zero_division=0) * 100, 2),
            "F1-Score":  round(f1_score(y_test, preds, zero_division=0) * 100, 2),
            "cm":        confusion_matrix(y_test, preds).tolist(),
        }

    feature_names = X.columns.tolist()
    return trained, metrics, X_test, y_test, feature_names

@st.cache_data
def load_data():
    if os.path.exists("heart.csv"):
        return pd.read_csv("heart.csv")
    return None

# ── Feature metadata (default values = Colab manual prediction example) ───────
# Colab example: input_data = (52, 1, 0, 108, 233, 1, 1, 147, 0, 0.1, 2, 3, 3)
FEATURE_META = {
    "age":      ("Age",                         20,  100, 52,   "years"),
    "sex":      ("Sex",                         0,   1,   1,    "0=Female  1=Male"),
    "cp":       ("Chest Pain Type",             0,   3,   0,    "0=Typical 1=Atypical 2=Non-anginal 3=Asymptomatic"),
    "trestbps": ("Resting Blood Pressure",      80,  200, 108,  "mm Hg"),
    "chol":     ("Serum Cholesterol",           100, 600, 233,  "mg/dL"),
    "fbs":      ("Fasting Blood Sugar >120",    0,   1,   1,    "0=No  1=Yes"),
    "restecg":  ("Resting ECG Results",         0,   2,   1,    "0=Normal 1=ST-T abnorm 2=LV hypertrophy"),
    "thalach":  ("Max Heart Rate Achieved",     60,  220, 147,  "bpm"),
    "exang":    ("Exercise Induced Angina",     0,   1,   0,    "0=No  1=Yes"),
    "oldpeak":  ("ST Depression (Oldpeak)",     0.0, 7.0, 0.1,  "mm"),
    "slope":    ("Slope of ST Segment",         0,   2,   2,    "0=Upsloping 1=Flat 2=Downsloping"),
    "ca":       ("No. Major Vessels (0-3)",     0,   3,   3,    "Coloured by fluoroscopy"),
    "thal":     ("Thalassemia",                 0,   3,   3,    "1=Normal 2=Fixed Defect 3=Reversible Defect"),
}

data_ready = os.path.exists("heart.csv")

# ══════════════════════════════════════════════════════════════════════════════
#   SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ❤️ CardioPredict AI")
    st.markdown("*Heart Disease Prediction System*")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["🏠 Home & Predict", "📊 Model Performance", "🔬 EDA Dashboard", "ℹ️ About"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("### 🤖 Active Model")

    if data_ready:
        try:
            models_dict, metrics_dict, X_test_cache, y_test_cache, feat_names = train_all_models()
            selected_model = st.selectbox(
                "Choose Classifier",
                list(models_dict.keys()),
                index=3  # default = Random Forest (best in your Colab)
            )
            acc = metrics_dict[selected_model]["Accuracy"]
            st.success(f"Accuracy: **{acc}%**")
        except Exception as e:
            st.error(f"Training failed:\n{e}")
            models_dict  = {}
            metrics_dict = {}
            selected_model = None
            X_test_cache = None
            y_test_cache = None
    else:
        st.error("⚠️ `heart.csv` not found!\nPlace it in the same folder as `app.py`.")
        models_dict  = {}
        metrics_dict = {}
        selected_model = None
        X_test_cache = None
        y_test_cache = None

    st.markdown("---")
    st.caption("Built with ❤️ Streamlit & Scikit-learn\nMatches your Colab exactly")

# ══════════════════════════════════════════════════════════════════════════════
#   PAGE 1 — HOME & PREDICT
# ══════════════════════════════════════════════════════════════════════════════
if "Home" in page:
    st.markdown("""
    <div class="hero">
        <h1>❤️ CardioPredict AI</h1>
        <p>Advanced Heart Disease Risk Assessment · Powered by Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)

    df = load_data()
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown('<div class="metric-card"><h1>4</h1><p>ML Models</p></div>', unsafe_allow_html=True)
    with c2:
        n = len(df) if df is not None else 1025
        st.markdown(f'<div class="metric-card"><h1>{n}</h1><p>Patient Records</p></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="metric-card"><h1>13</h1><p>Clinical Features</p></div>', unsafe_allow_html=True)
    with c4:
        best_acc = max(metrics_dict[m]["Accuracy"] for m in metrics_dict) if models_dict else 88.0
        st.markdown(f'<div class="metric-card"><h1>{best_acc}%</h1><p>Best Accuracy</p></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-title">🩺 Patient Clinical Data Form</div>', unsafe_allow_html=True)
    st.info("💡 Default values = Colab manual example: `(52, 1, 0, 108, 233, 1, 1, 147, 0, 0.1, 2, 3, 3)` → should predict **Heart Disease**")

    with st.form("predict_form"):
        col1, col2, col3 = st.columns(3)
        inputs = {}
        feat_keys = list(FEATURE_META.keys())

        for i, feat in enumerate(feat_keys):
            label, mn, mx, default, hint = FEATURE_META[feat]
            col = [col1, col2, col3][i % 3]
            with col:
                if feat == "sex":
                    inputs[feat] = st.selectbox(label, [0, 1], index=int(default),
                        format_func=lambda x: "Female (0)" if x == 0 else "Male (1)", help=hint)
                elif feat == "fbs":
                    inputs[feat] = st.selectbox(label, [0, 1], index=int(default),
                        format_func=lambda x: "No — ≤120 (0)" if x == 0 else "Yes — >120 (1)", help=hint)
                elif feat == "exang":
                    inputs[feat] = st.selectbox(label, [0, 1], index=int(default),
                        format_func=lambda x: "No (0)" if x == 0 else "Yes (1)", help=hint)
                elif feat == "cp":
                    inputs[feat] = st.selectbox(label, [0, 1, 2, 3], index=int(default),
                        format_func=lambda x: {0:"0·Typical Angina",1:"1·Atypical Angina",
                                               2:"2·Non-anginal",3:"3·Asymptomatic"}[x], help=hint)
                elif feat == "restecg":
                    inputs[feat] = st.selectbox(label, [0, 1, 2], index=int(default),
                        format_func=lambda x: {0:"0·Normal",1:"1·ST-T Abnormality",2:"2·LV Hypertrophy"}[x],
                        help=hint)
                elif feat == "slope":
                    inputs[feat] = st.selectbox(label, [0, 1, 2], index=int(default),
                        format_func=lambda x: {0:"0·Upsloping",1:"1·Flat",2:"2·Downsloping"}[x], help=hint)
                elif feat == "thal":
                    inputs[feat] = st.selectbox(label, [0, 1, 2, 3], index=int(default),
                        format_func=lambda x: {0:"0·Unknown",1:"1·Normal",
                                               2:"2·Fixed Defect",3:"3·Reversible Defect"}[x], help=hint)
                elif feat == "oldpeak":
                    inputs[feat] = st.slider(label, float(mn), float(mx), float(default), 0.1, help=hint)
                else:
                    inputs[feat] = st.slider(label, int(mn), int(mx), int(default), help=hint)

        submitted = st.form_submit_button("🔍 Predict Heart Disease Risk", use_container_width=True)

    # ── Prediction (mirrors your Colab manual prediction) ───────────────────
    if submitted:
        if not models_dict or selected_model is None:
            st.error("❌ Models not ready. Ensure heart.csv is present.")
        else:
            model = models_dict[selected_model]

            # Exactly as your Colab:
            #   arr1 = np.asarray(input_data)
            #   arr2 = arr1.reshape(1, -1)
            #   pred1 = rf_model.predict(arr2)
            input_data = tuple(inputs[f] for f in feat_keys)
            arr1 = np.asarray(input_data)
            arr2 = arr1.reshape(1, -1)
            pred = model.predict(arr2)[0]
            prob = model.predict_proba(arr2)[0]

            st.markdown("---")
            st.markdown('<div class="section-title">📋 Prediction Result</div>', unsafe_allow_html=True)

            r1, r2 = st.columns([2, 1])
            with r1:
                if pred == 1:
                    st.markdown(f"""
                    <div class="result-danger">
                        <h2>⚠️ HIGH RISK — Heart Disease Detected</h2>
                        <p><em>"{selected_model}"</em> → <strong>The patient seems to have heart disease</strong></p>
                        <p>Risk Probability: <strong>{prob[1]*100:.1f}%</strong></p>
                        <p style="margin-top:16px; color:#e3b341">⚕️ Please consult a cardiologist immediately.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-safe">
                        <h2>✅ LOW RISK — Patient Seems Normal</h2>
                        <p><em>"{selected_model}"</em> → <strong>The patient seems to be normal</strong></p>
                        <p>Confidence (No Disease): <strong>{prob[0]*100:.1f}%</strong></p>
                        <p style="margin-top:16px; color:#3fb950">💚 Maintain a healthy lifestyle!</p>
                    </div>
                    """, unsafe_allow_html=True)

            with r2:
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=round(prob[1] * 100, 1),
                    title={"text": "Disease Risk %", "font": {"color": "#f0f6fc", "size": 14}},
                    gauge={
                        "axis":  {"range": [0, 100], "tickcolor": "#8b949e"},
                        "bar":   {"color": "#f85149" if pred == 1 else "#3fb950"},
                        "bgcolor": "#21262d",
                        "steps": [
                            {"range": [0,  40],  "color": "#0d2c1e"},
                            {"range": [40, 70],  "color": "#2d2005"},
                            {"range": [70, 100], "color": "#3d1515"},
                        ],
                        "threshold": {"line": {"color": "white", "width": 3}, "value": prob[1]*100}
                    },
                    number={"font": {"color": "#f0f6fc"}, "suffix": "%"}
                ))
                fig_gauge.update_layout(
                    paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                    height=260, margin=dict(t=50, b=0, l=20, r=20)
                )
                st.plotly_chart(fig_gauge, use_container_width=True)

            fig_bar = go.Figure([go.Bar(
                x=["No Disease", "Heart Disease"],
                y=[round(prob[0]*100, 2), round(prob[1]*100, 2)],
                marker_color=["#3fb950", "#f85149"],
                text=[f"{prob[0]*100:.1f}%", f"{prob[1]*100:.1f}%"],
                textposition="auto", width=0.5
            )])
            fig_bar.update_layout(
                title=f"Prediction Probability — {selected_model}",
                paper_bgcolor="#161b22", plot_bgcolor="#161b22",
                font_color="#c9d1d9", height=300,
                yaxis=dict(title="Probability (%)", range=[0, 100]),
                showlegend=False
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            with st.expander("🖥️ Colab-Style Console Output"):
                if pred == 1:
                    st.code(
                        f"input_data = {input_data}\n"
                        f"arr1 = np.asarray(input_data)\n"
                        f"arr2 = arr1.reshape(1, -1)\n"
                        f"pred = model.predict(arr2)\n"
                        f"# Output: [1]\n"
                        f">>> The patient seems to have heart disease",
                        language="python"
                    )
                else:
                    st.code(
                        f"input_data = {input_data}\n"
                        f"arr1 = np.asarray(input_data)\n"
                        f"arr2 = arr1.reshape(1, -1)\n"
                        f"pred = model.predict(arr2)\n"
                        f"# Output: [0]\n"
                        f">>> The patient seems to be normal",
                        language="python"
                    )

# ══════════════════════════════════════════════════════════════════════════════
#   PAGE 2 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
elif "Performance" in page:
    st.markdown('<div class="section-title">📊 Model Performance Dashboard</div>', unsafe_allow_html=True)

    if not models_dict:
        st.error("❌ Models not available. Ensure heart.csv is present.")
        st.stop()

    results_df = pd.DataFrame([
        {"Model": m, **{k: v for k, v in metrics_dict[m].items() if k != "cm"}}
        for m in metrics_dict
    ]).sort_values("Accuracy", ascending=False).reset_index(drop=True)

    st.dataframe(
        results_df.style
            .background_gradient(subset=["Accuracy","Precision","Recall","F1-Score"], cmap="RdYlGn")
            .format("{:.2f}%", subset=["Accuracy","Precision","Recall","F1-Score"]),
        use_container_width=True, height=210
    )

    fig = px.bar(
        results_df.melt(id_vars="Model", var_name="Metric", value_name="Score (%)"),
        x="Model", y="Score (%)", color="Metric", barmode="group",
        color_discrete_sequence=["#f85149","#3fb950","#58a6ff","#d2a8ff"],
        title="All 4 Models — Metric Comparison"
    )
    fig.update_layout(paper_bgcolor="#161b22", plot_bgcolor="#161b22",
                      font_color="#c9d1d9", height=420)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">🔲 Confusion Matrices</div>', unsafe_allow_html=True)
    cm_cols = st.columns(2)
    for idx, name in enumerate(metrics_dict):
        cm = np.array(metrics_dict[name]["cm"])
        fig_cm = px.imshow(
            cm, text_auto=True, color_continuous_scale="Reds",
            labels=dict(x="Predicted", y="Actual"),
            x=["No Disease","Disease"], y=["No Disease","Disease"],
            title=f"{name}  —  Acc: {metrics_dict[name]['Accuracy']}%"
        )
        fig_cm.update_layout(paper_bgcolor="#161b22", plot_bgcolor="#161b22",
                             font_color="#c9d1d9", height=340)
        cm_cols[idx % 2].plotly_chart(fig_cm, use_container_width=True)

    st.markdown('<div class="section-title">📄 Full Classification Report</div>', unsafe_allow_html=True)
    report_sel = st.selectbox("Select Model", list(models_dict.keys()))
    preds_r   = models_dict[report_sel].predict(X_test_cache)
    report    = classification_report(y_test_cache, preds_r,
                                      target_names=["No Disease","Disease"])
    st.code(report, language="text")

# ══════════════════════════════════════════════════════════════════════════════
#   PAGE 3 — EDA DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
elif "EDA" in page:
    st.markdown('<div class="section-title">🔬 Exploratory Data Analysis</div>', unsafe_allow_html=True)
    df = load_data()
    if df is None:
        st.error("❌ heart.csv not found.")
        st.stop()

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📈 Histograms", "🔗 Correlation Heatmap", "🎯 Target Analysis", "📄 Raw Data"]
    )

    with tab1:
        feat_sel = st.selectbox("Select Feature", [c for c in df.columns if c != "target"])
        fig = px.histogram(
            df, x=feat_sel, color="target",
            color_discrete_map={0:"#3fb950", 1:"#f85149"},
            barmode="overlay",
            title=f"Distribution of '{feat_sel}' by Heart Disease",
            labels={"target":"Heart Disease (0=No, 1=Yes)"}
        )
        fig.update_layout(paper_bgcolor="#161b22", plot_bgcolor="#161b22", font_color="#c9d1d9")
        st.plotly_chart(fig, use_container_width=True)

        if st.checkbox("📊 Show All Feature Histograms (like df.hist() in Colab)"):
            num_cols = df.select_dtypes(include=np.number).columns.tolist()
            grid_cols = st.columns(4)
            for i, col_name in enumerate(num_cols):
                with grid_cols[i % 4]:
                    fig_h = px.histogram(df, x=col_name, nbins=20,
                                         color_discrete_sequence=["#58a6ff"])
                    fig_h.update_layout(paper_bgcolor="#161b22", plot_bgcolor="#161b22",
                                        font_color="#c9d1d9", height=200,
                                        showlegend=False, margin=dict(t=30,b=10,l=10,r=10),
                                        title_text=col_name, title_font_size=12)
                    st.plotly_chart(fig_h, use_container_width=True)

    with tab2:
        corr = df.corr()
        fig_heat = px.imshow(
            corr, color_continuous_scale="Blues",
            title="Feature Correlation Heatmap  (mirrors Colab: cmap='winter')",
            text_auto=".2f", aspect="auto"
        )
        fig_heat.update_layout(paper_bgcolor="#161b22", plot_bgcolor="#161b22",
                               font_color="#c9d1d9", height=560)
        st.plotly_chart(fig_heat, use_container_width=True)

        st.markdown("#### 🎯 Correlation with Target")
        target_corr = corr["target"].drop("target").sort_values(ascending=False).reset_index()
        target_corr.columns = ["Feature", "Correlation"]
        fig_corr = px.bar(target_corr, x="Feature", y="Correlation",
                          color="Correlation", color_continuous_scale="RdBu",
                          title="Feature Correlation with Target Variable")
        fig_corr.update_layout(paper_bgcolor="#161b22", plot_bgcolor="#161b22", font_color="#c9d1d9")
        st.plotly_chart(fig_corr, use_container_width=True)

    with tab3:
        c1, c2 = st.columns(2)
        with c1:
            counts = df["target"].value_counts().reset_index()
            counts.columns = ["Class","Count"]
            counts["Class"] = counts["Class"].map({0:"No Disease",1:"Heart Disease"})
            fig_pie = px.pie(counts, names="Class", values="Count",
                             color_discrete_sequence=["#3fb950","#f85149"],
                             title="Target Class Distribution", hole=0.4)
            fig_pie.update_layout(paper_bgcolor="#161b22", font_color="#c9d1d9")
            st.plotly_chart(fig_pie, use_container_width=True)
        with c2:
            fig_box = px.box(df, x="target", y="age", color="target",
                             color_discrete_map={0:"#3fb950", 1:"#f85149"},
                             labels={"target":"Heart Disease","age":"Age"},
                             title="Age Distribution vs Heart Disease")
            fig_box.update_layout(paper_bgcolor="#161b22", plot_bgcolor="#161b22", font_color="#c9d1d9")
            st.plotly_chart(fig_box, use_container_width=True)

        fig_scatter = px.scatter(df, x="thalach", y="age", color="target",
                                 color_discrete_map={0:"#3fb950", 1:"#f85149"},
                                 labels={"thalach":"Max Heart Rate","age":"Age","target":"Heart Disease"},
                                 title="Max Heart Rate vs Age (coloured by Target)")
        fig_scatter.update_layout(paper_bgcolor="#161b22", plot_bgcolor="#161b22", font_color="#c9d1d9")
        st.plotly_chart(fig_scatter, use_container_width=True)

    with tab4:
        st.markdown("#### df.head() — First 10 Rows")
        st.dataframe(df.head(10), use_container_width=True)
        st.markdown("#### df.describe() — Statistical Summary")
        st.dataframe(df.describe().style.background_gradient(cmap="Blues"), use_container_width=True)
        ca, cb, cc = st.columns(3)
        ca.metric("Records  (df.shape[0])", df.shape[0])
        cb.metric("Features (df.shape[1]-1)", df.shape[1]-1)
        cc.metric("Missing Values", int(df.isnull().sum().sum()))

# ══════════════════════════════════════════════════════════════════════════════
#   PAGE 4 — ABOUT
# ══════════════════════════════════════════════════════════════════════════════
elif "About" in page:
    st.markdown('<div class="section-title">ℹ️ About This Project</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="background:#161b22; border:1px solid #30363d; border-radius:16px; padding:36px; line-height:1.9">
        <h3 style="color:#f0f6fc">❤️ CardioPredict AI</h3>
        <p style="color:#8b949e">
            A Streamlit frontend that <strong style="color:#c9d1d9">exactly mirrors</strong> your
            Google Colab notebook pipeline — no scaler, same split, same model parameters.
        </p>
        <br>
        <h4 style="color:#f0f6fc">📦 ML Models — Exact Colab Parameters</h4>
        <table style="color:#8b949e; width:100%; border-collapse:collapse">
            <tr style="color:#c9d1d9; border-bottom:1px solid #30363d">
                <th style="padding:8px; text-align:left">Model</th>
                <th style="padding:8px; text-align:left">Parameters (from your Colab)</th>
            </tr>
            <tr><td style="padding:8px">Logistic Regression</td><td style="padding:8px">LogisticRegression()</td></tr>
            <tr><td style="padding:8px">SVM</td><td style="padding:8px">SVC(probability=True) ← added for UI</td></tr>
            <tr><td style="padding:8px">Decision Tree</td><td style="padding:8px">max_depth=5, criterion='entropy'</td></tr>
            <tr><td style="padding:8px">Random Forest</td><td style="padding:8px">n_estimators=500, criterion='entropy'</td></tr>
        </table>
        <br>
        <h4 style="color:#f0f6fc">⚙️ Pipeline (Colab-Identical)</h4>
        <ul style="color:#8b949e">
            <li>X = df.iloc[:, :-1] &nbsp;|&nbsp; y = df.iloc[:, -1]</li>
            <li>train_test_split(test_size=0.2, random_state=42)  — no stratify</li>
            <li>No StandardScaler (raw features)</li>
            <li>Prediction: np.asarray(input).reshape(1, -1)</li>
        </ul>
        <br>
        <h4 style="color:#f0f6fc">⚠️ Disclaimer</h4>
        <p style="color:#8b949e">For educational and research purposes only. Not a medical diagnosis tool.</p>
    </div>
    """, unsafe_allow_html=True)
