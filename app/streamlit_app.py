#!/usr/bin/env python3
from __future__ import annotations
import sys
import io
import json
import joblib
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# Add src to sys.path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from clean_text import clean_text, batch_clean
from features import extract_numeric_features, highlight_text_flags, COMMON_FAKE_PHRASES
from train import train, build_pipeline, load_data

# File paths
BASE_DIR = Path(__file__).resolve().parents[1]
PIPE_PATH = BASE_DIR / "outputs" / "pipeline.joblib"
METRICS_PATH = BASE_DIR / "outputs" / "metrics.json"
DATA_PATH = BASE_DIR / "data" / "reviews_sample.csv"

# Streamlit Page Config
st.set_page_config(
    page_title="Fake Review Detector Pro",
    page_icon="🕵️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Injected Modern Custom CSS Design System
st.markdown("""
<style>
    /* Dark Theme & Modern Typography */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main Background & Container styling */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #f8fafc;
    }
    
    /* Card Glassmorphism Containers */
    .glass-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
    }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.8), rgba(15, 23, 42, 0.9));
        border-left: 4px solid #3b82f6;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .metric-title {
        color: #94a3b8;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .metric-value {
        color: #f8fafc;
        font-size: 1.8rem;
        font-weight: 700;
        margin-top: 4px;
    }

    /* Badges */
    .badge-fake {
        background: linear-gradient(135deg, #ef4444, #dc2626);
        color: white;
        padding: 8px 20px;
        border-radius: 30px;
        font-weight: 700;
        font-size: 1.3rem;
        display: inline-block;
        box-shadow: 0 4px 15px rgba(239, 68, 68, 0.4);
    }
    
    .badge-real {
        background: linear-gradient(135deg, #10b981, #059669);
        color: white;
        padding: 8px 20px;
        border-radius: 30px;
        font-weight: 700;
        font-size: 1.3rem;
        display: inline-block;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.4);
    }
    
    /* Highlight box */
    .highlight-box {
        background: rgba(15, 23, 42, 0.8);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 16px;
        line-height: 1.8;
        font-size: 1.05rem;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0b1329 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
</style>
""", unsafe_allow_html=True)

# Helper function to load model
@st.cache_resource
def load_pipeline():
    if PIPE_PATH.exists():
        return joblib.load(PIPE_PATH)
    return None

pipe = load_pipeline()

# Load metrics json helper
def load_metrics():
    if METRICS_PATH.exists():
        try:
            return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None

metrics_data = load_metrics()

# Sidebar Header & Navigation
with st.sidebar:
    st.image("https://img.icons8.com/color/96/detective.png", width=70)
    st.title("Fake Review Detector")
    st.caption("AI-Powered Sentiment & Behavioral Review Analysis")
    st.markdown("---")
    
    nav_option = st.radio(
        "Navigation Menu",
        [
            "🔍 Live Review Analyzer",
            "📂 Batch CSV Analyzer",
            "📊 Model Analytics & Metrics",
            "🧪 Dataset Explorer",
            "⚙️ Model Retraining Suite"
        ],
        index=0
    )
    
    st.markdown("---")
    # System Status Widget in Sidebar
    st.markdown("#### ⚡ System Status")
    if pipe is not None:
        st.success("Model Active & Ready")
        if metrics_data:
            st.caption(f"ROC-AUC: **{metrics_data.get('roc_auc', 1.0):.2f}** | Accuracy: **{metrics_data.get('report', {}).get('accuracy', 1.0)*100:.0f}%**")
    else:
        st.warning("Model Not Found")
        st.caption("Please run training in Retraining Suite.")

# Header Bar
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.title(nav_option)
with col_h2:
    st.write("")
    st.markdown(
        '<div style="text-align: right; padding-top: 10px;">'
        '<span style="background: rgba(59, 130, 246, 0.2); color: #60a5fa; padding: 6px 14px; border-radius: 20px; font-weight: 600; font-size: 0.85rem; border: 1px solid rgba(59, 130, 246, 0.3);">'
        'v2.0 Production ML</span></div>',
        unsafe_allow_html=True
    )

st.markdown("---")

# ==============================================================================
# SECTION 1: LIVE REVIEW ANALYZER
# ==============================================================================
if nav_option == "🔍 Live Review Analyzer":
    st.markdown("""
    Analyze individual customer or product reviews in real time using hybrid **TF-IDF Vectorization**, 
    **Logistic Regression**, and **Linguistic & Behavioral Feature Extraction**.
    """)
    
    # Preset Selector
    presets = {
        "Custom Input": "",
        "Example 1 (Fake): Overly enthusiastic with clichés": "Best product ever!!! Totally changed my life. HIGHLY RECOMMEND!!!",
        "Example 2 (Fake): Sponsored / Discount disclaimer": "Use my discount code for more!! Sponsored but opinions are my own",
        "Example 3 (Fake): Free sample disclosure": "I got a free sample for my honest review and wow this is amazing",
        "Example 4 (Real): Balanced product feedback": "The build quality is decent for the price, but the battery drains fast.",
        "Example 5 (Real): Service & delivery review": "Arrived on time and works as expected. Packaging could be better."
    }
    
    selected_preset = st.selectbox("💡 Quick Select Sample Review Preset:", list(presets.keys()))
    default_text = presets[selected_preset]
    
    col_input, col_config = st.columns([2, 1])
    
    with col_input:
        txt = st.text_area(
            "Paste or Type Product Review:",
            value=default_text,
            height=180,
            placeholder="Type or paste any product review text here to analyze..."
        )
    
    with col_config:
        st.markdown("#### 🎛️ Analysis Controls")
        thr = st.slider("Decision Threshold (Fake sensitivity)", 0.1, 0.9, 0.5, 0.05)
        st.caption("Higher threshold requires higher confidence before marking as FAKE.")
        
        analyze_btn = st.button("🚀 Analyze Review", type="primary", use_container_width=True)

    if analyze_btn or txt.strip():
        if pipe is None:
            st.error("Model pipeline not found. Please train the model in the Retraining Suite tab.")
        elif not txt.strip():
            st.warning("Please enter a review text to analyze.")
        else:
            # Run Prediction Pipeline
            s = clean_text(txt)
            df = pd.DataFrame([{"text": txt, "text_clean": s}])
            num = extract_numeric_features([txt])
            X = pd.concat([df, num], axis=1)
            
            prob = float(pipe.predict_proba(X)[0, 1])
            is_fake = prob >= thr
            label_str = "FAKE REVIEW" if is_fake else "AUTHENTIC REVIEW"
            
            st.markdown("---")
            st.subheader("🔍 Analysis Results")
            
            res_col1, res_col2 = st.columns([1, 1])
            
            with res_col1:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown("#### Classification Result")
                if is_fake:
                    st.markdown(f'<div class="badge-fake">🚨 {label_str}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="badge-real">✅ {label_str}</div>', unsafe_allow_html=True)
                
                st.write("")
                st.write(f"**Fake Probability Score:** `{prob:.1%}` (Threshold: `{thr:.2f}`)")
                
                # Gauge representation using Plotly
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Fake Risk Meter (%)", 'font': {'color': 'white', 'size': 14}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickcolor': "white"},
                        'bar': {'color': "#ef4444" if is_fake else "#10b981"},
                        'steps': [
                            {'range': [0, 40], 'color': "rgba(16, 185, 129, 0.2)"},
                            {'range': [40, 70], 'color': "rgba(245, 158, 11, 0.2)"},
                            {'range': [70, 100], 'color': "rgba(239, 68, 68, 0.2)"}
                        ],
                        'threshold': {
                            'line': {'color': "white", 'width': 4},
                            'thickness': 0.75,
                            'value': thr * 100
                        }
                    }
                ))
                fig_gauge.update_layout(height=200, margin=dict(l=20, r=20, t=30, b=20), paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"})
                st.plotly_chart(fig_gauge, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with res_col2:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown("#### 🧬 Behavioral & Linguistic Features")
                
                f_col1, f_col2 = st.columns(2)
                with f_col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">Sentiment Polarity</div>
                        <div class="metric-value">{num['sentiment'].iloc[0]:.2f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">ALL-CAPS Words</div>
                        <div class="metric-value">{num['all_caps_tokens'].iloc[0]}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with f_col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">Exclamations (!)</div>
                        <div class="metric-value">{num['exclamation_count'].iloc[0]}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">Cliché Phrases</div>
                        <div class="metric-value">{num['repeated_phrases'].iloc[0]}</div>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # Suspicious Flags Highlighter
            st.markdown("#### 🚩 Suspicious Pattern Highlighter")
            highlighted_html = highlight_text_flags(txt)
            st.markdown(f'<div class="highlight-box">{highlighted_html}</div>', unsafe_allow_html=True)
            st.caption("🔴 Red badges indicate synthetic/cliché phrases. 🟠 Orange badges highlight excessive ALL-CAPS usage.")

# ==============================================================================
# SECTION 2: BATCH CSV ANALYZER
# ==============================================================================
elif nav_option == "📂 Batch CSV Analyzer":
    st.markdown("""
    Upload a custom CSV file of reviews to run high-speed bulk inference, visualize distributions, and export predictions.
    """)
    
    col_up, col_sample = st.columns([2, 1])
    
    with col_up:
        uploaded_file = st.file_uploader("Upload Reviews CSV File", type=["csv"])
    
    with col_sample:
        st.write("")
        st.write("")
        use_sample = st.button("📁 Load Default Sample Dataset", use_container_width=True)

    df_batch = None
    if uploaded_file is not None:
        try:
            df_batch = pd.read_csv(uploaded_file)
            st.success(f"Successfully loaded {len(df_batch)} rows from uploaded file.")
        except Exception as e:
            st.error(f"Error loading CSV: {e}")
    elif use_sample:
        if DATA_PATH.exists():
            df_batch = pd.read_csv(DATA_PATH)
            st.success(f"Successfully loaded sample dataset ({len(df_batch)} rows).")

    if df_batch is not None:
        text_cols = [c for c in df_batch.columns if "text" in c.lower() or "review" in c.lower()]
        selected_text_col = st.selectbox(
            "Select Column Containing Review Text:",
            options=df_batch.columns,
            index=df_batch.columns.get_loc(text_cols[0]) if text_cols else 0
        )
        
        batch_thr = st.slider("Batch Threshold for FAKE", 0.1, 0.9, 0.5, 0.05, key="batch_thr")
        
        if st.button("⚡ Run Bulk Batch Inference", type="primary"):
            if pipe is None:
                st.error("Model pipeline is missing. Please train the model first.")
            else:
                with st.spinner("Processing batch predictions..."):
                    texts = df_batch[selected_text_col].astype(str).tolist()
                    cleaned_texts = batch_clean(texts)
                    df_proc = pd.DataFrame({"text": texts, "text_clean": cleaned_texts})
                    num_feats = extract_numeric_features(texts)
                    X_batch = pd.concat([df_proc, num_feats], axis=1)
                    
                    probs = pipe.predict_proba(X_batch)[:, 1]
                    preds = ["FAKE" if p >= batch_thr else "REAL" for p in probs]
                    
                    df_result = df_batch.copy()
                    df_result["predicted_label"] = preds
                    df_result["fake_probability"] = np.round(probs, 4)
                    
                    st.session_state["batch_results"] = df_result

        if "batch_results" in st.session_state:
            df_res = st.session_state["batch_results"]
            st.markdown("---")
            st.subheader("📊 Batch Inference Dashboard")
            
            total_count = len(df_res)
            fake_count = sum(df_res["predicted_label"] == "FAKE")
            real_count = sum(df_res["predicted_label"] == "REAL")
            avg_prob = df_res["fake_probability"].mean()
            
            b_col1, b_col2, b_col3, b_col4 = st.columns(4)
            b_col1.metric("Total Reviews", f"{total_count:,}")
            b_col2.metric("Predicted FAKE", f"{fake_count:,}", delta=f"{fake_count/total_count:.1%}", delta_color="inverse")
            b_col3.metric("Predicted REAL", f"{real_count:,}", delta=f"{real_count/total_count:.1%}", delta_color="normal")
            b_col4.metric("Avg Fake Probability", f"{avg_prob:.1%}")
            
            # Visual Distribution Charts
            c_chart1, c_chart2 = st.columns(2)
            
            with c_chart1:
                fig_pie = px.pie(
                    names=["REAL", "FAKE"],
                    values=[real_count, fake_count],
                    title="Real vs Fake Review Ratio",
                    color_discrete_sequence=["#10b981", "#ef4444"],
                    hole=0.4
                )
                fig_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color="white")
                st.plotly_chart(fig_pie, use_container_width=True)
                
            with c_chart2:
                fig_hist = px.histogram(
                    df_res,
                    x="fake_probability",
                    nbins=20,
                    title="Fake Probability Score Distribution",
                    labels={"fake_probability": "Fake Risk Probability"},
                    color_discrete_sequence=["#3b82f6"]
                )
                fig_hist.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color="white", yaxis_title="Review Count")
                st.plotly_chart(fig_hist, use_container_width=True)
                
            # Filterable Results Table
            st.markdown("#### 📋 Detailed Predictions Table")
            filter_label = st.radio("Filter Table by Label:", ["All", "FAKE Only", "REAL Only"], horizontal=True)
            
            if filter_label == "FAKE Only":
                df_show = df_res[df_res["predicted_label"] == "FAKE"]
            elif filter_label == "REAL Only":
                df_show = df_res[df_res["predicted_label"] == "REAL"]
            else:
                df_show = df_res
                
            st.dataframe(df_show, use_container_width=True, height=350)
            
            # Download Button
            csv_buffer = df_res.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Analyzed CSV Results",
                data=csv_buffer,
                file_name="fake_review_predictions.csv",
                mime="text/csv",
                type="primary"
            )

# ==============================================================================
# SECTION 3: MODEL ANALYTICS & METRICS
# ==============================================================================
elif nav_option == "📊 Model Analytics & Metrics":
    st.markdown("""
    Evaluate model diagnostics, statistical benchmarks, confusion matrix, ROC/PR curves, and word feature importances.
    """)
    
    if metrics_data is not None:
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("ROC-AUC Score", f"{metrics_data.get('roc_auc', 1.0):.4f}")
        m_col2.metric("Average Precision", f"{metrics_data.get('avg_precision', 1.0):.4f}")
        m_col3.metric("Model Accuracy", f"{metrics_data.get('report', {}).get('accuracy', 1.0)*100:.1f}%")
        m_col4.metric("Algorithm", "Logistic Regression + TFIDF")
        
        st.markdown("---")
        
        tab_plots, tab_features, tab_report = st.tabs(["📉 Evaluation Curves & Matrices", "🔑 Feature Importance / Word Weights", "📄 Classification Report JSON"])
        
        with tab_plots:
            img_col1, img_col2, img_col3 = st.columns(3)
            
            cm_img = BASE_DIR / "outputs" / "confusion_matrix.png"
            roc_img = BASE_DIR / "outputs" / "roc_curve.png"
            pr_img = BASE_DIR / "outputs" / "pr_curve.png"
            
            with img_col1:
                st.markdown("##### Confusion Matrix")
                if cm_img.exists():
                    st.image(str(cm_img), use_container_width=True)
                else:
                    st.info("Confusion matrix image not found.")
                    
            with img_col2:
                st.markdown("##### ROC Curve")
                if roc_img.exists():
                    st.image(str(roc_img), use_container_width=True)
                else:
                    st.info("ROC curve image not found.")
                    
            with img_col3:
                st.markdown("##### Precision-Recall Curve")
                if pr_img.exists():
                    st.image(str(pr_img), use_container_width=True)
                else:
                    st.info("Precision-recall curve image not found.")
                    
        with tab_features:
            st.markdown("##### Top Predictive Words & N-grams (Model Coefficients)")
            if pipe is not None:
                try:
                    pre_transformer = pipe.named_steps["pre"]
                    tfidf_vec = pre_transformer.named_transformers_["tfidf"]
                    feature_names = list(tfidf_vec.get_feature_names_out())
                    num_feature_names = ["sentiment", "exclamation_count", "all_caps_tokens", "repeated_phrases", "char_length", "unique_word_ratio"]
                    all_features = feature_names + num_feature_names
                    
                    coefs = pipe.named_steps["clf"].coef_[0]
                    
                    feat_df = pd.DataFrame({"feature": all_features, "coef": coefs})
                    top_fake = feat_df.sort_values(by="coef", ascending=False).head(10)
                    top_real = feat_df.sort_values(by="coef", ascending=True).head(10)
                    
                    f_c1, f_c2 = st.columns(2)
                    with f_c1:
                        fig_fake_words = px.bar(
                            top_fake,
                            x="coef",
                            y="feature",
                            orientation='h',
                            title="Top Features Driving FAKE Classification",
                            color_discrete_sequence=["#ef4444"]
                        )
                        fig_fake_words.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color="white", yaxis=dict(autorange="reversed"))
                        st.plotly_chart(fig_fake_words, use_container_width=True)
                        
                    with f_c2:
                        fig_real_words = px.bar(
                            top_real,
                            x="coef",
                            y="feature",
                            orientation='h',
                            title="Top Features Driving REAL Classification",
                            color_discrete_sequence=["#10b981"]
                        )
                        fig_real_words.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color="white")
                        st.plotly_chart(fig_real_words, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not extract feature importances: {e}")
                    
        with tab_report:
            st.json(metrics_data)
    else:
        st.warning("Metrics metadata missing. Train model in Retraining Suite.")

# ==============================================================================
# SECTION 4: DATASET EXPLORER
# ==============================================================================
elif nav_option == "🧪 Dataset Explorer":
    st.markdown("""
    Explore and inspect the benchmark training dataset used for training the Fake Review Detector model.
    """)
    
    if DATA_PATH.exists():
        df_data = pd.read_csv(DATA_PATH)
        
        d_col1, d_col2, d_col3 = st.columns(3)
        d_col1.metric("Total Dataset Size", f"{len(df_data):,} rows")
        d_col2.metric("Total Real Reviews", f"{sum(df_data['label'].astype(str).str.upper()=='REAL'):,}")
        d_col3.metric("Total Fake Reviews", f"{sum(df_data['label'].astype(str).str.upper()=='FAKE'):,}")
        
        st.markdown("---")
        
        s_col1, s_col2 = st.columns([1, 2])
        
        with s_col1:
            st.markdown("#### 🎯 Filter & Search")
            label_filter = st.multiselect("Filter by Label:", options=["REAL", "FAKE"], default=["REAL", "FAKE"])
            search_query = st.text_input("Search Keyword in Text:", placeholder="Type to filter...")
            
            df_filtered = df_data.copy()
            if label_filter:
                df_filtered = df_filtered[df_filtered["label"].astype(str).str.upper().isin(label_filter)]
            if search_query.strip():
                df_filtered = df_filtered[df_filtered["text"].str.contains(search_query, case=False, na=False)]
                
            st.write(f"Showing **{len(df_filtered)}** out of **{len(df_data)}** entries.")
            
        with s_col2:
            st.markdown("#### 📊 Label Distribution")
            fig_data_dist = px.bar(
                df_filtered["label"].value_counts().reset_index(),
                x="label",
                y="count",
                color="label",
                color_discrete_map={"REAL": "#10b981", "FAKE": "#ef4444"},
                title="Class Count Breakdown"
            )
            fig_data_dist.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color="white", height=250)
            st.plotly_chart(fig_data_dist, use_container_width=True)
            
        st.markdown("#### 📖 Dataset Table Browser")
        st.dataframe(df_filtered, use_container_width=True, height=400)
    else:
        st.error(f"Sample dataset file not found at {DATA_PATH}.")

# ==============================================================================
# SECTION 5: MODEL RETRAINING SUITE
# ==============================================================================
elif nav_option == "⚙️ Model Retraining Suite":
    st.markdown("""
    Retrain the machine learning model with custom parameters or new training datasets directly from the interface.
    """)
    
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 🛠️ Retraining Configurations")
    
    col_t1, col_t2 = st.columns([2, 1])
    
    with col_t1:
        train_csv_file = st.file_uploader("Upload Custom Training CSV File (Must contain `text` and `label` columns)", type=["csv"])
        if train_csv_file is None:
            st.info("Using default dataset (`data/reviews_sample.csv`) for retraining.")
            target_csv = DATA_PATH
        else:
            # Save uploaded CSV temporarily
            temp_path = BASE_DIR / "data" / "uploaded_train.csv"
            temp_path.write_bytes(train_csv_file.getvalue())
            target_csv = temp_path
            st.success(f"Custom training file uploaded ({train_csv_file.name}).")
            
    with col_t2:
        test_size = st.slider("Test Split Ratio", 0.1, 0.4, 0.2, 0.05)
        seed = st.number_input("Random Seed", value=42, step=1)
        max_features = st.select_slider("Max TF-IDF Features", options=[5000, 10000, 20000, 50000], value=20000)

    st.write("")
    if st.button("🚀 Train & Save New Pipeline Model", type="primary", use_container_width=True):
        with st.spinner("Retraining model pipeline... Please wait."):
            try:
                new_metrics = train(target_csv, BASE_DIR / "outputs", test_size=test_size, seed=seed)
                st.cache_resource.clear()
                st.success("🎉 Model trained and saved successfully!")
                
                st.markdown("#### 📈 New Model Performance Metrics")
                st.json(new_metrics)
            except Exception as err:
                st.error(f"Failed to train model: {err}")
    st.markdown('</div>', unsafe_allow_html=True)
