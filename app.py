"""
AL Essentials Competency Self-Assessment & Personalized Learning Roadmap
========================================================================
Roberta Christopher, Jacksonville University
DSIM 608 Capstone

Run: streamlit run app.py

Data Source: O*NET 30.1 (December 2025), CC-BY 4.0 License
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
import sqlite3
import json
import os
import io
from datetime import datetime

# ── Page Config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="AL Essentials Competency Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Paths (self-contained: all data in ./data/) ──────────────────
APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(APP_DIR, "data")

# Try local data/ first (for Streamlit Cloud), fall back to project paths
DB_PATH = os.path.join(DATA_DIR, "onet_healthcare.db")
if not os.path.exists(DB_PATH):
    DB_PATH = os.path.join(os.path.dirname(APP_DIR), "Week_3", "outputs", "onet_healthcare.db")

CROSSWALK_PATH = os.path.join(DATA_DIR, "crosswalk_45_competencies_to_onet.json")
if not os.path.exists(CROSSWALK_PATH):
    CROSSWALK_PATH = os.path.join(os.path.dirname(APP_DIR), "data", "crosswalk_45_competencies_to_onet.json")

FEATURE_IMP_PATH = os.path.join(DATA_DIR, "feature_importance.csv")
CLUSTER_PATH = os.path.join(DATA_DIR, "cluster_assignments.csv")
MODEL_COMP_PATH = os.path.join(DATA_DIR, "model_comparison_results.csv")

# ── Constants ─────────────────────────────────────────────────────
TARGET_CODES = [
    '29-1171.00', '29-1141.04', '29-1151.00', '29-1161.00',
    '29-1141.00', '25-1072.00', '15-1211.01', '11-9111.00',
]

OCC_LABELS = {
    '29-1171.00': 'Nurse Practitioner',
    '29-1141.04': 'Clinical Nurse Specialist',
    '29-1151.00': 'Nurse Anesthetist',
    '29-1161.00': 'Nurse Midwife',
    '29-1141.00': 'Registered Nurse',
    '25-1072.00': 'Nursing Instructor',
    '15-1211.01': 'Health Informatics Specialist',
    '11-9111.00': 'Medical/Health Services Manager',
}

DOMAIN_LABELS = {
    'knowledge': 'Knowledge', 'skills': 'Skills', 'abilities': 'Abilities',
    'work_activities': 'Work Activities', 'work_styles': 'Work Styles',
    'work_context': 'Work Context',
}
DOMAIN_ORDER = ['knowledge', 'skills', 'abilities', 'work_activities',
                'work_styles', 'work_context']

AACN_DOMAIN_NAMES = {
    "1": "Knowledge for Nursing Practice",
    "2": "Person-Centered Care",
    "3": "Population Health",
    "4": "Scholarship for Nursing Practice",
    "5": "Quality and Safety",
    "6": "Interprofessional Partnerships",
    "7": "Systems-Based Practice",
    "8": "Informatics and Healthcare Technologies",
    "9": "Professionalism",
    "10": "Personal, Professional Development and Leadership",
}

DREYFUS_SCALE = {
    1: "Novice",
    2: "Advanced Beginner",
    3: "Competent",
    4: "Proficient",
    5: "Expert",
}

# Color palette
COLORS = {
    'primary': '#1B2A4A',
    'secondary': '#2a9d8f',
    'accent': '#e07a5f',
    'warning': '#e9c46a',
    'light': '#f8f9fa',
    'dark': '#264653',
}


# ── Data Loading ──────────────────────────────────────────────────
@st.cache_data
def load_onet_data():
    """Load O*NET normalized scores from SQLite."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("""
        SELECT v.soc_code, o.title, o.is_target,
               v.element_name, v.domain, v.normalized_value
        FROM v_normalized_scores v
        INNER JOIN occupations o ON v.soc_code = o.soc_code
    """, conn)
    jz = pd.read_sql_query("SELECT soc_code, job_zone FROM job_zones", conn)
    conn.close()
    return df, jz


@st.cache_data
def load_crosswalk():
    """Load AACN-to-O*NET crosswalk."""
    with open(CROSSWALK_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)


@st.cache_data
def load_feature_importance():
    """Load RF feature importance from Week 4."""
    if os.path.exists(FEATURE_IMP_PATH):
        return pd.read_csv(FEATURE_IMP_PATH)
    return None


@st.cache_data
def load_cluster_assignments():
    """Load cluster assignments from Week 4."""
    if os.path.exists(CLUSTER_PATH):
        return pd.read_csv(CLUSTER_PATH)
    return None


@st.cache_data
def load_model_comparison():
    """Load 6-model comparison results."""
    if os.path.exists(MODEL_COMP_PATH):
        return pd.read_csv(MODEL_COMP_PATH)
    return None


@st.cache_data
def build_occupation_matrix(df):
    """Build wide-format occupation-dimension matrix."""
    df_copy = df.copy()
    df_copy['dimension'] = df_copy['domain'] + ':' + df_copy['element_name']
    matrix = df_copy.pivot_table(
        index='soc_code', columns='dimension',
        values='normalized_value', aggfunc='mean'
    )
    dim_cols = list(matrix.columns)
    matrix = matrix.dropna()
    return matrix, dim_cols


@st.cache_data
def compute_np_benchmark(df):
    """Compute NP (29-1171.00) domain-level benchmark scores."""
    np_data = df[df['soc_code'] == '29-1171.00']
    return np_data.groupby('domain')['normalized_value'].mean().to_dict()


def get_competency_importance(crosswalk, feat_imp):
    """Map RF feature importance to each AACN competency."""
    if feat_imp is None:
        return {}

    imp_map = dict(zip(feat_imp['feature'], feat_imp['importance']))
    comp_importance = {}

    for comp_id, comp_info in crosswalk.items():
        total_imp = 0
        n_matched = 0
        for domain_label, dimensions in comp_info['onet_dimensions'].items():
            domain_key = [k for k, v in DOMAIN_LABELS.items() if v == domain_label]
            if not domain_key:
                continue
            dk = domain_key[0]
            for dim_name in dimensions:
                feature_key = f"{dk}:{dim_name}"
                if feature_key in imp_map:
                    total_imp += imp_map[feature_key]
                    n_matched += 1
        comp_importance[comp_id] = {
            'importance': total_imp,
            'n_features': n_matched,
            'strength': comp_info['mapping_strength'],
        }
    return comp_importance


def compute_gap_scores(self_assessment, crosswalk, np_benchmark, feat_imp):
    """Compute weighted gap scores for each competency."""
    comp_importance = get_competency_importance(crosswalk, feat_imp)
    gaps = []

    for comp_id, comp_info in crosswalk.items():
        student_score = self_assessment.get(comp_id, 3)
        student_norm = (student_score - 1) / 4  # normalize 1-5 to 0-1

        # Compute benchmark from NP's O*NET dimensions
        benchmark_scores = []
        for domain_label, dimensions in comp_info['onet_dimensions'].items():
            domain_key = [k for k, v in DOMAIN_LABELS.items() if v == domain_label]
            if not domain_key:
                continue
            dk = domain_key[0]
            if dk in np_benchmark:
                benchmark_scores.append(np_benchmark[dk])

        benchmark = np.mean(benchmark_scores) if benchmark_scores else 0.5
        gap = benchmark - student_norm
        importance = comp_importance.get(comp_id, {}).get('importance', 0)
        weighted_gap = gap * (1 + importance * 10)  # scale importance impact

        aacn_domain = comp_id.split('.')[0]
        gaps.append({
            'competency_id': comp_id,
            'competency': comp_info['competency'],
            'aacn_domain': aacn_domain,
            'aacn_domain_name': AACN_DOMAIN_NAMES.get(aacn_domain, f'Domain {aacn_domain}'),
            'student_score': student_score,
            'student_norm': student_norm,
            'benchmark': benchmark,
            'gap': gap,
            'ml_importance': importance,
            'weighted_gap': weighted_gap,
            'mapping_strength': comp_info['mapping_strength'],
        })

    return pd.DataFrame(gaps)


# ── Export / Import Helpers ────────────────────────────────────────
def export_assessment_to_csv(self_assessment, crosswalk, np_benchmark, feat_imp):
    """Build a CSV with the student's full assessment, gaps, and roadmap."""
    gaps_df = compute_gap_scores(self_assessment, crosswalk, np_benchmark, feat_imp)
    export_df = gaps_df[[
        'competency_id', 'competency', 'aacn_domain_name',
        'student_score', 'benchmark', 'gap', 'ml_importance',
        'weighted_gap', 'mapping_strength'
    ]].copy()
    export_df.columns = [
        'Competency ID', 'Competency', 'AACN Domain',
        'Self-Assessment Score', 'NP Benchmark', 'Gap',
        'ML Importance', 'Weighted Gap', 'Mapping Strength'
    ]
    # Add metadata rows at top
    meta = pd.DataFrame({
        'Competency ID': ['_META_DATE', '_META_MEAN', '_META_GAPS'],
        'Competency': [
            f'Assessment Date: {datetime.now().strftime("%Y-%m-%d %H:%M")}',
            f'Mean Score: {np.mean(list(self_assessment.values())):.2f}',
            f'Competencies with Gaps: {len(gaps_df[gaps_df["gap"] > 0])}'
        ],
        'AACN Domain': ['', '', ''],
        'Self-Assessment Score': ['', '', ''],
        'NP Benchmark': ['', '', ''],
        'Gap': ['', '', ''],
        'ML Importance': ['', '', ''],
        'Weighted Gap': ['', '', ''],
        'Mapping Strength': ['', '', ''],
    })
    return pd.concat([meta, export_df], ignore_index=True)


def parse_imported_csv(uploaded_file):
    """Parse a previously exported CSV to restore assessment scores."""
    df = pd.read_csv(uploaded_file)
    # Skip metadata rows
    data_rows = df[~df['Competency ID'].str.startswith('_META', na=False)]
    scores = {}
    for _, row in data_rows.iterrows():
        comp_id = str(row['Competency ID']).strip()
        score = row['Self-Assessment Score']
        if pd.notna(score) and comp_id:
            try:
                scores[comp_id] = int(float(score))
            except (ValueError, TypeError):
                pass
    return scores


def export_for_instructor(self_assessment, crosswalk, np_benchmark, feat_imp,
                          student_name=""):
    """Build a single-row summary suitable for instructor collection."""
    gaps_df = compute_gap_scores(self_assessment, crosswalk, np_benchmark, feat_imp)
    row = {
        'Student Name': student_name,
        'Date': datetime.now().strftime("%Y-%m-%d %H:%M"),
        'Mean Score': round(np.mean(list(self_assessment.values())), 2),
        'Competencies with Gaps': int((gaps_df['gap'] > 0).sum()),
        'Largest Gap Competency': gaps_df.loc[gaps_df['gap'].idxmax(), 'competency_id']
                                  if not gaps_df.empty else '',
        'Largest Gap Value': round(gaps_df['gap'].max(), 4) if not gaps_df.empty else 0,
        'Mean Weighted Gap': round(gaps_df['weighted_gap'].mean(), 4),
    }
    # Add per-domain average scores
    for d_num, d_name in AACN_DOMAIN_NAMES.items():
        d_scores = [v for k, v in self_assessment.items() if k.startswith(f"{d_num}.")]
        row[f'D{d_num} Avg'] = round(np.mean(d_scores), 2) if d_scores else ''
    # Add all 45 individual scores
    for comp_id in sorted(crosswalk.keys(), key=lambda x: (int(x.split('.')[0]), int(x.split('.')[1]))):
        row[f'Score_{comp_id}'] = self_assessment.get(comp_id, '')
    return pd.DataFrame([row])


# ── App Layout ────────────────────────────────────────────────────
def main():
    # Load all data
    df, jz = load_onet_data()
    crosswalk = load_crosswalk()
    feat_imp = load_feature_importance()
    clusters = load_cluster_assignments()
    model_comp = load_model_comparison()
    np_benchmark = compute_np_benchmark(df)

    # Store in session state for export functions
    st.session_state.crosswalk = crosswalk
    st.session_state.np_benchmark = np_benchmark
    st.session_state.feat_imp = feat_imp

    # ── Sidebar ───────────────────────────────────────────────────
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Home", "Self-Assessment", "Gap Analysis", "Learning Roadmap",
         "Occupation Explorer", "ML Insights", "About"]
    )

    if page == "Home":
        render_home(df, jz, clusters)
    elif page == "Self-Assessment":
        render_self_assessment(crosswalk)
    elif page == "Gap Analysis":
        render_gap_analysis(crosswalk, np_benchmark, feat_imp, df)
    elif page == "Learning Roadmap":
        render_learning_roadmap(crosswalk, np_benchmark, feat_imp)
    elif page == "Occupation Explorer":
        render_occupation_explorer(df, jz, clusters)
    elif page == "ML Insights":
        render_ml_insights(feat_imp, model_comp, clusters, df)
    elif page == "About":
        render_about()


# ── Page: Home ────────────────────────────────────────────────────
def render_home(df, jz, clusters):
    st.title("AL Essentials Competency Self-Assessment Dashboard")
    st.markdown(
        "An interactive tool integrating **O*NET 30.1 workforce data** with "
        "**AACN 2021 Advanced-Level Essentials** to support DNP student "
        "competency development."
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Healthcare Occupations", df['soc_code'].nunique())
    with col2:
        st.metric("Competency Dimensions", df['element_name'].nunique())
    with col3:
        st.metric("AACN Competencies", "45")
    with col4:
        st.metric("AACN Domains", "10")

    st.markdown("---")
    st.subheader("How to Use This Dashboard")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("**Step 1: Self-Assess**")
        st.markdown(
            "Rate your competency level on each of the 45 AACN "
            "Advanced-Level Essentials using the Dreyfus/Benner scale "
            "(Novice to Expert)."
        )
    with col_b:
        st.markdown("**Step 2: Analyze Gaps**")
        st.markdown(
            "View your competency gaps weighted by machine learning "
            "feature importance derived from O*NET workforce data."
        )
    with col_c:
        st.markdown("**Step 3: Plan Development**")
        st.markdown(
            "Receive a personalized learning roadmap prioritizing "
            "the competencies with the highest weighted gaps."
        )

    st.markdown("---")
    st.caption(
        "Data Source: O*NET 30.1 (December 2025), CC-BY 4.0 License. "
        "National Center for O*NET Development. U.S. Department of Labor."
    )


# ── Page: Self-Assessment ─────────────────────────────────────────
def render_self_assessment(crosswalk):
    st.title("Competency Self-Assessment")
    st.markdown(
        "Rate your current competency level for each AACN 2021 "
        "Advanced-Level Essential using the **Dreyfus/Benner** scale."
    )

    # Scale legend
    with st.expander("Rating Scale Guide", expanded=False):
        for level, label in DREYFUS_SCALE.items():
            descriptions = {
                1: "Rule-governed behavior, limited situational perception",
                2: "Recognizes meaningful aspects of situations from experience",
                3: "Conscious, deliberate planning; sees actions in terms of goals",
                4: "Perceives situations as wholes; guided by maxims",
                5: "Intuitive grasp of situations; fluid, flexible performance",
            }
            st.markdown(f"**{level} - {label}**: {descriptions[level]}")

    # Initialize session state for assessments
    if 'self_assessment' not in st.session_state:
        st.session_state.self_assessment = {}

    # Group competencies by AACN domain
    domains = {}
    for comp_id, comp_info in crosswalk.items():
        d = comp_id.split('.')[0]
        if d not in domains:
            domains[d] = []
        domains[d].append((comp_id, comp_info))

    # Quick-fill and import options
    col1, col2, col3 = st.columns([2, 1, 1])
    with col2:
        if st.button("Set All to Competent (3)"):
            for comp_id in crosswalk:
                st.session_state.self_assessment[comp_id] = 3
            st.rerun()
    with col3:
        uploaded = st.file_uploader(
            "Import Previous Assessment", type=['csv'],
            help="Upload a CSV exported from a prior session to restore your scores.",
            key="import_csv"
        )
        if uploaded is not None:
            imported = parse_imported_csv(uploaded)
            if imported:
                st.session_state.self_assessment = imported
                st.success(f"Imported {len(imported)} scores from previous assessment.")
                st.rerun()
            else:
                st.error("Could not parse the uploaded file. Is it from this dashboard?")

    # Render by domain
    for d_num in sorted(domains.keys(), key=int):
        d_name = AACN_DOMAIN_NAMES.get(d_num, f"Domain {d_num}")
        with st.expander(f"Domain {d_num}: {d_name}", expanded=False):
            for comp_id, comp_info in domains[d_num]:
                current = st.session_state.self_assessment.get(comp_id, 3)
                val = st.slider(
                    f"**{comp_id}**: {comp_info['competency']}",
                    min_value=1, max_value=5, value=current,
                    format="%d",
                    help=f"Mapping strength: {comp_info['mapping_strength']}",
                    key=f"slider_{comp_id}"
                )
                st.session_state.self_assessment[comp_id] = val

    # Summary and export
    if st.session_state.self_assessment:
        st.markdown("---")
        st.subheader("Assessment Summary")
        scores = list(st.session_state.self_assessment.values())
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Mean Score", f"{np.mean(scores):.1f}")
        with col2:
            st.metric("Competencies Rated", len(scores))
        with col3:
            st.metric("Below Competent (< 3)", sum(1 for s in scores if s < 3))

        st.info("Navigate to **Gap Analysis** to see your personalized results.")

        # Export options
        st.markdown("---")
        st.subheader("Save Your Assessment")
        exp_col1, exp_col2 = st.columns(2)

        with exp_col1:
            st.markdown("**Download Full Report (CSV)**")
            st.markdown(
                "Save your scores, gaps, and ML importance weights. "
                "You can re-import this file later to pick up where you left off."
            )
            # Need crosswalk, np_benchmark, feat_imp from parent scope
            # They are loaded globally in main(), pass via session state
            if 'crosswalk' in st.session_state and 'np_benchmark' in st.session_state:
                csv_df = export_assessment_to_csv(
                    st.session_state.self_assessment,
                    st.session_state.crosswalk,
                    st.session_state.np_benchmark,
                    st.session_state.feat_imp
                )
                csv_buffer = csv_df.to_csv(index=False)
                st.download_button(
                    "Download My Assessment (CSV)",
                    data=csv_buffer,
                    file_name=f"competency_assessment_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv"
                )

        with exp_col2:
            st.markdown("**Submit to Instructor**")
            st.markdown(
                "Generate a summary row for your instructor to collect "
                "across all students."
            )
            student_name = st.text_input("Your Name (for submission)")
            if student_name and 'crosswalk' in st.session_state:
                summary_df = export_for_instructor(
                    st.session_state.self_assessment,
                    st.session_state.crosswalk,
                    st.session_state.np_benchmark,
                    st.session_state.feat_imp,
                    student_name=student_name
                )
                csv_summary = summary_df.to_csv(index=False)
                st.download_button(
                    "Download Instructor Summary (CSV)",
                    data=csv_summary,
                    file_name=f"instructor_summary_{student_name.replace(' ','_')}_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )


# ── Page: Gap Analysis ────────────────────────────────────────────
def render_gap_analysis(crosswalk, np_benchmark, feat_imp, df):
    st.title("ML-Weighted Gap Analysis")

    if not st.session_state.get('self_assessment'):
        st.warning("Please complete the Self-Assessment first.")
        return

    gaps_df = compute_gap_scores(
        st.session_state.self_assessment, crosswalk, np_benchmark, feat_imp
    )

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Avg Gap", f"{gaps_df['gap'].mean():.3f}")
    with col2:
        st.metric("Largest Gap", f"{gaps_df['gap'].max():.3f}")
    with col3:
        positive_gaps = gaps_df[gaps_df['gap'] > 0]
        st.metric("Competencies with Gaps", len(positive_gaps))
    with col4:
        st.metric("Avg Weighted Gap", f"{gaps_df['weighted_gap'].mean():.3f}")

    tab1, tab2, tab3 = st.tabs(["Domain Radar", "Gap Priority Matrix", "Detail Table"])

    with tab1:
        st.subheader("Domain-Level Gap Analysis")
        domain_gaps = gaps_df.groupby('aacn_domain_name').agg({
            'student_norm': 'mean',
            'benchmark': 'mean',
            'gap': 'mean',
        }).reset_index()

        fig = go.Figure()
        categories = domain_gaps['aacn_domain_name'].tolist()
        categories_wrap = [c.replace(' ', '\n') if len(c) > 20 else c for c in categories]

        student_vals = domain_gaps['student_norm'].tolist() + [domain_gaps['student_norm'].iloc[0]]
        bench_vals = domain_gaps['benchmark'].tolist() + [domain_gaps['benchmark'].iloc[0]]

        fig.add_trace(go.Scatterpolar(
            r=bench_vals, theta=categories + [categories[0]],
            fill='toself', name='NP Benchmark (O*NET)',
            line=dict(color=COLORS['primary'], width=2),
            fillcolor='rgba(27,42,74,0.1)'
        ))
        fig.add_trace(go.Scatterpolar(
            r=student_vals, theta=categories + [categories[0]],
            fill='toself', name='Your Self-Assessment',
            line=dict(color=COLORS['accent'], width=2),
            fillcolor='rgba(224,122,95,0.1)'
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            height=550, showlegend=True,
            legend=dict(orientation='h', y=-0.15)
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Gap Priority Matrix")
        st.markdown("Competencies in the **upper-right** quadrant are the highest priority.")

        fig = px.scatter(
            gaps_df[gaps_df['gap'] > 0],
            x='gap', y='ml_importance',
            size='weighted_gap',
            color='aacn_domain_name',
            hover_data=['competency_id', 'competency', 'student_score'],
            labels={
                'gap': 'Gap Magnitude (Benchmark - Self)',
                'ml_importance': 'ML Feature Importance',
                'aacn_domain_name': 'AACN Domain',
            },
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Detailed Gap Table")
        display_df = gaps_df.sort_values('weighted_gap', ascending=False)
        display_df = display_df[[
            'competency_id', 'competency', 'aacn_domain_name',
            'student_score', 'benchmark', 'gap', 'ml_importance',
            'weighted_gap', 'mapping_strength'
        ]].copy()
        display_df.columns = [
            'ID', 'Competency', 'Domain', 'Your Score', 'Benchmark',
            'Gap', 'ML Importance', 'Weighted Gap', 'Mapping Strength'
        ]
        for col in ['Benchmark', 'Gap', 'ML Importance', 'Weighted Gap']:
            display_df[col] = display_df[col].round(4)
        st.dataframe(display_df, use_container_width=True, height=600)


# ── Page: Learning Roadmap ────────────────────────────────────────
def render_learning_roadmap(crosswalk, np_benchmark, feat_imp):
    st.title("Personalized Learning Roadmap")

    if not st.session_state.get('self_assessment'):
        st.warning("Please complete the Self-Assessment first.")
        return

    gaps_df = compute_gap_scores(
        st.session_state.self_assessment, crosswalk, np_benchmark, feat_imp
    )

    # Sort by weighted gap descending
    priority = gaps_df[gaps_df['gap'] > 0].sort_values('weighted_gap', ascending=False)

    if priority.empty:
        st.success("No competency gaps detected. Your self-assessment meets or exceeds all benchmarks.")
        return

    st.subheader(f"Top Priority Competencies ({len(priority)} gaps identified)")

    # Simulation: achievement probability
    st.markdown("---")
    st.subheader("Development Simulation")
    effort_level = st.select_slider(
        "Weekly Development Hours",
        options=[2, 4, 6, 8, 10, 12, 15, 20],
        value=6
    )
    timeline_weeks = st.slider("Timeline (weeks)", 8, 52, 26)

    # Simple simulation: more hours = faster gap closure
    base_rate = effort_level / 100  # gap closure per week
    total_closure = min(base_rate * timeline_weeks, 1.0)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Projected Gap Closure", f"{total_closure:.0%}")
    with col2:
        closable = sum(1 for _, row in priority.iterrows()
                       if row['gap'] * (1 - total_closure) < 0.05)
        st.metric("Competencies Resolved", f"{closable}/{len(priority)}")
    with col3:
        achievement_prob = min(0.95, total_closure * 0.85 + 0.1)
        st.metric("Achievement Probability", f"{achievement_prob:.0%}")

    # Before/after radar
    st.markdown("---")
    st.subheader("Projected Improvement")

    domain_current = gaps_df.groupby('aacn_domain_name')['student_norm'].mean()
    projected_scores = {}
    for d_name in domain_current.index:
        d_gaps = gaps_df[gaps_df['aacn_domain_name'] == d_name]
        avg_gap = d_gaps['gap'].mean()
        improvement = avg_gap * total_closure
        projected_scores[d_name] = min(1.0, domain_current[d_name] + improvement)

    fig = go.Figure()
    categories = list(domain_current.index)
    bench = gaps_df.groupby('aacn_domain_name')['benchmark'].mean()

    fig.add_trace(go.Scatterpolar(
        r=[bench.get(c, 0) for c in categories] + [bench.get(categories[0], 0)],
        theta=categories + [categories[0]],
        name='NP Benchmark', line=dict(color=COLORS['primary'], width=2, dash='dash'),
    ))
    fig.add_trace(go.Scatterpolar(
        r=[domain_current.get(c, 0) for c in categories] + [domain_current.get(categories[0], 0)],
        theta=categories + [categories[0]],
        name='Current', line=dict(color=COLORS['accent'], width=2),
        fill='toself', fillcolor='rgba(224,122,95,0.1)'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[projected_scores.get(c, 0) for c in categories] + [projected_scores.get(categories[0], 0)],
        theta=categories + [categories[0]],
        name='Projected', line=dict(color=COLORS['secondary'], width=2),
        fill='toself', fillcolor='rgba(42,157,143,0.1)'
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        height=500, showlegend=True,
        legend=dict(orientation='h', y=-0.15)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Priority list with recommendations
    st.markdown("---")
    st.subheader("Recommended Development Activities")

    # Learning activity mapping
    activity_map = {
        "1": ["Clinical case analysis", "Evidence-based practice projects", "Literature synthesis"],
        "2": ["Standardized patient simulations", "Motivational interviewing practice", "Cultural competency workshops"],
        "3": ["Community health assessment", "Epidemiology coursework", "Health equity projects"],
        "4": ["Systematic review participation", "Quality improvement project", "Manuscript preparation"],
        "5": ["Root cause analysis exercises", "Patient safety simulations", "Quality metrics dashboards"],
        "6": ["Interprofessional team rounds", "Collaborative practice agreements", "Team-based care projects"],
        "7": ["Health policy analysis", "Systems thinking exercises", "Organizational assessment"],
        "8": ["Health informatics coursework", "EHR optimization projects", "Data analytics workshops"],
        "9": ["Ethics case studies", "Professional portfolio development", "Leadership reflection journals"],
        "10": ["Executive coaching", "Mentorship programs", "Strategic planning exercises"],
    }

    for i, (_, row) in enumerate(priority.head(10).iterrows()):
        d_num = row['aacn_domain']
        activities = activity_map.get(d_num, ["General coursework"])
        with st.expander(
            f"Priority {i+1}: {row['competency_id']} - {row['competency'][:60]}... "
            f"(Gap: {row['gap']:.3f}, Weight: {row['weighted_gap']:.3f})"
        ):
            st.markdown(f"**Domain**: {row['aacn_domain_name']}")
            st.markdown(f"**Your Score**: {row['student_score']} ({DREYFUS_SCALE[row['student_score']]})")
            st.markdown(f"**Benchmark**: {row['benchmark']:.3f}")
            st.markdown(f"**ML Importance**: {row['ml_importance']:.4f}")
            st.markdown("**Recommended Activities:**")
            for act in activities:
                st.markdown(f"- {act}")


# ── Page: Occupation Explorer ─────────────────────────────────────
def render_occupation_explorer(df, jz, clusters):
    st.title("Occupation Explorer")

    tab1, tab2, tab3 = st.tabs(["PCA Landscape", "Occupation Comparison", "Cluster Analysis"])

    with tab1:
        st.subheader("Healthcare Occupation Landscape (PCA)")
        matrix, dim_cols = build_occupation_matrix(df)

        scaler = StandardScaler()
        X = scaler.fit_transform(matrix.values)
        pca = PCA(n_components=3)
        X_pca = pca.fit_transform(X)

        pca_df = pd.DataFrame({
            'PC1': X_pca[:, 0], 'PC2': X_pca[:, 1], 'PC3': X_pca[:, 2],
            'soc_code': matrix.index
        })

        titles_map = dict(zip(df['soc_code'], df['title']))
        targets_map = dict(zip(df['soc_code'], df['is_target']))

        pca_df['title'] = pca_df['soc_code'].map(titles_map)
        pca_df['is_target'] = pca_df['soc_code'].map(targets_map)
        pca_df = pca_df.merge(jz, on='soc_code', how='left')

        if clusters is not None:
            cluster_map = dict(zip(clusters['soc_code'], clusters['cluster']))
            pca_df['cluster'] = pca_df['soc_code'].map(cluster_map).fillna(-1).astype(int)
            pca_df['Cluster'] = pca_df['cluster'].astype(str)
        else:
            pca_df['Cluster'] = '0'

        pca_df['Type'] = pca_df['is_target'].map({1: 'Target', 0: 'Other'})

        view_mode = st.radio("Color by", ['Cluster', 'Job Zone', 'Target Status'], horizontal=True)

        if view_mode == 'Cluster':
            fig = px.scatter(pca_df, x='PC1', y='PC2', color='Cluster',
                             hover_data=['title', 'job_zone'],
                             symbol='Type', symbol_map={'Target': 'star', 'Other': 'circle'})
        elif view_mode == 'Job Zone':
            pca_df['Job Zone'] = pca_df['job_zone'].astype(str)
            fig = px.scatter(pca_df, x='PC1', y='PC2', color='Job Zone',
                             hover_data=['title'],
                             symbol='Type', symbol_map={'Target': 'star', 'Other': 'circle'})
        else:
            fig = px.scatter(pca_df, x='PC1', y='PC2', color='Type',
                             hover_data=['title', 'job_zone'],
                             color_discrete_map={'Target': 'red', 'Other': COLORS['primary']})

        fig.update_layout(
            height=550,
            xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)",
            yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)",
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Occupation Comparison")
        col1, col2 = st.columns(2)
        with col1:
            occ1 = st.selectbox("Occupation 1", TARGET_CODES,
                                format_func=lambda x: OCC_LABELS.get(x, x), key='occ1')
        with col2:
            other_codes = [c for c in TARGET_CODES if c != occ1]
            occ2 = st.selectbox("Occupation 2", other_codes,
                                format_func=lambda x: OCC_LABELS.get(x, x), key='occ2')

        occ1_data = df[df['soc_code'] == occ1].groupby('domain')['normalized_value'].mean()
        occ2_data = df[df['soc_code'] == occ2].groupby('domain')['normalized_value'].mean()

        fig = go.Figure()
        cats = [DOMAIN_LABELS[d] for d in DOMAIN_ORDER]

        for occ_code, occ_data, color, name in [
            (occ1, occ1_data, COLORS['primary'], OCC_LABELS[occ1]),
            (occ2, occ2_data, COLORS['accent'], OCC_LABELS[occ2])
        ]:
            vals = [occ_data.get(d, 0) for d in DOMAIN_ORDER] + [occ_data.get(DOMAIN_ORDER[0], 0)]
            fig.add_trace(go.Scatterpolar(
                r=vals, theta=cats + [cats[0]], fill='toself',
                name=name, line=dict(color=color, width=2)
            ))

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            height=500, showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("K-Means Cluster Analysis")
        if clusters is not None:
            cluster_summary = clusters.groupby('cluster').agg({
                'soc_code': 'count',
                'job_zone': lambda x: x.mode().iloc[0] if not x.isna().all() else 'N/A',
                'is_target': 'sum'
            }).reset_index()
            cluster_summary.columns = ['Cluster', 'Occupations', 'Job Zone Mode', 'Target Count']
            st.dataframe(cluster_summary, use_container_width=True)

            fig = px.histogram(clusters, x='cluster', color='cluster',
                               title='Cluster Size Distribution')
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Cluster data not available. Run Week 4 ML script first.")


# ── Page: ML Insights ─────────────────────────────────────────────
def render_ml_insights(feat_imp, model_comp, clusters, df):
    st.title("Machine Learning Insights")

    tab1, tab2, tab3 = st.tabs(["Feature Importance", "Model Comparison", "Cluster Profiles"])

    with tab1:
        st.subheader("Random Forest Feature Importance")
        if feat_imp is not None:
            top_n = st.slider("Show top N features", 10, 50, 20)
            top = feat_imp.head(top_n).copy()
            top['label'] = top['dimension'] + ' (' + top['domain'].map(DOMAIN_LABELS) + ')'

            fig = px.bar(top, x='importance', y='label', orientation='h',
                         color='domain', color_discrete_map={
                             'knowledge': '#264653', 'skills': '#2a9d8f',
                             'abilities': '#e9c46a', 'work_activities': '#f4a261',
                             'work_styles': '#e76f51', 'work_context': '#606c38',
                         })
            fig.update_layout(height=max(400, top_n * 25), yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)

            # Domain-level aggregation
            st.subheader("Domain-Level Importance")
            domain_imp = feat_imp.groupby('domain')['importance'].sum().sort_values(ascending=False)
            domain_imp.index = domain_imp.index.map(DOMAIN_LABELS)
            fig2 = px.bar(x=domain_imp.index, y=domain_imp.values,
                          labels={'x': 'Domain', 'y': 'Cumulative Importance'})
            fig2.update_layout(height=350)
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Feature importance data not available.")

    with tab2:
        st.subheader("6-Model Comparison")
        if model_comp is not None:
            st.dataframe(model_comp, use_container_width=True)

            fig = px.bar(model_comp, x='Model', y='CV Mean',
                         error_y='CV Std',
                         title='Cross-Validated Accuracy by Model',
                         color='CV Mean', color_continuous_scale='teal')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Model comparison data not available.")

    with tab3:
        st.subheader("Cluster Competency Profiles")
        if clusters is not None:
            matrix, dim_cols = build_occupation_matrix(df)
            merged = pd.DataFrame({'soc_code': matrix.index}).merge(
                clusters[['soc_code', 'cluster']], on='soc_code', how='inner'
            )

            profiles = []
            for c in sorted(merged['cluster'].unique()):
                c_codes = merged[merged['cluster'] == c]['soc_code']
                c_data = df[df['soc_code'].isin(c_codes)]
                for d in DOMAIN_ORDER:
                    d_mean = c_data[c_data['domain'] == d]['normalized_value'].mean()
                    profiles.append({
                        'Cluster': str(c), 'Domain': DOMAIN_LABELS[d], 'Score': d_mean
                    })

            prof_df = pd.DataFrame(profiles)
            fig = px.line_polar(prof_df, r='Score', theta='Domain', color='Cluster',
                                line_close=True)
            fig.update_layout(
                polar=dict(radialaxis=dict(range=[0, 1])),
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Cluster data not available.")


# ── Page: About ───────────────────────────────────────────────────
def render_about():
    st.title("About This Dashboard")
    st.markdown("""
**AL Essentials Competency Self-Assessment and Personalized Learning Roadmap Dashboard**

This application was developed as part of the DSIM 608: Applied Managerial Analytics
capstone project at Jacksonville University.

**Purpose**: Enable DNP students to self-assess competency levels across
AACN 2021 Advanced-Level Essentials and receive ML-weighted, personalized
learning roadmaps based on O*NET 30.1 workforce data.

**Methods**:
- K-Means clustering to discover natural occupation groups
- Random Forest classification to identify most predictive competency dimensions
- 6-model comparison (RF, SVM, GBM, K-NN, MLP, Logistic Regression)
- Student-developed crosswalk mapping 45 AACN competencies to O*NET dimensions
- Gap analysis weighted by ML feature importance

**Data**: O*NET 30.1 (December 2025), CC-BY 4.0 License.
National Center for O*NET Development. U.S. Department of Labor,
Employment and Training Administration.

**Developer**: Roberta Christopher, EdD, MSN, APRN, FNP-BC, NE-BC, EBP-C, CAIF

**Instructor**: Dr. Zachary Davis, Jacksonville University
    """)


if __name__ == "__main__":
    main()
