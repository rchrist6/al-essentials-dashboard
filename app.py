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
from fpdf import FPDF

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

MILLERS_SCALE = {
    1: "Knows",
    2: "Knows How",
    3: "Shows How",
    4: "Does",
    5: "Teaches/Leads",
}

SCALE_OPTIONS = {
    "Dreyfus/Benner": DREYFUS_SCALE,
    "Miller's Pyramid": MILLERS_SCALE,
}


def get_level_label(score):
    """Get the label for a score using the currently active scale."""
    scale_name = st.session_state.get('active_scale_name', 'Dreyfus/Benner')
    scale = SCALE_OPTIONS.get(scale_name, DREYFUS_SCALE)
    return scale.get(int(score), "")

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
         "My Report", "Faculty Analytics", "Priority & Simulation", "About"]
    )

    if page == "Home":
        render_home(df, jz, clusters)
    elif page == "Self-Assessment":
        render_self_assessment(crosswalk)
    elif page == "Gap Analysis":
        render_gap_analysis(crosswalk, np_benchmark, feat_imp, df)
    elif page == "Learning Roadmap":
        render_learning_roadmap(crosswalk, np_benchmark, feat_imp)
    elif page == "My Report":
        render_my_report(crosswalk, np_benchmark, feat_imp, df)
    elif page == "Faculty Analytics":
        render_ml_insights(feat_imp, model_comp, clusters, df)
    elif page == "Priority & Simulation":
        render_priority_simulation(df, feat_imp)
    elif page == "About":
        render_about()


# ── Page: Home ────────────────────────────────────────────────────
def render_home(df, jz, clusters):
    st.title("AL Essentials Competency Self-Assessment Dashboard")
    st.markdown(
        "An interactive tool for DNP students to assess their competency "
        "development across the **AACN 2026 Advanced-Level Essentials** "
        "and receive a personalized, data-driven learning roadmap."
    )

    st.markdown("---")

    # What is this tool?
    st.subheader("What Is This Tool?")
    st.markdown(
        "The AACN (American Association of Colleges of Nursing) published "
        "the 2026 Essentials framework to define what every nursing graduate "
        "should know and be able to do. At the Advanced Level (DNP), there are "
        "**10 domains** and **45 competencies** that describe the full scope "
        "of practice for nurse practitioners, clinical nurse specialists, "
        "nurse anesthetists, nurse midwives, and other advanced practice roles."
    )
    st.markdown(
        "This dashboard lets you **rate yourself** on each of those 45 "
        "competencies, then compares your self-assessment against real "
        "workforce data from the U.S. Department of Labor (O*NET). Machine "
        "learning identifies which competencies matter most for advanced "
        "practice, so your results are weighted by actual workforce importance, "
        "not just equal treatment of all 45 items."
    )

    st.markdown("---")

    # The 10 Domains
    st.subheader("The 10 AACN Essentials Domains")
    st.markdown(
        "Each domain represents a broad area of nursing competence. "
        "The self-assessment asks you to rate specific competencies within "
        "each domain."
    )

    domain_descriptions = {
        "1": ("Knowledge for Nursing Practice",
              "The scientific foundation of nursing: biology, pharmacology, "
              "pathophysiology, and the ability to apply theory and research "
              "to clinical decisions."),
        "2": ("Person-Centered Care",
              "Providing holistic, individualized care: assessment, diagnosis, "
              "treatment planning, care coordination, and health promotion "
              "tailored to each patient."),
        "3": ("Population Health",
              "Addressing health at the community and population level: "
              "epidemiology, access to healthcare resources, disaster "
              "preparedness, and advocacy for health policy."),
        "4": ("Scholarship for Nursing Practice",
              "Generating and applying evidence: conducting research, "
              "integrating best evidence, disseminating findings, and "
              "advancing the science of nursing."),
        "5": ("Quality and Safety",
              "Ensuring safe, high-quality care: quality improvement methods, "
              "patient safety culture, and system-level safety initiatives."),
        "6": ("Interprofessional Partnerships",
              "Working effectively across disciplines: team communication, "
              "collaborative decision-making, and shared accountability "
              "with physicians, pharmacists, social workers, and others."),
        "7": ("Systems-Based Practice",
              "Understanding and improving healthcare systems: cost-effectiveness, "
              "systems thinking, innovation, and organizational leadership."),
        "8": ("Informatics and Healthcare Technologies",
              "Using technology to improve care: electronic health records, "
              "data analytics, telehealth, and information security."),
        "9": ("Professionalism",
              "Embodying nursing values: ethical practice, accountability, "
              "professional identity, regulatory compliance, and commitment "
              "to access, connection, and engagement."),
        "10": ("Personal, Professional Development and Leadership",
               "Lifelong growth and leading others: self-reflection, "
               "mentorship, resilience, and developing leadership capacity "
               "to transform healthcare."),
    }

    col_left, col_right = st.columns(2)
    for i, (d_num, (d_name, d_desc)) in enumerate(domain_descriptions.items()):
        col = col_left if i % 2 == 0 else col_right
        with col:
            st.markdown(f"**Domain {d_num}: {d_name}**")
            st.markdown(f"{d_desc}")
            st.markdown("")

    st.markdown("---")

    # How it works
    st.subheader("How It Works")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("**Step 1: Self-Assess**")
        st.markdown(
            "Go to the **Self-Assessment** page and rate your current "
            "competency level (1 = Novice through 5 = Expert) on each of "
            "the 45 competencies. Be honest; this is for your own development."
        )
    with col_b:
        st.markdown("**Step 2: Review Your Results**")
        st.markdown(
            "The **Gap Analysis** page shows where you stand relative to "
            "the Nurse Practitioner workforce benchmark, weighted by which "
            "competencies matter most according to machine learning."
        )
    with col_c:
        st.markdown("**Step 3: Plan Your Growth**")
        st.markdown(
            "The **Learning Roadmap** and **My Report** pages give you "
            "a prioritized list of development activities and a downloadable "
            "narrative report you can share with your advisor."
        )

    st.markdown("---")

    # Rating scale explanation
    st.subheader("Rating Scales")
    st.markdown(
        "You can choose between two well-established competency frameworks "
        "when completing the self-assessment. Both use a 1-5 scale."
    )
    tab_benner, tab_miller = st.tabs(["Dreyfus/Benner", "Miller's Pyramid"])
    with tab_benner:
        st.markdown(
            "Patricia Benner's adaptation of the Dreyfus model describes how "
            "clinicians progress from rule-following to intuitive expertise:"
        )
        st.dataframe(pd.DataFrame({
            'Level': [1, 2, 3, 4, 5],
            'Label': ['Novice', 'Advanced Beginner', 'Competent', 'Proficient', 'Expert'],
            'Description': [
                'New to this area; relies on rules and guidelines',
                'Recognizes meaningful patterns from experience',
                'Deliberate planning; sees actions in terms of goals',
                'Perceives situations holistically; deep experience guides decisions',
                'Intuitive grasp; fluid, flexible, highly skilled performance',
            ]
        }), use_container_width=True, hide_index=True)
    with tab_miller:
        st.markdown(
            "George Miller's pyramid of clinical competence focuses on "
            "what learners can demonstrate at each stage, from foundational "
            "knowledge through independent practice and teaching:"
        )
        st.dataframe(pd.DataFrame({
            'Level': [1, 2, 3, 4, 5],
            'Label': ['Knows', 'Knows How', 'Shows How', 'Does', 'Teaches/Leads'],
            'Description': [
                'Can define and explain the concept (foundational knowledge)',
                'Knows how to apply it; can describe the steps and reasoning',
                'Can demonstrate in a supervised or simulated setting',
                'Performs independently in real clinical or professional settings',
                'Can teach, mentor, or lead others in this competency',
            ]
        }), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.caption(
        "Data Source: O*NET 30.1 (December 2025), CC-BY 4.0 License. "
        "National Center for O*NET Development. U.S. Department of Labor. "
        "Dashboard developed at Jacksonville University, Keigwin School of Nursing."
    )


# ── Page: Self-Assessment ─────────────────────────────────────────
def render_self_assessment(crosswalk):
    st.title("Competency Self-Assessment")
    st.markdown(
        "Rate your current competency level for each of the 45 AACN 2026 "
        "Advanced-Level Essential competencies. Choose your preferred "
        "rating framework below."
    )

    # Scale selection
    scale_choice = st.radio(
        "Rating Framework",
        list(SCALE_OPTIONS.keys()),
        horizontal=True,
        help="Choose the framework that feels most familiar to you."
    )
    active_scale = SCALE_OPTIONS[scale_choice]
    st.session_state['active_scale_name'] = scale_choice

    # Scale legend
    dreyfus_descriptions = {
        1: "New to this area; relies on rules and guidelines; limited situational awareness",
        2: "Recognizes meaningful patterns from experience; beginning to see the bigger picture",
        3: "Deliberate planning and prioritization; sees actions in terms of long-range goals",
        4: "Perceives situations holistically; draws on deep experience to guide decisions",
        5: "Intuitive grasp of situations; fluid, flexible, highly skilled performance",
    }
    millers_descriptions = {
        1: "I can define and explain this concept (foundational knowledge)",
        2: "I know how to apply this in practice and can describe the steps",
        3: "I can demonstrate this competency in a supervised or simulated setting",
        4: "I perform this independently in real clinical or professional settings",
        5: "I can teach, mentor, or lead others in this competency",
    }
    descriptions = millers_descriptions if "Miller" in scale_choice else dreyfus_descriptions

    with st.expander("Rating Scale Guide", expanded=True):
        for level, label in active_scale.items():
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
    with col1:
        uploaded_restore = st.file_uploader(
            "Restore a Previous Assessment", type=['csv'],
            help="Upload a CSV from a prior session to restore your scores.",
            key="import_csv"
        )
        if uploaded_restore is not None:
            imported = parse_imported_csv(uploaded_restore)
            if imported:
                st.session_state.self_assessment = imported
                st.success(f"Imported {len(imported)} scores.")
                st.rerun()
            else:
                st.error("Could not parse the file.")
    with col3:
        uploaded_compare = st.file_uploader(
            "Upload Baseline for Progress Tracking", type=['csv'],
            help="Upload an EARLIER assessment to compare against your current scores.",
            key="import_baseline"
        )
        if uploaded_compare is not None:
            baseline = parse_imported_csv(uploaded_compare)
            if baseline:
                st.session_state.previous_assessment = baseline
                st.success(
                    f"Baseline loaded ({len(baseline)} scores). "
                    f"Complete your new assessment, then check Gap Analysis "
                    f"to see your progress."
                )
            else:
                st.error("Could not parse the baseline file.")

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
            st.markdown("**Submit My Assessment**")
            st.markdown(
                "Enter your name and click Submit to save your assessment "
                "to the program's records for tracking your development."
            )
            student_name = st.text_input("Your Name (required for submission)")
            if student_name and 'crosswalk' in st.session_state:
                summary_df = export_for_instructor(
                    st.session_state.self_assessment,
                    st.session_state.crosswalk,
                    st.session_state.np_benchmark,
                    st.session_state.feat_imp,
                    student_name=student_name
                )

                # Google Sheets submission
                SHEETS_URL = st.secrets.get("GOOGLE_SHEETS_URL", "") if hasattr(st, 'secrets') else ""
                try:
                    SHEETS_URL = st.secrets.get("GOOGLE_SHEETS_URL", "")
                except Exception:
                    SHEETS_URL = ""

                if SHEETS_URL:
                    if st.button("Submit Assessment", type="primary"):
                        import requests
                        try:
                            payload = summary_df.iloc[0].to_dict()
                            # Convert any non-serializable values
                            for k, v in payload.items():
                                if pd.isna(v):
                                    payload[k] = ""
                                elif hasattr(v, 'item'):
                                    payload[k] = v.item()
                            resp = requests.post(SHEETS_URL, json=payload, timeout=10)
                            if resp.status_code == 200:
                                st.success(
                                    f"Assessment submitted successfully for {student_name}."
                                )
                            else:
                                st.error("Submission failed. Please download the CSV instead.")
                        except Exception as e:
                            st.error(f"Could not connect to the collection sheet. "
                                     f"Please download the CSV instead.")

                # Always offer CSV download as fallback
                csv_summary = summary_df.to_csv(index=False)
                st.download_button(
                    "Download Summary (CSV)",
                    data=csv_summary,
                    file_name=f"instructor_summary_{student_name.replace(' ','_')}_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )


# ── Page: Gap Analysis ────────────────────────────────────────────
def render_gap_analysis(crosswalk, np_benchmark, feat_imp, df):
    st.title("Competency Gap Analysis")

    if not st.session_state.get('self_assessment'):
        st.warning("Please complete the Self-Assessment first.")
        return

    # Explanation for students
    with st.expander("How does this analysis work?", expanded=False):
        st.markdown(
            "This page compares your self-assessment scores against the "
            "**Nurse Practitioner workforce benchmark** derived from O*NET, "
            "the U.S. Department of Labor's occupational database.\n\n"
            "Not all gaps are equally important. Machine learning analysis of "
            "237 workforce competency dimensions identified which ones best "
            "distinguish advanced practice roles from other healthcare "
            "occupations. Competencies linked to those high-importance "
            "dimensions are **weighted more heavily** in your results.\n\n"
            "In plain terms: a gap in *Complex Problem Solving* (the #1 "
            "predictor of advanced practice) matters more than a gap in a "
            "dimension that doesn't differentiate practice levels."
        )

    gaps_df = compute_gap_scores(
        st.session_state.self_assessment, crosswalk, np_benchmark, feat_imp
    )

    # Add importance rank for student-friendly labels
    imp_sorted = gaps_df.sort_values('ml_importance', ascending=False).reset_index(drop=True)
    imp_sorted['importance_rank'] = range(1, len(imp_sorted) + 1)
    rank_map = dict(zip(imp_sorted['competency_id'], imp_sorted['importance_rank']))
    gaps_df['importance_rank'] = gaps_df['competency_id'].map(rank_map)

    def importance_label(rank):
        if rank <= 5:
            return "Very High"
        elif rank <= 15:
            return "High"
        elif rank <= 30:
            return "Moderate"
        else:
            return "Lower"

    gaps_df['importance_level'] = gaps_df['importance_rank'].apply(importance_label)

    # Summary metrics
    positive_gaps = gaps_df[gaps_df['gap'] > 0]
    high_priority = positive_gaps[positive_gaps['importance_rank'] <= 15]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Competencies with Gaps", len(positive_gaps))
    with col2:
        st.metric("High-Priority Gaps", len(high_priority),
                  help="Gaps in competencies ranked in the top 15 by workforce importance")
    with col3:
        st.metric("Strongest Domain",
                  gaps_df.groupby('aacn_domain_name')['student_score'].mean().idxmax()[:20])
    with col4:
        if not positive_gaps.empty:
            biggest = positive_gaps.sort_values('weighted_gap', ascending=False).iloc[0]
            st.metric("Top Priority",
                      f"{biggest['competency_id']}",
                      help=biggest['competency'][:60])
        else:
            st.metric("Top Priority", "None")

    # Progress comparison
    if 'previous_assessment' in st.session_state:
        st.markdown("---")
        st.subheader("Progress Since Last Assessment")
        prev = st.session_state.previous_assessment
        curr = st.session_state.self_assessment
        improved = sum(1 for k in curr if k in prev and curr[k] > prev[k])
        declined = sum(1 for k in curr if k in prev and curr[k] < prev[k])
        unchanged = sum(1 for k in curr if k in prev and curr[k] == prev[k])
        prev_mean = np.mean([v for k, v in prev.items() if k in curr])
        curr_mean = np.mean(list(curr.values()))

        pcol1, pcol2, pcol3, pcol4 = st.columns(4)
        with pcol1:
            delta = curr_mean - prev_mean
            st.metric("Mean Score", f"{curr_mean:.1f}",
                      delta=f"{delta:+.1f}")
        with pcol2:
            st.metric("Improved", improved)
        with pcol3:
            st.metric("Unchanged", unchanged)
        with pcol4:
            st.metric("Declined", declined)

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
        st.markdown(
            "**Workforce Importance** reflects how strongly each competency's "
            "underlying O*NET dimensions distinguish advanced practice roles "
            "from other healthcare occupations."
        )
        display_df = gaps_df.sort_values('weighted_gap', ascending=False)
        display_df = display_df[[
            'competency_id', 'competency', 'aacn_domain_name',
            'student_score', 'gap', 'importance_level',
            'importance_rank', 'mapping_strength'
        ]].copy()
        display_df.columns = [
            'ID', 'Competency', 'Domain', 'Your Score',
            'Gap', 'Workforce Importance', 'Importance Rank',
            'Crosswalk Strength'
        ]
        display_df['Gap'] = display_df['Gap'].round(3)
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
        "2": ["Standardized patient simulations", "Motivational interviewing practice", "Cultural humility workshops"],
        "3": ["Community health assessment", "Epidemiology coursework", "Access to care initiatives"],
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
            st.markdown(f"**Your Score**: {row['student_score']} ({get_level_label(row['student_score'])})")
            st.markdown(f"**Benchmark**: {row['benchmark']:.3f}")
            st.markdown(f"**ML Importance**: {row['ml_importance']:.4f}")
            st.markdown("**Recommended Activities:**")
            for act in activities:
                st.markdown(f"- {act}")


# ── Page: My Report ───────────────────────────────────────────────
def render_my_report(crosswalk, np_benchmark, feat_imp, df):
    st.title("My Competency Report")

    if not st.session_state.get('self_assessment'):
        st.warning("Please complete the Self-Assessment first.")
        return

    assessment = st.session_state.self_assessment
    gaps_df = compute_gap_scores(assessment, crosswalk, np_benchmark, feat_imp)
    scores = list(assessment.values())
    mean_score = np.mean(scores)
    date_str = datetime.now().strftime("%B %d, %Y")

    # ── Build narrative sections ──────────────────────────────────

    # Overall readiness level
    if mean_score >= 4.5:
        readiness = "Expert-Level Readiness"
        readiness_narrative = (
            "Your overall self-assessment indicates expert-level competency "
            "across the AACN Advanced-Level Essentials. You demonstrate strong "
            "confidence in your ability to perform at an advanced practice level."
        )
    elif mean_score >= 3.5:
        readiness = "Proficient Readiness"
        readiness_narrative = (
            "Your overall self-assessment indicates proficient competency "
            "across the AACN Essentials. You have a solid foundation with "
            "targeted areas for continued growth."
        )
    elif mean_score >= 2.5:
        readiness = "Competent, Developing"
        readiness_narrative = (
            "Your self-assessment indicates competent performance with meaningful "
            "opportunities for development. Focused attention on priority gaps "
            "will strengthen your readiness for advanced practice."
        )
    else:
        readiness = "Building Foundations"
        readiness_narrative = (
            "Your self-assessment identifies substantial growth opportunities. "
            "This is normal early in a DNP program. The roadmap below will "
            "help you prioritize your development."
        )

    # Identify strengths (score >= 4) and gaps (positive gap value)
    strengths = gaps_df[gaps_df['student_score'] >= 4].sort_values(
        'student_score', ascending=False
    )
    strength_domains = strengths.groupby('aacn_domain_name').size().sort_values(
        ascending=False
    )

    gaps_positive = gaps_df[gaps_df['gap'] > 0].sort_values(
        'weighted_gap', ascending=False
    )
    gap_domains = gaps_positive.groupby('aacn_domain_name').size().sort_values(
        ascending=False
    )

    # Domain-level summary
    domain_summary = gaps_df.groupby('aacn_domain_name').agg({
        'student_score': 'mean',
        'gap': 'mean',
        'weighted_gap': 'mean',
    }).round(2)
    domain_summary = domain_summary.sort_values('student_score', ascending=False)

    # Learning activity mapping
    activity_map = {
        "1": ["Clinical case analysis", "Evidence-based practice projects",
              "Literature synthesis seminars"],
        "2": ["Standardized patient simulations", "Motivational interviewing practice",
              "Cultural humility workshops"],
        "3": ["Community health needs assessment", "Epidemiology coursework",
              "Access to care initiative participation"],
        "4": ["Systematic review participation", "Quality improvement project leadership",
              "Manuscript preparation and submission"],
        "5": ["Root cause analysis exercises", "Patient safety simulations",
              "Quality metrics dashboard development"],
        "6": ["Interprofessional team rounds", "Collaborative practice agreement development",
              "Team-based care project participation"],
        "7": ["Health policy analysis", "Systems thinking case studies",
              "Organizational assessment projects"],
        "8": ["Health informatics coursework", "EHR optimization projects",
              "Data analytics and visualization workshops"],
        "9": ["Ethics case study discussions", "Professional portfolio development",
              "Leadership reflection journaling"],
        "10": ["Executive coaching or mentorship", "Strategic planning exercises",
               "Professional development goal-setting"],
    }

    # ── Render the report ─────────────────────────────────────────

    # Header
    st.markdown(f"**Report Date:** {date_str}")
    st.markdown(f"**Competencies Assessed:** {len(scores)} of 45")

    st.markdown("---")

    # Section 1: Overall Readiness
    st.subheader(f"Overall Readiness: {readiness}")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Mean Score", f"{mean_score:.1f} / 5.0")
    with col2:
        st.metric("Strengths (4+)", f"{len(strengths)}")
    with col3:
        st.metric("Growth Areas", f"{len(gaps_positive)}")
    with col4:
        below = sum(1 for s in scores if s < 3)
        st.metric("Below Competent", f"{below}")

    st.markdown(readiness_narrative)

    st.markdown("---")

    # Section 2: Your Strengths
    st.subheader("Your Strengths")
    if not strength_domains.empty:
        top_strength = strength_domains.index[0]
        st.markdown(
            f"Your strongest domain is **{top_strength}**, where you rated "
            f"{strength_domains.iloc[0]} of its competencies at Proficient or Expert level. "
        )
        if len(strength_domains) > 1:
            others = ", ".join(strength_domains.index[1:3])
            st.markdown(f"Other strong domains include **{others}**.")

        # Visual: horizontal bar of domain averages
        fig = go.Figure(go.Bar(
            x=domain_summary['student_score'].values,
            y=domain_summary.index,
            orientation='h',
            marker_color=[
                COLORS['secondary'] if v >= 4 else
                COLORS['warning'] if v >= 3 else
                COLORS['accent']
                for v in domain_summary['student_score'].values
            ],
            text=[f"{v:.1f}" for v in domain_summary['student_score'].values],
            textposition='outside',
        ))
        fig.update_layout(
            xaxis=dict(range=[0, 5.5], title='Average Self-Assessment Score'),
            yaxis=dict(title=''),
            height=400,
            margin=dict(l=10),
        )
        fig.add_vline(x=3, line_dash="dash", line_color="gray",
                      annotation_text="Competent", annotation_position="top")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Complete more of the self-assessment to see your strengths.")

    st.markdown("---")

    # Section 3: Priority Growth Areas
    st.subheader("Priority Growth Areas")
    if not gaps_positive.empty:
        st.markdown(
            f"The analysis identified **{len(gaps_positive)} competencies** where your "
            f"self-assessment falls below the Nurse Practitioner workforce benchmark. "
            f"These are ranked below by **weighted priority**, which combines the gap "
            f"size with the competency's importance for advanced practice (derived from "
            f"machine learning analysis of O*NET workforce data)."
        )

        for i, (_, row) in enumerate(gaps_positive.head(5).iterrows()):
            d_num = row['aacn_domain']
            level_name = get_level_label(row['student_score'])
            activities = activity_map.get(d_num, ["General coursework"])

            st.markdown(
                f"**{i+1}. {row['competency_id']}: {row['competency']}**\n\n"
                f"You rated yourself at **{int(row['student_score'])} ({level_name})** "
                f"in this competency, which falls in the **{row['aacn_domain_name']}** "
                f"domain. The NP workforce benchmark for related O*NET dimensions is "
                f"**{row['benchmark']:.2f}** (on a 0-1 normalized scale). "
                f"Machine learning analysis ranks this competency's underlying "
                f"dimensions at **{row['ml_importance']:.3f}** cumulative importance "
                f"for distinguishing advanced-level practice from other healthcare roles."
            )
            st.markdown("**Recommended development activities:**")
            for act in activities:
                st.markdown(f"- {act}")
            st.markdown("")
    else:
        st.success(
            "No competency gaps detected. Your self-assessment meets or exceeds "
            "all NP workforce benchmarks."
        )

    st.markdown("---")

    # Section 4: Domain-by-Domain Summary
    st.subheader("Domain-by-Domain Summary")
    for d_num in sorted(AACN_DOMAIN_NAMES.keys(), key=int):
        d_name = AACN_DOMAIN_NAMES[d_num]
        d_comps = gaps_df[gaps_df['aacn_domain'] == d_num]
        if d_comps.empty:
            continue
        d_mean = d_comps['student_score'].mean()
        d_gaps = d_comps[d_comps['gap'] > 0]

        if d_mean >= 4:
            icon = "**Strong**"
        elif d_mean >= 3:
            icon = "Developing"
        else:
            icon = "Needs Focus"

        with st.expander(f"Domain {d_num}: {d_name} (Avg: {d_mean:.1f}, {icon})"):
            for _, row in d_comps.iterrows():
                level = get_level_label(row['student_score'])
                gap_indicator = ""
                if row['gap'] > 0:
                    gap_indicator = f" | Gap: {row['gap']:.2f}"
                st.markdown(
                    f"- **{row['competency_id']}**: {row['competency'][:70]}... "
                    f"Score: {int(row['student_score'])} ({level}){gap_indicator}"
                )

    st.markdown("---")

    # Section 5: Download full report as text
    st.subheader("Download Your Report")

    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("AACN AL ESSENTIALS COMPETENCY SELF-ASSESSMENT REPORT")
    report_lines.append("=" * 60)
    report_lines.append(f"Date: {date_str}")
    report_lines.append(f"Competencies Assessed: {len(scores)} of 45")
    report_lines.append(f"Overall Readiness: {readiness}")
    report_lines.append(f"Mean Score: {mean_score:.1f} / 5.0")
    report_lines.append(f"Strengths (score >= 4): {len(strengths)}")
    report_lines.append(f"Growth Areas: {len(gaps_positive)}")
    report_lines.append("")
    report_lines.append(readiness_narrative)
    report_lines.append("")
    report_lines.append("-" * 60)
    report_lines.append("PRIORITY GROWTH AREAS (ranked by ML-weighted importance)")
    report_lines.append("-" * 60)

    for i, (_, row) in enumerate(gaps_positive.iterrows()):
        level_name = get_level_label(row['student_score'])
        report_lines.append(
            f"\n{i+1}. {row['competency_id']}: {row['competency']}\n"
            f"   Domain: {row['aacn_domain_name']}\n"
            f"   Your Score: {int(row['student_score'])} ({level_name})\n"
            f"   NP Benchmark: {row['benchmark']:.3f}\n"
            f"   Gap: {row['gap']:.3f}\n"
            f"   ML Importance: {row['ml_importance']:.4f}\n"
            f"   Weighted Priority: {row['weighted_gap']:.3f}"
        )

    report_lines.append("")
    report_lines.append("-" * 60)
    report_lines.append("DOMAIN SUMMARY")
    report_lines.append("-" * 60)
    for d_num in sorted(AACN_DOMAIN_NAMES.keys(), key=int):
        d_name = AACN_DOMAIN_NAMES[d_num]
        d_comps = gaps_df[gaps_df['aacn_domain'] == d_num]
        if d_comps.empty:
            continue
        d_mean = d_comps['student_score'].mean()
        report_lines.append(f"\nDomain {d_num}: {d_name} (Avg: {d_mean:.1f})")
        for _, row in d_comps.iterrows():
            level = get_level_label(row['student_score'])
            report_lines.append(
                f"  {row['competency_id']}: Score {int(row['student_score'])} ({level})"
            )

    report_lines.append("")
    report_lines.append("-" * 60)
    report_lines.append(
        "Data Source: O*NET 30.1 (December 2025), CC-BY 4.0 License.\n"
        "National Center for O*NET Development.\n"
        "U.S. Department of Labor, Employment and Training Administration.\n"
        "Dashboard: Jacksonville University, DSIM 608 Capstone."
    )

    report_text = "\n".join(report_lines)

    dl_col1, dl_col2 = st.columns(2)
    with dl_col1:
        st.download_button(
            "Download Report (Text)",
            data=report_text,
            file_name=f"competency_report_{datetime.now().strftime('%Y%m%d')}.txt",
            mime="text/plain",
        )
    with dl_col2:
        # Generate PDF
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=20)
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, "AACN AL Essentials Competency Report", ln=True, align="C")
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 6, f"Date: {date_str}", ln=True, align="C")
        pdf.cell(0, 6, "Jacksonville University, Keigwin School of Nursing", ln=True, align="C")
        pdf.ln(8)

        # Summary box
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, f"Overall Readiness: {readiness}", ln=True)
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 6, f"Mean Score: {mean_score:.1f} / 5.0    |    "
                       f"Strengths: {len(strengths)}    |    "
                       f"Growth Areas: {len(gaps_positive)}", ln=True)
        pdf.ln(4)
        pdf.multi_cell(0, 5, readiness_narrative)
        pdf.ln(6)

        # Strengths
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Your Strengths", ln=True)
        pdf.set_font("Helvetica", "", 10)
        if not strength_domains.empty:
            pdf.multi_cell(0, 5,
                f"Strongest domain: {strength_domains.index[0]} "
                f"({strength_domains.iloc[0]} competencies at Proficient or Expert)."
            )
        pdf.ln(4)

        # Priority growth areas
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Priority Growth Areas", ln=True)
        pdf.set_font("Helvetica", "", 10)

        for i, (_, row) in enumerate(gaps_positive.head(10).iterrows()):
            level_name = get_level_label(row['student_score'])
            pdf.set_font("Helvetica", "B", 10)
            pdf.multi_cell(0, 5,
                f"{i+1}. {row['competency_id']}: "
                f"{row['competency'][:70]}"
            )
            pdf.set_font("Helvetica", "", 9)
            pdf.cell(0, 5,
                f"   Score: {int(row['student_score'])} ({level_name})  |  "
                f"Benchmark: {row['benchmark']:.2f}  |  "
                f"Gap: {row['gap']:.3f}", ln=True
            )
            pdf.ln(2)

        # Domain summary
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Domain Summary", ln=True)
        pdf.set_font("Helvetica", "", 9)
        for d_num in sorted(AACN_DOMAIN_NAMES.keys(), key=int):
            d_name = AACN_DOMAIN_NAMES[d_num]
            d_comps = gaps_df[gaps_df['aacn_domain'] == d_num]
            if d_comps.empty:
                continue
            d_mean = d_comps['student_score'].mean()
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(0, 6, f"Domain {d_num}: {d_name} (Avg: {d_mean:.1f})", ln=True)
            pdf.set_font("Helvetica", "", 9)
            for _, row in d_comps.iterrows():
                level = get_level_label(row['student_score'])
                pdf.cell(0, 5,
                    f"  {row['competency_id']}: {int(row['student_score'])} ({level})",
                    ln=True
                )
            pdf.ln(2)

        # Footer
        pdf.ln(6)
        pdf.set_font("Helvetica", "I", 8)
        pdf.multi_cell(0, 4,
            "Data Source: O*NET 30.1 (December 2025), CC-BY 4.0 License. "
            "National Center for O*NET Development. "
            "U.S. Department of Labor, Employment and Training Administration. "
            "Dashboard: Jacksonville University, DSIM 608 Capstone."
        )

        pdf_bytes = pdf.output()
        st.download_button(
            "Download Report (PDF)",
            data=bytes(pdf_bytes),
            file_name=f"competency_report_{datetime.now().strftime('%Y%m%d')}.pdf",
            mime="application/pdf",
        )


# ── Page: ML Insights ─────────────────────────────────────────────
def render_ml_insights(feat_imp, model_comp, clusters, df):
    st.title("Faculty Analytics")
    st.markdown(
        "This page shows the machine learning methods behind the dashboard's "
        "competency weighting system. It is designed for faculty, program "
        "directors, and analytics audiences."
    )

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
AACN 2026 Advanced-Level Essentials and receive ML-weighted, personalized
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


# ── Page: Priority & Simulation (Week 6) ─────────────────────────
# O*NET domain name map between lowercase (DB) and Capitalized (crosswalk JSON)
_ONET_DOMAIN_CAP = {
    'knowledge': 'Knowledge', 'skills': 'Skills', 'abilities': 'Abilities',
    'work_activities': 'Work Activities', 'work_styles': 'Work Styles',
    'work_context': 'Work Context',
}


@st.cache_data
def _aacn_domain_scores_from_occupation(_df, _crosswalk, soc_code):
    """Compute AACN domain scores (1-10) for a single occupation.

    For each AACN competency, average the scores of the O*NET dimensions
    listed in the crosswalk. Then average competencies within each AACN domain.
    Returns dict: {domain_num (str): score}.
    """
    occ = _df[_df.soc_code == soc_code]
    # dim scores lookup: (O*NET domain cap, element_name) -> score
    dim_scores = {}
    for _, row in occ.iterrows():
        key = (_ONET_DOMAIN_CAP.get(row['domain'], row['domain']), row['element_name'])
        dim_scores[key] = row['normalized_value']

    from collections import defaultdict
    comp_scores = {}
    for comp_id, entry in _crosswalk.items():
        od = entry.get('onet_dimensions', {}) or {}
        vals = []
        for onet_dom, dim_list in od.items():
            for dim in (dim_list or []):
                v = dim_scores.get((onet_dom, dim))
                if v is not None:
                    vals.append(v)
        if vals:
            comp_scores[comp_id] = float(np.mean(vals))

    # Aggregate to AACN domain
    by_dom = defaultdict(list)
    for comp_id, score in comp_scores.items():
        dom_num = comp_id.split('.')[0]
        by_dom[dom_num].append(score)
    return {d: float(np.mean(scores)) for d, scores in by_dom.items()}


@st.cache_data
def _simulate_aacn_cohort(_df, _crosswalk, n_students=50, seed=42):
    """Simulate N DNP students taking the self-assessment.

    Each simulated student is drawn as a mix of RN baseline and NP target
    competency with a random progress factor (Beta(2, 3), mean ~0.4) plus noise.
    This represents a realistic DNP cohort mid-program: some closer to RN
    (entering), some closer to NP (near graduation).

    Returns DataFrame with columns: student_id, aacn_domain, score.
    """
    rn_aacn = _aacn_domain_scores_from_occupation(_df, _crosswalk, '29-1141.00')
    np_aacn = _aacn_domain_scores_from_occupation(_df, _crosswalk, '29-1171.00')

    rng = np.random.default_rng(seed)
    rows = []
    for sid in range(n_students):
        progress = float(rng.beta(2, 3))  # 0 = pure RN, 1 = pure NP; mean ~0.4
        for d in sorted(rn_aacn.keys(), key=int):
            baseline = rn_aacn[d]
            target = np_aacn.get(d, baseline)
            expected = baseline + (target - baseline) * progress
            noise = rng.normal(0, 0.05)
            score = float(np.clip(expected + noise, 0.0, 1.0))
            rows.append({'student_id': sid, 'aacn_domain': d, 'score': score})
    cohort = pd.DataFrame(rows)
    # also return the RN and NP anchors for overlay
    return cohort, rn_aacn, np_aacn


@st.cache_data
def _compute_priority_table(_df, _feat_imp):
    """Join NP-vs-RN dimension gaps with RF importance; rank by composite z-score."""
    rn = (_df[_df.soc_code == '29-1141.00']
          .groupby(['domain', 'element_name'])['normalized_value'].mean())
    npp = (_df[_df.soc_code == '29-1171.00']
           .groupby(['domain', 'element_name'])['normalized_value'].mean())
    gaps = (npp - rn).dropna().reset_index()
    gaps.columns = ['domain', 'dimension', 'gap']
    # feat_imp has columns: domain, dimension, importance (or rf_importance)
    imp = _feat_imp.copy()
    if 'rf_importance' not in imp.columns and 'importance' in imp.columns:
        imp = imp.rename(columns={'importance': 'rf_importance'})
    merged = gaps.merge(
        imp[['domain', 'dimension', 'rf_importance']],
        on=['domain', 'dimension'], how='inner',
    )
    pos = merged[merged.gap > 0].copy()
    pos['z_gap'] = (pos['gap'] - pos['gap'].mean()) / pos['gap'].std()
    pos['z_imp'] = (pos['rf_importance'] - pos['rf_importance'].mean()) / pos['rf_importance'].std()
    pos['priority'] = pos['z_gap'] + pos['z_imp']
    return pos


@st.cache_data
def _run_simulation(_df, seed=42, n_boot=10000, n_sim=5000):
    """Re-run Week 5 simulation (bootstrap + pathway Monte Carlo), seed = 42."""
    profiles = {}
    for label, code in [('RN', '29-1141.00'), ('NP', '29-1171.00')]:
        occ = _df[_df.soc_code == code]
        p = {}
        for dom in DOMAIN_ORDER:
            vals = occ[occ.domain == dom]['normalized_value'].values
            p[dom] = {'values': vals, 'mean': float(vals.mean())}
        profiles[label] = p

    np.random.seed(seed)

    gap_results = {}
    for dom in DOMAIN_ORDER:
        rv = profiles['RN'][dom]['values']
        nv = profiles['NP'][dom]['values']
        obs = float(nv.mean() - rv.mean())
        boot = np.empty(n_boot)
        for i in range(n_boot):
            r = np.random.choice(rv, len(rv), True)
            n = np.random.choice(nv, len(nv), True)
            boot[i] = n.mean() - r.mean()
        ci_lo = float(np.percentile(boot, 2.5))
        ci_hi = float(np.percentile(boot, 97.5))
        gap_results[dom] = {
            'obs': obs, 'boot': boot.tolist(),
            'ci_lo': ci_lo, 'ci_hi': ci_hi,
            'sig': not (ci_lo <= 0 <= ci_hi),
        }

    scenario_weights = {
        'Balanced': {d: 1.0 for d in DOMAIN_ORDER},
        'Knowledge-Heavy': {'knowledge': 2.5, 'skills': 1.0, 'abilities': 0.8,
                            'work_activities': 0.8, 'work_styles': 0.5, 'work_context': 0.4},
        'Clinical-Focus': {'knowledge': 1.5, 'skills': 2.0, 'abilities': 1.5,
                           'work_activities': 2.0, 'work_styles': 0.5, 'work_context': 0.5},
        'Leadership-Focus': {'knowledge': 1.0, 'skills': 1.5, 'abilities': 0.8,
                             'work_activities': 1.5, 'work_styles': 1.5, 'work_context': 1.2},
    }
    pathway_results = {}
    for name, w in scenario_weights.items():
        tot = sum(w.values())
        nw = {d: v / tot for d, v in w.items()}
        sim = np.zeros((n_sim, len(DOMAIN_ORDER)))
        for s in range(n_sim):
            for di, dom in enumerate(DOMAIN_ORDER):
                rv = profiles['RN'][dom]['values']
                nv = profiles['NP'][dom]['values']
                rb = np.random.choice(rv, len(rv), True).mean()
                nb = np.random.choice(nv, len(nv), True).mean()
                closure = nw[dom] * np.random.beta(3, 2)
                sim[s, di] = rb + (nb - rb) * closure
        probs = {d: float((sim[:, i] >= profiles['NP'][d]['mean'] * 0.9).mean())
                 for i, d in enumerate(DOMAIN_ORDER)}
        pathway_results[name] = {
            'weights': nw, 'probs': probs,
            'overall': float(np.mean(list(probs.values()))),
        }
    return gap_results, pathway_results


def render_priority_simulation(df, feat_imp):
    """Week 6 Priority & Simulation page.

    Seven analytical views framed around AACN 2026 Advanced-Level Essentials.
    O*NET 30.1 workforce data is the baseline reference: the Registered Nurse
    (RN) profile represents incoming DNP students, and the Nurse Practitioner
    (NP) profile represents the target competency level post-DNP.
    """
    st.title("Priority & Simulation")
    st.markdown(
        "**Week 6 Faculty Analytics.** Seven views aligned with the "
        "**AACN 2026 Advanced-Level Essentials** (10 domains, 45 competencies). "
        "O*NET 30.1 workforce data provides the baseline: RN = entering DNP "
        "student, NP = expected post-DNP advanced practice competency level. "
        "All stochastic steps use `seed = 42` for reproducibility."
    )

    domain_colors = {
        'knowledge': '#264653', 'skills': '#2a9d8f', 'abilities': '#e9c46a',
        'work_activities': '#f4a261', 'work_styles': '#e76f51', 'work_context': '#606c38',
    }
    aacn_palette = [
        '#264653', '#2a9d8f', '#e9c46a', '#f4a261', '#e76f51',
        '#606c38', '#1b4965', '#bc6c25', '#9d6b53', '#457b9d',
    ]
    scenario_colors = {
        'Balanced': '#1B2A4A', 'Knowledge-Heavy': '#2a9d8f',
        'Clinical-Focus': '#e07a5f', 'Leadership-Focus': '#e9c46a',
    }

    if feat_imp is None or feat_imp.empty:
        st.warning("Feature importance data not available. Cannot render priority views.")
        return

    pos = _compute_priority_table(df, feat_imp)

    section = st.radio(
        "Section",
        ["Cohort Overview (AACN)",
         "Gap Priority Matrix",
         "Personalized Learning Roadmap",
         "Priority 3D Space",
         "Top 12 Priority Flow",
         "Bootstrap Distributions",
         "Learning Pathway Simulation"],
        horizontal=True,
    )

    # 0. Cohort Overview (AACN) ──────────────────────────────────
    if section == "Cohort Overview (AACN)":
        st.subheader("Cohort Competency Overview, by AACN Domain")
        st.caption(
            "Simulated DNP cohort of 50 students taking the self-assessment. "
            "Scores are modeled as a mix of RN baseline (entering) and NP target "
            "(exiting), with a random progress factor and small noise. "
            "This is the faculty view that informs program improvement decisions."
        )

        crosswalk = st.session_state.get('crosswalk') or load_crosswalk()
        n_students = st.slider("Simulated cohort size", 20, 200, 50, step=10,
                                 help="How many students taking the self-assessment to simulate")

        cohort, rn_aacn, np_aacn = _simulate_aacn_cohort(
            df, crosswalk, n_students=n_students, seed=42)

        # Order domains by AACN number and add label
        dom_order = sorted(rn_aacn.keys(), key=int)
        dom_labels = {d: f"{d}. {AACN_DOMAIN_NAMES.get(d, 'Domain ' + d)}" for d in dom_order}
        cohort['label'] = cohort['aacn_domain'].map(dom_labels)

        # Box plot per AACN domain
        fig = go.Figure()
        for i, d in enumerate(dom_order):
            sub = cohort[cohort.aacn_domain == d]
            fig.add_trace(go.Box(
                y=sub['score'], name=dom_labels[d],
                boxpoints='outliers', fillcolor=aacn_palette[i % len(aacn_palette)],
                marker=dict(color=aacn_palette[i % len(aacn_palette)], size=4),
                line=dict(color='#1B2A4A'),
                hovertemplate=(f"<b>Domain {d}</b><br>"
                               f"Score: %{{y:.3f}}<extra></extra>"),
            ))
        # overlay RN baseline (dashed) and NP target (solid) per domain as markers
        for i, d in enumerate(dom_order):
            fig.add_trace(go.Scatter(
                x=[dom_labels[d]], y=[rn_aacn[d]],
                mode='markers', name='RN baseline' if i == 0 else None,
                marker=dict(symbol='line-ew-open', size=22,
                            color='#888', line=dict(width=3)),
                showlegend=(i == 0),
                hovertemplate=f"RN baseline domain {d}: {rn_aacn[d]:.3f}<extra></extra>",
            ))
            fig.add_trace(go.Scatter(
                x=[dom_labels[d]], y=[np_aacn[d]],
                mode='markers', name='NP target' if i == 0 else None,
                marker=dict(symbol='star', size=14,
                            color='#e07a5f', line=dict(color='white', width=1)),
                showlegend=(i == 0),
                hovertemplate=f"NP target domain {d}: {np_aacn[d]:.3f}<extra></extra>",
            ))
        fig.update_layout(
            height=620, template='plotly_white',
            yaxis=dict(title='Competency Score (0 to 1)', range=[0, 1]),
            xaxis=dict(tickangle=-25, tickfont=dict(size=10)),
            margin=dict(b=180),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Summary stats
        st.markdown("#### Program Improvement Insights")
        summary_rows = []
        for d in dom_order:
            sub = cohort[cohort.aacn_domain == d]['score']
            gap_to_np = np_aacn[d] - sub.mean()
            summary_rows.append({
                'AACN Domain': dom_labels[d],
                'Cohort Mean': f"{sub.mean():.3f}",
                'Cohort SD': f"{sub.std():.3f}",
                'NP Target': f"{np_aacn[d]:.3f}",
                'Gap to NP': f"{gap_to_np:+.3f}",
                'Students below 0.5': int((sub < 0.5).sum()),
            })
        sum_df = pd.DataFrame(summary_rows)
        # Sort by gap (largest first)
        sum_df['_sort'] = sum_df['Gap to NP'].str.replace('+', '').astype(float)
        sum_df = sum_df.sort_values('_sort', ascending=False).drop(columns=['_sort'])
        st.dataframe(sum_df.reset_index(drop=True), use_container_width=True, hide_index=True)

        # Top callout
        gaps = {d: np_aacn[d] - cohort[cohort.aacn_domain == d]['score'].mean()
                for d in dom_order}
        top_gap_d = max(gaps, key=gaps.get)
        st.success(
            f"**Program improvement priority:** Domain {top_gap_d}, "
            f"{AACN_DOMAIN_NAMES[top_gap_d]}, has the largest cohort gap to NP "
            f"({gaps[top_gap_d]:+.3f}). This is where curriculum reinforcement "
            "would have the highest return on cohort preparation."
        )

    # 1. Gap Priority Matrix
    if section == "Gap Priority Matrix":
        st.subheader("Gap Priority Matrix")
        st.caption(
            f"All {len(pos)} positive-gap dimensions, colored by O*NET domain. "
            "X = RF importance (workforce signal). Y = gap (developmental need). "
            "Upper-right quadrant = high-priority curriculum targets."
        )
        fig = go.Figure()
        for dom in DOMAIN_ORDER:
            sub = pos[pos.domain == dom]
            fig.add_trace(go.Scatter(
                x=sub['rf_importance'], y=sub['gap'],
                mode='markers', name=DOMAIN_LABELS[dom],
                marker=dict(size=10, color=domain_colors[dom], opacity=0.78,
                            line=dict(color='white', width=0.7)),
                text=sub['dimension'],
                hovertemplate="<b>%{text}</b><br>Gap: %{y:+.3f}<br>RF: %{x:.4f}<extra></extra>",
            ))
        median_imp = pos['rf_importance'].median()
        fig.add_vline(x=median_imp, line_dash='dash', line_color='gray')
        fig.update_layout(
            height=560, template='plotly_white',
            xaxis_title='Random Forest Feature Importance',
            yaxis_title='NP - RN Gap')
        st.plotly_chart(fig, use_container_width=True)

    # 2. Personalized Learning Roadmap
    elif section == "Personalized Learning Roadmap":
        st.subheader("Personalized Learning Roadmap, Top 12")
        top12 = pos.nlargest(12, 'priority').reset_index(drop=True)
        top12['rank'] = top12.index + 1
        disp = top12.iloc[::-1].reset_index(drop=True)
        fig = go.Figure(go.Bar(
            x=disp['priority'], y=disp['dimension'],
            orientation='h',
            marker_color=[domain_colors[d] for d in disp['domain']],
            text=[f"gap {g:+.3f}  |  RF {imp:.4f}"
                  for g, imp in zip(disp['gap'], disp['rf_importance'])],
            textposition='outside',
        ))
        fig.update_layout(
            height=600, template='plotly_white',
            xaxis_title='Composite Priority (z-gap + z-importance)',
            yaxis=dict(tickfont=dict(size=11)))
        st.plotly_chart(fig, use_container_width=True)
        st.info(
            f"**Top priority dimension:** {top12.iloc[0]['dimension']} "
            f"({DOMAIN_LABELS[top12.iloc[0]['domain']]}). "
            "Higher bars = larger development impact from focused effort."
        )

    # 3. Priority 3D Space
    elif section == "Priority 3D Space":
        st.subheader("Priority 3D Space")
        st.caption(
            f"{len(pos)} positive-gap dimensions plotted in 3D. "
            "Rotate with the mouse. Hover for dimension name."
        )
        fig = go.Figure()
        for dom in DOMAIN_ORDER:
            sub = pos[pos.domain == dom]
            fig.add_trace(go.Scatter3d(
                x=sub['gap'], y=sub['rf_importance'], z=sub['priority'],
                mode='markers', name=DOMAIN_LABELS[dom],
                marker=dict(size=5, color=domain_colors[dom], opacity=0.85,
                            line=dict(color='white', width=0.4)),
                text=sub['dimension'],
                hovertemplate="<b>%{text}</b><br>Gap: %{x:+.3f}<br>"
                              "RF: %{y:.4f}<br>Priority: %{z:.3f}<extra></extra>",
            ))
        fig.update_layout(
            height=660,
            scene=dict(xaxis_title='NP - RN Gap',
                       yaxis_title='RF Importance',
                       zaxis_title='Composite Priority'))
        st.plotly_chart(fig, use_container_width=True)

    # 4. Priority Flow (improved Sankey) ────────────────────────
    elif section == "Top 12 Priority Flow":
        st.subheader("Top 12 Priority Competencies, Flow to O*NET Domain")
        st.caption(
            "Where the top 12 priority workforce competencies sit within the "
            "six O*NET domains. Link width = composite priority score (z-gap + z-importance)."
        )
        top12 = pos.nlargest(12, 'priority').copy()

        def shorten(name, n=32):
            return name if len(name) <= n else name[: n - 1] + '…'

        dim_labels = [shorten(x) for x in top12['dimension']]
        dom_labels = [DOMAIN_LABELS[d] for d in DOMAIN_ORDER]
        nodes = dim_labels + dom_labels
        dim_idx = {top12['dimension'].iloc[i]: i for i in range(len(top12))}
        dom_idx = {DOMAIN_LABELS[d]: len(dim_labels) + i
                    for i, d in enumerate(DOMAIN_ORDER)}

        def hex_to_rgba(hx, a=0.55):
            hx = hx.lstrip('#')
            r, g, b = int(hx[0:2], 16), int(hx[2:4], 16), int(hx[4:6], 16)
            return f'rgba({r},{g},{b},{a})'

        sources, targets, values, link_colors, link_labels = [], [], [], [], []
        for _, row in top12.iterrows():
            sources.append(dim_idx[row['dimension']])
            targets.append(dom_idx[DOMAIN_LABELS[row['domain']]])
            values.append(float(row['priority']))
            link_colors.append(hex_to_rgba(domain_colors[row['domain']]))
            link_labels.append(f"{row['dimension']}  (priority {row['priority']:.2f})")
        node_colors = (['#1B2A4A'] * len(dim_labels)
                       + [domain_colors[d] for d in DOMAIN_ORDER])
        fig = go.Figure(data=[go.Sankey(
            arrangement='snap',
            node=dict(label=nodes, color=node_colors,
                      pad=28, thickness=28,
                      line=dict(color='white', width=1)),
            link=dict(source=sources, target=targets, value=values,
                      color=link_colors, label=link_labels,
                      hovertemplate='%{label}<extra></extra>'),
        )])
        fig.update_layout(
            height=780,
            font=dict(size=14, family='Calibri, Arial'),
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("See the top 12 with full dimension names"):
            disp = top12.copy()
            disp['domain'] = disp['domain'].map(DOMAIN_LABELS)
            disp = disp[['dimension', 'domain', 'gap', 'rf_importance', 'priority']]
            disp.columns = ['Dimension (full name)', 'O*NET Domain', 'Gap', 'RF Importance', 'Priority']
            st.dataframe(disp.reset_index(drop=True), use_container_width=True, hide_index=True)

    # 5. Bootstrap Distributions
    elif section == "Bootstrap Distributions":
        st.subheader("Bootstrap Sampling Distributions (NP vs RN)")
        with st.spinner("Running 10,000 bootstrap resamples per domain..."):
            gap_results, _ = _run_simulation(df, seed=42, n_boot=10000, n_sim=5000)
        fig = go.Figure()
        for dom in DOMAIN_ORDER:
            d = gap_results[dom]
            fig.add_trace(go.Violin(
                y=d['boot'], name=DOMAIN_LABELS[dom],
                box_visible=True, meanline_visible=True,
                fillcolor=domain_colors[dom], opacity=0.78,
                line_color='#1B2A4A',
                hovertemplate=(f"<b>{DOMAIN_LABELS[dom]}</b><br>"
                               f"Observed: {d['obs']:+.4f}<br>"
                               f"95% CI: [{d['ci_lo']:+.4f}, {d['ci_hi']:+.4f}]<br>"
                               f"Sig: {'Yes' if d['sig'] else 'No'}<extra></extra>"),
            ))
        fig.update_layout(height=500, template='plotly_white',
                          yaxis_title='Gap (NP minus RN)', showlegend=False)
        fig.add_hline(y=0, line_dash='dot', line_color='gray')
        st.plotly_chart(fig, use_container_width=True)
        st.info(
            "**Knowledge** is the only domain with a significant NP-vs-RN gap "
            "(observed +0.113, 95% CI [+0.007, +0.220]). The other five CIs cross zero."
        )

    # 6. Learning Pathway Simulation
    elif section == "Learning Pathway Simulation":
        st.subheader("Learning Pathway Simulation")
        with st.spinner("Running 20,000 pathway Monte Carlo simulations..."):
            _, pathway_results = _run_simulation(df, seed=42, n_boot=10000, n_sim=5000)

        scenario = st.radio(
            "Pathway scenario",
            list(pathway_results.keys()), horizontal=True, index=2,  # Clinical-Focus default
        )
        probs = pathway_results[scenario]['probs']
        weights = pathway_results[scenario]['weights']
        overall = pathway_results[scenario]['overall']

        c1, c2 = st.columns([2, 1])
        with c1:
            st.markdown(f"#### Per-domain achievement probability, {scenario}")
            fig = go.Figure(go.Bar(
                x=[DOMAIN_LABELS[d] for d in DOMAIN_ORDER],
                y=[probs[d] for d in DOMAIN_ORDER],
                marker_color=[domain_colors[d] for d in DOMAIN_ORDER],
                text=[f"{probs[d]:.0%}" for d in DOMAIN_ORDER],
                textposition='outside',
            ))
            fig.add_hline(y=0.8, line_dash='dash', line_color='#e07a5f',
                          annotation_text="80% goal")
            fig.update_layout(
                height=420, template='plotly_white', showlegend=False,
                yaxis=dict(tickformat='.0%', range=[0, 1.15]))
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.metric(
                f"Overall P(reach 90% NP), {scenario}", f"{overall:.1%}"
            )
            st.markdown("**Effort breakdown:**")
            fig_w = go.Figure(go.Pie(
                labels=[DOMAIN_LABELS[d] for d in DOMAIN_ORDER],
                values=[weights[d] for d in DOMAIN_ORDER],
                marker=dict(colors=[domain_colors[d] for d in DOMAIN_ORDER]),
                hole=0.45,
                textinfo='label+percent', textfont=dict(size=9),
            ))
            fig_w.update_layout(height=340, showlegend=False,
                                margin=dict(t=10, b=10, l=10, r=10))
            st.plotly_chart(fig_w, use_container_width=True)

        st.markdown("---")
        st.markdown("#### Compare all four pathways")
        names = list(pathway_results.keys())
        overalls = [pathway_results[n]['overall'] for n in names]
        order = np.argsort(overalls)[::-1]
        fig_race = go.Figure(go.Bar(
            x=[names[i] for i in order],
            y=[overalls[i] for i in order],
            marker_color=[scenario_colors[names[i]] for i in order],
            text=[f"{overalls[i]:.1%}" for i in order], textposition='outside',
        ))
        fig_race.update_layout(height=380, template='plotly_white',
                               yaxis=dict(tickformat='.0%', range=[0.6, 0.75]))
        st.plotly_chart(fig_race, use_container_width=True)
        st.caption(
            "Clinical-Focus leads overall at 70.3%. Margin over Balanced is narrow "
            "(0.9 percentage points). The meaningful differences are at the domain level, "
            "not overall. 5,000 Monte Carlo simulations per scenario, seed = 42."
        )


if __name__ == "__main__":
    main()
