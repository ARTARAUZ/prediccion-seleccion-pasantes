import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

st.set_page_config(page_title="Predicción de Selección de Pasantes", page_icon="🎓", layout="wide")

DATA_PATH = "data/internship_selection_dataset.csv"

NUMERIC_FEATURES = [
    "CGPA", "skills_score", "projects_count", "internships_done",
    "communication_score", "aptitude_score", "coding_test_score", "resume_score",
    "hackathons_participated", "certifications_count", "linkedin_activity_score",
    "github_score", "soft_skills_score", "interview_score", "consistency_score",
    "backlogs",
]
CATEGORICAL_FEATURES = ["extracurricular", "college_tier", "placement_training"]
TARGET = "selected"

# ---------------------------------------------------------------------------
# Paleta pastel (fondo muy claro + texto oscuro = buen contraste garantizado)
# ---------------------------------------------------------------------------
INK = "#25293B"
MUTED_INK = "#6B7089"
CARD_BLUE = "#DCE7FB"
CARD_MINT = "#DBF3E8"
CARD_LAVENDER = "#E7E1FB"
CARD_PEACH = "#FCE7DC"
SUCCESS_BG = "#D6F5E3"
SUCCESS_BORDER = "#8FD9B6"
ERROR_BG = "#FBE0E3"
ERROR_BORDER = "#F0A8B4"
BAR_COLOR = "#6C86CE"
BAR_TRACK = "#EDEBFB"

st.markdown(f"""
<style>
    .stat-card {{
        border-radius: 16px;
        padding: 1.1rem 1.2rem;
        text-align: center;
        border: 1px solid rgba(37,41,59,0.06);
        box-shadow: 0 2px 10px rgba(37,41,59,0.06);
    }}
    .stat-label {{
        font-size: 0.8rem;
        color: {MUTED_INK};
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 0.3rem;
    }}
    .stat-value {{
        font-size: 1.9rem;
        font-weight: 700;
        color: {INK};
    }}
    .result-card {{
        border-radius: 18px;
        padding: 1.6rem 1.8rem;
        margin-top: 0.5rem;
        border: 1.5px solid;
    }}
    .result-title {{
        font-size: 1.4rem;
        font-weight: 700;
        color: {INK};
        margin-bottom: 0.2rem;
    }}
    .result-sub {{
        font-size: 0.95rem;
        color: {MUTED_INK};
    }}
    .hero-banner {{
        background: linear-gradient(135deg, {CARD_LAVENDER} 0%, {CARD_BLUE} 100%);
        border-radius: 20px;
        padding: 1.8rem 2rem;
        margin-bottom: 1.2rem;
        border: 1px solid rgba(37,41,59,0.06);
    }}
    .hero-title {{
        font-size: 1.9rem;
        font-weight: 800;
        color: {INK};
        margin-bottom: 0.2rem;
    }}
    .hero-sub {{
        font-size: 1rem;
        color: {MUTED_INK};
    }}
    section[data-testid="stForm"] {{
        border-radius: 16px;
        border: 1px solid rgba(37,41,59,0.08);
        padding: 1.2rem 1.4rem;
        background-color: rgba(255,255,255,0.5);
    }}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def train_model(df: pd.DataFrame):
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=300, random_state=42)),
    ])
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    }

    feature_names = model.named_steps["preprocessor"].get_feature_names_out()
    importances = model.named_steps["classifier"].feature_importances_
    importance_df = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=True)
        .tail(10)
    )
    importance_df["feature"] = importance_df["feature"].str.replace(r"^(num__|cat__)", "", regex=True)

    return model, metrics, importance_df


def stat_card(label, value, bg):
    st.markdown(f"""
    <div class="stat-card" style="background-color:{bg};">
        <div class="stat-label">{label}</div>
        <div class="stat-value">{value}</div>
    </div>
    """, unsafe_allow_html=True)


def feature_importance_chart(importance_df):
    fig = go.Figure(go.Bar(
        x=importance_df["importance"],
        y=importance_df["feature"],
        orientation="h",
        marker=dict(color=BAR_COLOR, line=dict(width=0)),
        text=[f"{v:.3f}" for v in importance_df["importance"]],
        textposition="outside",
        textfont=dict(color=INK, size=12),
        hovertemplate="%{y}: %{x:.3f}<extra></extra>",
    ))
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=30, t=10, b=10),
        height=380,
        xaxis=dict(showgrid=True, gridcolor=BAR_TRACK, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, color=INK, tickfont=dict(size=13)),
        font=dict(color=INK, family="system-ui, -apple-system, Segoe UI, sans-serif"),
    )
    return fig


def probability_gauge(probability):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        number={"suffix": "%", "font": {"size": 42, "color": INK}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": MUTED_INK, "tickfont": {"color": MUTED_INK}},
            "bar": {"color": BAR_COLOR, "thickness": 0.35},
            "bgcolor": BAR_TRACK,
            "borderwidth": 0,
            "steps": [
                {"range": [0, 50], "color": CARD_PEACH},
                {"range": [50, 100], "color": CARD_MINT},
            ],
        },
    ))
    fig.update_layout(
        height=220,
        margin=dict(l=20, r=20, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=INK, family="system-ui, -apple-system, Segoe UI, sans-serif"),
    )
    return fig


def main():
    st.markdown(f"""
    <div class="hero-banner">
        <div class="hero-title">🎓 Predicción de Selección de Pasantes</div>
        <div class="hero-sub">Modelo de clasificación entrenado en vivo · Internship Selection Dataset · TechNova Solutions</div>
    </div>
    """, unsafe_allow_html=True)

    df = load_data()
    model, metrics, importance_df = train_model(df)

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        stat_card("Accuracy", f"{metrics['accuracy']:.1%}", CARD_BLUE)
    with m2:
        stat_card("Precision", f"{metrics['precision']:.1%}", CARD_MINT)
    with m3:
        stat_card("Recall", f"{metrics['recall']:.1%}", CARD_LAVENDER)
    with m4:
        stat_card("F1-score", f"{metrics['f1']:.1%}", CARD_PEACH)

    base_rate = df[TARGET].mean()
    if metrics["accuracy"] <= base_rate + 0.02:
        st.caption(
            f"⚠️ El accuracy está muy cerca de la tasa base ({base_rate:.1%} de candidatos seleccionados en el histórico). "
            "Las variables individuales tienen correlación muy baja con `selected` en este dataset — vale la pena "
            "mencionarlo como hallazgo en las conclusiones del proyecto."
        )

    st.write("")
    left, right = st.columns([1.1, 1])

    with left:
        st.subheader("📋 Perfil del candidato")

        with st.form("candidate_form"):
            c1, c2 = st.columns(2)
            with c1:
                cgpa = st.slider("CGPA", 0.0, 10.0, 7.5, 0.01)
                skills_score = st.slider("Skills score", 0, 10, 5)
                projects_count = st.number_input("Proyectos completados", 0, 20, 3)
                internships_done = st.number_input("Pasantías previas", 0, 10, 0)
                communication_score = st.slider("Comunicación", 0, 10, 5)
                aptitude_score = st.slider("Aptitud lógica", 0, 10, 5)
                coding_test_score = st.slider("Prueba de código", 0, 10, 5)
                resume_score = st.slider("Calidad del CV", 0, 10, 5)
            with c2:
                hackathons_participated = st.number_input("Hackathons", 0, 20, 1)
                certifications_count = st.number_input("Certificaciones", 0, 20, 2)
                linkedin_activity_score = st.slider("Actividad LinkedIn", 0, 10, 5)
                github_score = st.slider("Actividad GitHub", 0, 10, 5)
                soft_skills_score = st.slider("Habilidades blandas", 0, 10, 5)
                interview_score = st.slider("Entrevista", 0, 10, 5)
                consistency_score = st.slider("Consistencia", 0, 10, 5)
                backlogs = st.number_input("Materias pendientes (backlogs)", 0, 20, 0)

            extracurricular = st.selectbox("Actividades extracurriculares", ["Yes", "No"])
            college_tier = st.selectbox("Categoría de la universidad", ["Tier 1", "Tier 2", "Tier 3"])
            placement_training = st.selectbox("Programa de formación laboral", ["Yes", "No"])

            submitted = st.form_submit_button("🔮 Predecir selección", use_container_width=True)

        if submitted:
            candidate = pd.DataFrame([{
                "CGPA": cgpa, "skills_score": skills_score, "projects_count": projects_count,
                "internships_done": internships_done, "communication_score": communication_score,
                "aptitude_score": aptitude_score, "coding_test_score": coding_test_score,
                "resume_score": resume_score, "hackathons_participated": hackathons_participated,
                "certifications_count": certifications_count, "linkedin_activity_score": linkedin_activity_score,
                "github_score": github_score, "soft_skills_score": soft_skills_score,
                "interview_score": interview_score, "consistency_score": consistency_score,
                "backlogs": backlogs, "extracurricular": extracurricular,
                "college_tier": college_tier, "placement_training": placement_training,
            }])

            prediction = model.predict(candidate)[0]
            probability = model.predict_proba(candidate)[0][1]

            if prediction == 1:
                st.markdown(f"""
                <div class="result-card" style="background-color:{SUCCESS_BG}; border-color:{SUCCESS_BORDER};">
                    <div class="result-title">✅ Seleccionado</div>
                    <div class="result-sub">Probabilidad de selección: {probability:.1%}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-card" style="background-color:{ERROR_BG}; border-color:{ERROR_BORDER};">
                    <div class="result-title">❌ No seleccionado</div>
                    <div class="result-sub">Probabilidad de selección: {probability:.1%}</div>
                </div>
                """, unsafe_allow_html=True)

            st.plotly_chart(probability_gauge(probability), use_container_width=True)

    with right:
        st.subheader("📊 Variables más influyentes")
        st.plotly_chart(feature_importance_chart(importance_df), use_container_width=True)

        st.subheader("📁 Vista rápida del dataset")
        st.dataframe(df.head(20), use_container_width=True, height=220)
        st.caption(f"{len(df):,} registros totales · {df[TARGET].mean():.0%} seleccionados en el histórico")


if __name__ == "__main__":
    main()
