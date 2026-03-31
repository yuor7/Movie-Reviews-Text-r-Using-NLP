# ============================================================
# Movie Review Text Classifier — Streamlit Web App
# NLP Project  |  21CSE356T  |  SRM Institute
# Run command : streamlit run app.py
# ============================================================

# pip install streamlit nltk scikit-learn pandas

import os
import re
import string
import streamlit as st
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import plotly.graph_objects as go
import plotly.express as px

# ─── Page Config ─────────────────────────────────────────────
st.set_page_config(
    page_title="Movie Review Classifier",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ──────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0D1B2A; color: #D8E8F5; }

    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #1A2E45; }

    /* Cards */
    .metric-card {
        background: #1A2E45;
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #1E3A55;
        text-align: center;
        margin-bottom: 10px;
    }
    .metric-card h2 { margin: 0; font-size: 2rem; }
    .metric-card p  { margin: 4px 0 0; color: #7FA9CC; font-size: 0.85rem; }

    /* Result box */
    .result-positive {
        background: linear-gradient(135deg, #0F6E56, #1A2E45);
        border-left: 5px solid #00C9A7;
        border-radius: 10px;
        padding: 20px;
        margin-top: 12px;
    }
    .result-negative {
        background: linear-gradient(135deg, #6B1E1E, #1A2E45);
        border-left: 5px solid #FC5C65;
        border-radius: 10px;
        padding: 20px;
        margin-top: 12px;
    }
    .result-neutral {
        background: linear-gradient(135deg, #5A4A00, #1A2E45);
        border-left: 5px solid #F7B731;
        border-radius: 10px;
        padding: 20px;
        margin-top: 12px;
    }

    /* Section headers */
    .section-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: #00C9A7;
        border-bottom: 2px solid #1E3A55;
        padding-bottom: 6px;
        margin-bottom: 14px;
    }

    /* Badge */
    .badge {
        display: inline-block;
        padding: 3px 12px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 700;
        margin: 3px;
    }

    /* Pipeline step */
    .pipeline-step {
        background: #1A2E45;
        border: 1px solid #1E3A55;
        border-radius: 8px;
        padding: 10px 16px;
        margin: 6px 0;
        font-size: 0.9rem;
    }

    /* Streamlit overrides */
    .stTextArea textarea {
        background-color: #1A2E45 !important;
        color: #D8E8F5 !important;
        border: 1px solid #1E3A55 !important;
        border-radius: 8px !important;
    }
    .stButton > button {
        background-color: #00C9A7 !important;
        color: #0D1B2A !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px 28px !important;
        font-size: 1rem !important;
        width: 100%;
    }
    .stButton > button:hover { background-color: #00a98a !important; }
    h1, h2, h3 { color: #D8E8F5 !important; }
    .stSelectbox label, .stTextArea label { color: #7FA9CC !important; }
</style>
""", unsafe_allow_html=True)

# ─── Download NLTK ───────────────────────────────────────────
@st.cache_resource
def download_nltk():
    nltk.download('punkt',     quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt_tab', quiet=True)

download_nltk()

# ─── Dataset ─────────────────────────────────────────────────
IMDB_CSV = "IMDB Dataset.csv"   # place this file in the same folder as app.py

FALLBACK_DATA = {
    'review': [
        # Positive (25)
        "This movie was absolutely fantastic and I loved every moment of it.",
        "Outstanding performance by all the actors. A must watch film.",
        "Brilliant storyline and amazing visuals. Highly recommended!",
        "One of the best movies I have ever seen in my entire life.",
        "The direction was superb and the background music was excellent.",
        "A wonderful cinematic experience that I will never forget.",
        "Great acting and a gripping storyline. Loved the ending too!",
        "A true masterpiece with breathtaking cinematography throughout.",
        "Perfect movie. The story touched my heart and made me emotional.",
        "Incredible film with outstanding and believable character development.",
        "Beautifully crafted film with a powerful and emotional storyline.",
        "The best film of the decade without any doubt. Truly amazing.",
        "Superb direction and a stellar cast made this a great experience.",
        "A heartwarming story that left me smiling throughout the movie.",
        "Phenomenal acting and a well-written script. Loved every scene.",
        "This movie exceeded all my expectations. Absolutely brilliant!",
        "A visual treat with a wonderful story. Highly enjoyable film.",
        "Loved the pace of the movie and the powerful performances given.",
        "An inspiring and uplifting story told with great skill and care.",
        "The cinematography and editing in this film were absolutely top notch.",
        "A classic film that everyone should watch at least once in life.",
        "Excellent movie with a strong message and great performances.",
        "The best acting I have seen this year. Simply outstanding work.",
        "A gripping thriller that kept me on the edge of my seat throughout.",
        "Perfect blend of emotion, drama and action. Truly loved this film.",
        # Negative (25)
        "This movie was absolutely terrible. A complete waste of my time.",
        "Awful plot with boring and completely unrelatable characters.",
        "I hated every minute of this movie. The story made no sense at all.",
        "Worst film I have ever seen in my entire life. Totally unwatchable.",
        "Poor acting and a completely dull and very predictable storyline.",
        "Very bad movie. Left the theater feeling angry and very cheated.",
        "Terrible script and horrible direction. An absolute disaster of a film.",
        "I fell asleep watching this. Extremely slow and incredibly boring.",
        "Disgusting film with absolutely no redeeming qualities at all.",
        "A complete disappointment from start to finish. Nothing worked here.",
        "The movie had no plot and the acting was painfully bad throughout.",
        "One of the worst films ever made. I want my money back now.",
        "Dreadful performances and a nonsensical confusing storyline throughout.",
        "A boring and pointless film that wasted two valuable hours of my life.",
        "The worst script I have ever heard in any movie. Just completely awful.",
        "Poorly directed with wooden acting and a terrible and unsatisfying ending.",
        "A terrible film that insults the intelligence of its audience completely.",
        "Nothing about this movie worked at all. The story was completely incoherent.",
        "Avoid this film at all costs. Absolutely dreadful viewing experience.",
        "The worst movie of the year by a very long margin. Simply horrible.",
        "A painfully bad movie with no story, no acting and no direction.",
        "I deeply regret watching this film. It was a complete waste of money.",
        "Horrendous film. The director clearly had absolutely no vision whatsoever.",
        "Boring, predictable, and badly acted throughout. A truly terrible film.",
        "The most disappointing movie I have seen in a very long time.",
        # Neutral (25)
        "The movie was okay. Not great but not terrible either I suppose.",
        "Average film. Some scenes were good while others were quite poor.",
        "It was a decent watch. Nothing particularly special about it though.",
        "The movie had its moments but overall it was just completely average.",
        "A fairly ordinary film. I expected more from this particular director.",
        "Mixed feelings about this one. It could have been so much better.",
        "It was alright. A one-time watch kind of movie at best honestly.",
        "Mediocre performances throughout. The film really does not stand out.",
        "Passable movie. Fine for a single casual viewing I suppose.",
        "Not bad, not great. Just a completely regular and forgettable film.",
        "The movie was fine but nothing about it was particularly memorable.",
        "An average film with some good ideas that were poorly executed.",
        "It was watchable but I would not really recommend it to anyone.",
        "Some good moments but too many flaws to truly enjoy the film.",
        "The movie started well but fell completely apart in the second half.",
        "Average at best. The story was predictable and entirely unoriginal.",
        "It was an okay film. Not something I would ever watch again though.",
        "Decent enough for a lazy evening watch. Nothing more than that really.",
        "The film had potential but ultimately failed to deliver on its promise.",
        "A middle-of-the-road movie. Neither particularly good nor very bad.",
        "Some scenes impressed me but the overall film was quite forgettable.",
        "An unremarkable film that neither entertained nor particularly bored me.",
        "The cast tried their best but the weak script really let them down.",
        "A mediocre effort overall. The movie was just about average at best.",
        "It passed the time but I would not call this a good movie at all.",
    ],
    'label': ['Positive']*25 + ['Negative']*25 + ['Neutral']*25
}

import pickle

# ─── Data Loading & Model Setup ──────────────────────────────
IMDB_CSV    = "IMDB Dataset.csv"
GZ_CSV      = "IMDB_Dataset_Full.csv.gz"  # GitHub Optimized Dataset (Compressed)
SAMPLE_CSV  = "sample_data.csv"
MODEL_PKL   = "model.pkl"
VECT_PKL    = "vectorizer.pkl"
METRICS_PKL = "metrics.pkl"

@st.cache_data
def load_data():
    """
    Tries to load the real IMDb dataset (CSV or Gzipped CSV).
    If not found, falls back to built-in 75 reviews.
    """
    source_file = None
    if os.path.exists(IMDB_CSV):
        source_file = IMDB_CSV
    elif os.path.exists(GZ_CSV):
        source_file = GZ_CSV
        
    if source_file:
        try:
            # Pandas automatically handles .gz decompression
            df_raw = pd.read_csv(source_file)
            # Use sentiment or label if present
            label_col = 'sentiment' if 'sentiment' in df_raw.columns else 'label'
            if 'review' not in df_raw.columns or label_col not in df_raw.columns:
                st.warning(f"File found but columns are wrong. Expected 'review' and '{label_col}'.")
                raise ValueError("Bad columns")

            # Sample substantially more data for robust accuracy
            pos_filter = (df_raw[label_col].str.lower() == 'positive') | (df_raw[label_col] == 'Positive')
            neg_filter = (df_raw[label_col].str.lower() == 'negative') | (df_raw[label_col] == 'Negative')
            
            df_pos = df_raw[pos_filter].sample(n=min(5000, pos_filter.sum()), random_state=42).copy()
            df_neg = df_raw[neg_filter].sample(n=min(5000, neg_filter.sum()), random_state=42).copy()

            # Create Neutral: reviews containing hedging language
            neutral_kw = ['okay','decent','average','fine','mixed','mediocre','ordinary','alright','passable','so-so','not bad','not great','fairly good']
            pattern    = '|'.join(neutral_kw)
            df_neutral = df_raw[df_raw['review'].str.lower().str.contains(pattern, na=False)].sample(n=min(2000, df_raw.shape[0]), random_state=42).copy()

            df_pos['label']     = 'Positive'
            df_neg['label']     = 'Negative'
            df_neutral['label'] = 'Neutral'

            df = pd.concat([df_pos[['review','label']], df_neg[['review','label']], df_neutral[['review','label']]], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
            source_name = "IMDb Gzipped (Compressed)" if ".gz" in source_file else "IMDb CSV (Original)"
            return df, f"{source_name} ({len(df):,} reviews)"
        except Exception as e:
            st.warning(f"Could not load dataset: {e} — using built-in reviews.")

    return pd.DataFrame(FALLBACK_DATA), "Built-in sample dataset (75 reviews)"


@st.cache_resource
def load_trained_model():
    """Loads pre-trained model and vectorizer if they exist."""
    if os.path.exists(MODEL_PKL) and os.path.exists(VECT_PKL) and os.path.exists(METRICS_PKL):
        try:
            m = pickle.load(open(MODEL_PKL, "rb"))
            v = pickle.load(open(VECT_PKL, "rb"))
            stats = pickle.load(open(METRICS_PKL, "rb"))
            return m, v, stats['accuracy'], stats['report'], stats['cm'], stats['total_reviews'], "Pre-trained Model (GitHub Optimized)"
        except Exception as e:
            st.error(f"Error loading pickle files: {e}")
    return None

@st.cache_data
def load_explorer_data():
    """Loads a portion of data for the explorer tab."""
    if os.path.exists(IMDB_CSV) or os.path.exists(GZ_CSV):
        df_exp, _ = load_data() 
        return df_exp
    elif os.path.exists(SAMPLE_CSV):
        return pd.read_csv(SAMPLE_CSV)
    else:
        return pd.DataFrame(FALLBACK_DATA)


# ─── Preprocessing ───────────────────────────────────────────
stemmer    = PorterStemmer()
negation_words = {'not', 'no', 'nor', 'neither', 'never', 'none', 'but', 'isnt', 'wasnt', 'werent', 'havent', 'hasnt', 'hadnt', 'wont', 'shouldnt', 'couldnt', 'mustnt', 'didnt', 'doesnt', 'arent'}
stop_words     = set(stopwords.words('english')) - negation_words

def preprocess_text(text):
    text   = re.sub(r'<.*?>', '', str(text))
    text   = text.lower()
    text   = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words and w.isalpha()]
    tokens = [stemmer.stem(w) for w in tokens]
    return ' '.join(tokens)

# ─── Main Initialization ─────────────────────────────────────
trained_bundle = load_trained_model()

if trained_bundle:
    model, vectorizer, accuracy, report, cm, total_reviews, data_source = trained_bundle
    df = load_explorer_data()
else:
    @st.cache_resource
    def build_classification_model():
        df_full, source = load_data()
        st.info("Training model on the fly... This might take a moment.")
        df_full['cleaned'] = df_full['review'].apply(preprocess_text)
        vec = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
        X = vec.fit_transform(df_full['cleaned'])
        y = df_full['label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        m = MultinomialNB(alpha=0.1)
        m.fit(X_train, y_train)
        y_pred = m.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        rep = classification_report(y_test, y_pred, output_dict=True)
        conf_m = confusion_matrix(y_test, y_pred, labels=['Positive','Negative','Neutral'])
        return m, vec, acc, rep, conf_m, df_full, source

    model, vectorizer, accuracy, report, cm, df, data_source = build_classification_model()
    total_reviews = len(df)



# ─── Sidebar ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎬 NLP Classifier")
    st.markdown("---")
    st.markdown("**Course:** 21CSE356T")
    st.markdown("**Dept:** Computational Intelligence")
    st.markdown("**Institute:** SRM IST")
    st.markdown("---")
    st.markdown("### 🔧 NLP Pipeline")
    steps = [
        ("1", "Lowercasing",       "#4F8EF7"),
        ("2", "Punctuation Removal","#4F8EF7"),
        ("3", "Tokenization",       "#00C9A7"),
        ("4", "Stopword Removal",   "#00C9A7"),
        ("5", "Stemming",           "#F7B731"),
        ("6", "TF-IDF Vectorizer",  "#F7B731"),
        ("7", "Naive Bayes",        "#FC5C65"),
    ]
    for num, step, col in steps:
        st.markdown(
            f'<div class="pipeline-step">'
            f'<span style="color:{col};font-weight:700">{num}.</span> {step}</div>',
            unsafe_allow_html=True
        )
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🏠 Home & Predict", "📊 Model Analytics", "📂 Dataset Explorer"]
    )

# ─── HOME PAGE ───────────────────────────────────────────────
if page == "🏠 Home & Predict":
    st.markdown("# 🎬 Movie Review Text Classifier")
    st.markdown(
        "An NLP-powered web app that classifies movie reviews as "
        "**Positive**, **Negative**, or **Neutral** using NLTK + Naive Bayes."
    )

    # Top metrics
    # Dataset source banner
    is_imdb     = "IMDb" in data_source
    is_pretrained = "Pre-trained" in data_source
    
    banner_col  = "#00C9A7" if (is_imdb or is_pretrained) else "#F7B731"
    banner_icon = "✅" if (is_imdb or is_pretrained) else "⚠️"
    
    if is_pretrained:
        banner_msg = f"{banner_icon} <strong>GitHub Optimized</strong> — Loaded pre-trained model & analytics (Dataset.csv not required)."
    elif is_imdb:
        banner_msg = f"{banner_icon} Using real-world <strong>IMDb dataset</strong> — {data_source}"
    else:
        banner_msg = (
            f"{banner_icon} <strong>IMDB Dataset.csv not found</strong> — using built-in 75-review dataset. "
            f"Place <code>IMDB Dataset.csv</code> in the same folder or run <code>train_model.py</code> to generate a better model."
        )

    st.markdown(
        f'<div style="background:#1A2E45;border:1px solid {banner_col};border-radius:8px;'
        f'padding:10px 16px;margin-bottom:16px;font-size:0.9rem;color:#D8E8F5">'
        f'{banner_msg}</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f'<div class="metric-card"><h2 style="color:#00C9A7">'
            f'{accuracy*100:.1f}%</h2><p>Model Accuracy</p></div>',
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            f'<div class="metric-card"><h2 style="color:#4F8EF7">'
            f'{total_reviews:,}</h2><p>Training Reviews</p></div>',
            unsafe_allow_html=True
        )
    with col3:
        st.markdown(
            '<div class="metric-card"><h2 style="color:#F7B731">3</h2>'
            '<p>Sentiment Classes</p></div>',
            unsafe_allow_html=True
        )
    with col4:
        st.markdown(
            '<div class="metric-card"><h2 style="color:#FC5C65">5000</h2>'
            '<p>TF-IDF Features</p></div>',
            unsafe_allow_html=True
        )

    st.markdown("---")

    # Predict section
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown('<div class="section-header">✍️ Enter Movie Review</div>', unsafe_allow_html=True)
        # Initialize focus text if not present
        if 'example_text' not in st.session_state:
            st.session_state['example_text'] = ""

        user_input = st.text_area(
            "Type or paste a movie review below:",
            value=st.session_state['example_text'],
            placeholder="e.g. This movie was absolutely fantastic! I loved every moment of it...",
            height=160,
            label_visibility="collapsed"
        )

        # Quick examples
        st.markdown("**Quick Examples:**")
        ex_cols = st.columns(3)
        examples = [
            ("😊 Positive", "This film was absolutely brilliant and I loved every moment of it!"),
            ("😠 Negative", "Terrible movie, worst I have seen. Complete waste of time and money."),
            ("😐 Neutral",  "The movie was okay. Not great but not too bad either I suppose."),
        ]
        for i, (label, ex) in enumerate(examples):
            with ex_cols[i]:
                if st.button(label, key=f"ex_{i}"):
                    st.session_state['example_text'] = ex
                    st.rerun()

        predict_btn = st.button("🔍 Classify Review")

    with col_right:
        st.markdown('<div class="section-header">📋 How It Works</div>', unsafe_allow_html=True)
        st.markdown("""
        1. **Input** → Raw movie review text
        2. **Clean** → Remove HTML, punctuation
        3. **Tokenize** → Split into words
        4. **Filter** → Remove stopwords
        5. **Stem** → Reduce to root form
        6. **Vectorize** → TF-IDF features
        7. **Classify** → Naive Bayes model
        8. **Output** → Positive / Negative / Neutral
        """)

    # Prediction result
    if user_input.strip():
        cleaned    = preprocess_text(user_input)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        proba      = model.predict_proba(vectorized)[0]
        classes    = model.classes_

        emoji_map = {"Positive": "😊", "Negative": "😠", "Neutral": "😐"}
        color_map = {"Positive": "result-positive", "Negative": "result-negative", "Neutral": "result-neutral"}
        accent_map= {"Positive": "#00C9A7", "Negative": "#FC5C65", "Neutral": "#F7B731"}
        emoji     = emoji_map[prediction]
        css_class = color_map[prediction]
        accent    = accent_map[prediction]
        conf      = max(proba) * 100

        st.markdown(
            f'<div class="{css_class}">'
            f'<h2 style="margin:0;color:{accent}">{emoji} {prediction}</h2>'
            f'<p style="margin:6px 0 0;color:#D8E8F5">Confidence: <strong>{conf:.1f}%</strong></p>'
            f'</div>',
            unsafe_allow_html=True
        )

        # Confidence bar chart
        fig = go.Figure(go.Bar(
            x=[p*100 for p in proba],
            y=list(classes),
            orientation='h',
            marker_color=["#00C9A7", "#FC5C65", "#F7B731"],
            text=[f"{p*100:.1f}%" for p in proba],
            textposition='outside'
        ))
        fig.update_layout(
            plot_bgcolor='#1A2E45',
            paper_bgcolor='#1A2E45',
            font_color='#D8E8F5',
            height=180,
            margin=dict(l=10, r=40, t=20, b=10),
            xaxis=dict(range=[0,110], showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False)
        )
        st.plotly_chart(fig, use_container_width=True)

        # Preprocessing steps shown
        with st.expander("🔬 View Preprocessing Steps"):
            st.markdown(f"**Original:** {user_input[:100]}...")
            lower   = user_input.lower()
            no_punc = lower.translate(str.maketrans('', '', string.punctuation))
            tokens  = word_tokenize(no_punc)
            filtered= [w for w in tokens if w not in stop_words and w.isalpha()]
            stemmed = [stemmer.stem(w) for w in filtered]
            st.markdown(f"**After lowercase:** `{lower[:80]}...`")
            st.markdown(f"**After tokenization:** `{tokens[:8]}...`")
            st.markdown(f"**After stopword removal:** `{filtered[:8]}...`")
            st.markdown(f"**After stemming:** `{stemmed[:8]}...`")

    elif predict_btn:
        st.warning("Please enter a review before clicking Classify.")

# ─── ANALYTICS PAGE ──────────────────────────────────────────
elif page == "📊 Model Analytics":
    st.markdown("# 📊 Model Analytics")
    st.markdown("Detailed evaluation metrics for the trained Naive Bayes classifier.")

    # Accuracy metrics
    col1, col2, col3 = st.columns(3)
    classes = ['Positive', 'Negative', 'Neutral']
    colors_ = ['#00C9A7', '#FC5C65', '#F7B731']

    for i, (cls, col, c) in enumerate(zip(classes, [col1, col2, col3], colors_)):
        prec = report[cls]['precision'] * 100
        rec  = report[cls]['recall'] * 100
        f1   = report[cls]['f1-score'] * 100
        with col:
            st.markdown(
                f'<div class="metric-card">'
                f'<h2 style="color:{c}">{cls}</h2>'
                f'<p>Precision: <strong style="color:{c}">{prec:.0f}%</strong></p>'
                f'<p>Recall: <strong style="color:{c}">{rec:.0f}%</strong></p>'
                f'<p>F1-Score: <strong style="color:{c}">{f1:.0f}%</strong></p>'
                f'</div>',
                unsafe_allow_html=True
            )

    st.markdown("---")
    col_a, col_b = st.columns(2)

    # Confusion Matrix
    with col_a:
        st.markdown('<div class="section-header">🔲 Confusion Matrix</div>', unsafe_allow_html=True)
        fig_cm = px.imshow(
            cm,
            labels=dict(x="Predicted", y="Actual"),
            x=classes, y=classes,
            color_continuous_scale=[[0,"#1A2E45"],[1,"#00C9A7"]],
            text_auto=True
        )
        fig_cm.update_layout(
            paper_bgcolor='#1A2E45',
            plot_bgcolor='#1A2E45',
            font_color='#D8E8F5',
            height=320,
            margin=dict(l=10,r=10,t=20,b=10)
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    # Class distribution
    with col_b:
        st.markdown('<div class="section-header">📊 Dataset Distribution</div>', unsafe_allow_html=True)
        counts = df['label'].value_counts()
        fig_pie = go.Figure(go.Pie(
            labels=counts.index,
            values=counts.values,
            hole=0.5,
            marker_colors=['#00C9A7', '#FC5C65', '#F7B731'],
            textinfo='label+percent'
        ))
        fig_pie.update_layout(
            paper_bgcolor='#1A2E45',
            font_color='#D8E8F5',
            height=320,
            margin=dict(l=10,r=10,t=20,b=10),
            showlegend=False
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # F1 Score bar
    st.markdown('<div class="section-header">📈 F1-Score per Class</div>', unsafe_allow_html=True)
    f1_vals = [report[cls]['f1-score']*100 for cls in classes]
    fig_f1 = go.Figure(go.Bar(
        x=classes, y=f1_vals,
        marker_color=['#00C9A7', '#FC5C65', '#F7B731'],
        text=[f"{v:.1f}%" for v in f1_vals],
        textposition='outside'
    ))
    fig_f1.update_layout(
        plot_bgcolor='#1A2E45',
        paper_bgcolor='#1A2E45',
        font_color='#D8E8F5',
        height=260,
        margin=dict(l=10,r=10,t=20,b=10),
        yaxis=dict(range=[0,115], showgrid=False),
        xaxis=dict(showgrid=False)
    )
    st.plotly_chart(fig_f1, use_container_width=True)

# ─── DATASET PAGE ────────────────────────────────────────────
elif page == "📂 Dataset Explorer":
    st.markdown("# 📂 Dataset Explorer")
    st.markdown("Browse the movie reviews used to train the classifier.")

    col1, col2 = st.columns([1, 2])
    with col1:
        filter_label = st.selectbox(
            "Filter by Sentiment",
            ["All", "Positive", "Negative", "Neutral"]
        )
    with col2:
        search_term = st.text_input("Search reviews", placeholder="Type a keyword...")

    filtered_df = df.copy()
    if filter_label != "All":
        filtered_df = filtered_df[filtered_df['label'] == filter_label]
    if search_term:
        filtered_df = filtered_df[
            filtered_df['review'].str.lower().str.contains(search_term.lower(), na=False)
        ]

    st.markdown(f"**Showing {len(filtered_df)} reviews**")
    st.markdown("---")

    color_map = {"Positive": "#00C9A7", "Negative": "#FC5C65", "Neutral": "#F7B731"}
    for _, row in filtered_df.iterrows():
        col = color_map[row['label']]
        st.markdown(
            f'<div style="background:#1A2E45;border-left:4px solid {col};'
            f'border-radius:8px;padding:12px 16px;margin-bottom:8px;">'
            f'<span style="color:{col};font-weight:700;font-size:0.8rem">{row["label"].upper()}</span>'
            f'<p style="margin:6px 0 0;color:#D8E8F5;font-size:0.9rem">{row["review"]}</p>'
            f'</div>',
            unsafe_allow_html=True
        )
