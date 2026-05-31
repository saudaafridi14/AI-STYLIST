import os
import json
import numpy as np
import cv2
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="AI STYLIST - Smart Styling Assistant",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,600;1,300;1,400&family=Jost:wght@300;400;500;600&display=swap');

    /* ── ROOT PALETTE ── */
    :root {
        --cream:    #f8f4ef;
        --parchment:#ede8e1;
        --blush:    #e8cfc6;
        --rose:     #c9877a;
        --burgundy: #7d3c3c;
        --espresso: #2b1f1f;
        --gold:     #c9a96e;
        --gold-lt:  #e8d5b0;
        --mink:     #9c8680;
        --sage:     #8da89a;
        --ink:      #1a1410;
    }

    /* ── GLOBAL RESET ── */
    html, body, [class*="css"] {
        font-family: 'Jost', sans-serif;
        color: var(--espresso);
    }

    /* ── PAGE BACKGROUND ── */
    .stApp {
        background-color: var(--cream);
        background-image:
            radial-gradient(ellipse 80% 50% at 10% -10%, rgba(201,169,110,0.12) 0%, transparent 60%),
            radial-gradient(ellipse 60% 40% at 90% 110%, rgba(201,135,122,0.10) 0%, transparent 55%),
            url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23c9a96e' fill-opacity='0.04'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
    }

    /* ── SIDEBAR ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, var(--espresso) 0%, #3d2020 100%);
        border-right: 1px solid rgba(201,169,110,0.2);
    }
    [data-testid="stSidebar"] * {
        color: var(--gold-lt) !important;
        font-family: 'Jost', sans-serif !important;
    }
    [data-testid="stSidebar"] h2 {
        color: var(--gold) !important;
        font-family: 'Cormorant Garamond', serif !important;
        font-size: 1.4rem !important;
        letter-spacing: 0.08em;
        border-bottom: 1px solid rgba(201,169,110,0.3);
        padding-bottom: 12px;
        margin-bottom: 16px !important;
    }
    [data-testid="stSidebar"] p {
        font-size: 0.82rem !important;
        line-height: 1.7 !important;
        color: rgba(232,213,176,0.75) !important;
    }
    [data-testid="stSidebar"] strong {
        color: var(--gold) !important;
        font-weight: 600 !important;
    }

    /* ── HERO HEADER ── */
    .hero-wrap {
        text-align: center;
        padding: 52px 20px 36px;
        position: relative;
    }
    .hero-eyebrow {
        font-family: 'Jost', sans-serif;
        font-weight: 400;
        font-size: 0.72rem;
        letter-spacing: 0.35em;
        text-transform: uppercase;
        color: var(--gold);
        margin-bottom: 14px;
    }
    .hero-title {
        font-family: 'Cormorant Garamond', serif;
        font-weight: 300;
        font-size: clamp(3rem, 7vw, 5.5rem);
        line-height: 1;
        color: var(--espresso);
        letter-spacing: -0.01em;
        margin: 0 0 6px;
    }
    .hero-title em {
        font-style: italic;
        color: var(--rose);
    }
    .hero-rule {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 14px;
        margin: 18px 0 16px;
    }
    .hero-rule-line {
        width: 60px;
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--gold));
    }
    .hero-rule-line.right {
        background: linear-gradient(90deg, var(--gold), transparent);
    }
    .hero-rule-diamond {
        width: 6px; height: 6px;
        background: var(--gold);
        transform: rotate(45deg);
    }
    .hero-sub {
        font-family: 'Jost', sans-serif;
        font-weight: 300;
        font-size: 0.95rem;
        letter-spacing: 0.12em;
        color: var(--mink);
        text-transform: uppercase;
    }

    /* ── CARDS ── */
    .card {
        background: rgba(255,255,255,0.82);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(201,169,110,0.18);
        border-radius: 2px;
        box-shadow: 0 2px 24px rgba(43,31,31,0.06), 0 0 0 0.5px rgba(201,169,110,0.1);
        padding: 28px 28px 24px;
        margin-bottom: 20px;
        position: relative;
        overflow: hidden;
    }
    .card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, var(--gold), var(--rose), var(--gold));
        opacity: 0.7;
    }

    /* ── SECTION LABELS ── */
    .card h2, .card h3, [data-testid="stMarkdownContainer"] h2,
    [data-testid="stMarkdownContainer"] h3 {
        font-family: 'Cormorant Garamond', serif !important;
        font-weight: 400 !important;
        letter-spacing: 0.04em !important;
        color: var(--espresso) !important;
    }

    /* ── ATTRIBUTE BADGES ── */
    .badge {
        display: inline-block;
        padding: 5px 14px;
        font-family: 'Jost', sans-serif;
        font-size: 0.72rem;
        font-weight: 500;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        border-radius: 0;
        margin: 0 6px 6px 0;
        color: white;
        border: none;
    }
    .badge-style {
        background: linear-gradient(135deg, var(--burgundy), #a04040);
        box-shadow: 0 2px 8px rgba(125,60,60,0.3);
    }
    .badge-pattern {
        background: linear-gradient(135deg, #4a6070, #6a8090);
        box-shadow: 0 2px 8px rgba(74,96,112,0.3);
    }
    .badge-color {
        background: linear-gradient(135deg, var(--sage), #6d8f7e);
        box-shadow: 0 2px 8px rgba(141,168,154,0.35);
    }

    /* ── PROGRESS BARS ── */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, var(--rose), var(--gold)) !important;
        border-radius: 0 !important;
    }
    .stProgress > div > div > div {
        background: var(--parchment) !important;
        border-radius: 0 !important;
        height: 5px !important;
    }

    /* ── TABS ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        border-bottom: 1px solid var(--parchment);
        background: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'Jost', sans-serif;
        font-size: 0.75rem;
        font-weight: 500;
        letter-spacing: 0.2em;
        text-transform: uppercase;
        color: var(--mink);
        padding: 10px 24px;
        border: none;
        background: transparent;
        border-bottom: 2px solid transparent;
        margin-bottom: -1px;
    }
    .stTabs [aria-selected="true"] {
        color: var(--espresso) !important;
        border-bottom: 2px solid var(--gold) !important;
        background: transparent !important;
    }
    .stTabs [data-baseweb="tab-highlight"] {
        display: none !important;
    }

    /* ── FILE UPLOADER ── */
    [data-testid="stFileUploader"] {
        border: 1px dashed rgba(201,169,110,0.5) !important;
        border-radius: 2px !important;
        background: rgba(248,244,239,0.6) !important;
        padding: 24px !important;
    }
    [data-testid="stFileUploader"] label {
        font-family: 'Jost', sans-serif !important;
        letter-spacing: 0.05em !important;
        color: var(--mink) !important;
    }

    /* ── TEXT INPUT ── */
    .stTextInput > div > div > input {
        font-family: 'Jost', sans-serif !important;
        border: 1px solid rgba(201,169,110,0.35) !important;
        border-radius: 0 !important;
        background: rgba(255,255,255,0.7) !important;
        color: var(--espresso) !important;
        letter-spacing: 0.03em !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: var(--gold) !important;
        box-shadow: 0 0 0 2px rgba(201,169,110,0.12) !important;
    }

    /* ── MARKDOWN BODY TEXT ── */
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li {
        font-family: 'Jost', sans-serif;
        font-size: 0.9rem;
        line-height: 1.75;
        color: #3d2c2c;
    }
    [data-testid="stMarkdownContainer"] strong {
        font-weight: 600;
        color: var(--espresso);
    }
    [data-testid="stMarkdownContainer"] hr {
        border-color: rgba(201,169,110,0.2) !important;
    }

    /* ── DIVIDERS ── */
    hr {
        border: none !important;
        border-top: 1px solid rgba(201,169,110,0.2) !important;
        margin: 16px 0 !important;
    }

    /* ── SPINNER ── */
    .stSpinner > div {
        border-top-color: var(--rose) !important;
    }

    /* ── ALERTS / INFO ── */
    .stAlert {
        border-radius: 2px !important;
        border-left: 3px solid var(--gold) !important;
        font-family: 'Jost', sans-serif !important;
    }

    /* ── RECOMMENDATION ITEMS ── */
    .rec-item {
        display: flex;
        gap: 14px;
        align-items: flex-start;
        padding: 14px 0;
        border-bottom: 1px solid rgba(201,169,110,0.15);
    }
    .rec-item:last-child { border-bottom: none; }
    .rec-icon {
        font-size: 1.2rem;
        line-height: 1.5;
        flex-shrink: 0;
        width: 28px;
        text-align: center;
    }
    .rec-label {
        font-family: 'Jost', sans-serif;
        font-size: 0.68rem;
        font-weight: 600;
        letter-spacing: 0.22em;
        text-transform: uppercase;
        color: var(--gold);
        margin-bottom: 3px;
    }
    .rec-value {
        font-family: 'Jost', sans-serif;
        font-size: 0.88rem;
        color: var(--espresso);
        line-height: 1.55;
    }

    /* ── COLOR SWATCH WRAPPER ── */
    .swatch-grid {
        display: flex;
        gap: 10px;
        margin-top: 14px;
    }
    .swatch-cell {
        flex: 1;
        text-align: center;
    }
    .swatch-block {
        height: 44px;
        border-radius: 1px;
        border: 1px solid rgba(0,0,0,0.06);
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        margin-bottom: 6px;
    }
    .swatch-hex {
        font-family: 'Jost', sans-serif;
        font-size: 0.68rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        color: var(--mink);
    }
    .swatch-pct {
        font-family: 'Jost', sans-serif;
        font-size: 0.63rem;
        color: #b5a9a3;
    }

    /* ── CONFIDENCE ROW ── */
    .conf-label {
        font-family: 'Jost', sans-serif;
        font-size: 0.68rem;
        font-weight: 600;
        letter-spacing: 0.2em;
        text-transform: uppercase;
        color: var(--mink);
        margin-bottom: 4px;
        margin-top: 14px;
    }
    .conf-value {
        font-family: 'Cormorant Garamond', serif;
        font-size: 1.05rem;
        color: var(--espresso);
        margin-bottom: 4px;
    }

    /* ── CAMERA INPUT ── */
    [data-testid="stCameraInput"] {
        border-radius: 2px !important;
    }

    /* ── SCROLLBAR ── */
    ::-webkit-scrollbar { width: 5px; }
    ::-webkit-scrollbar-track { background: var(--parchment); }
    ::-webkit-scrollbar-thumb { background: var(--blush); border-radius: 0; }

    /* ── IMAGE DISPLAY ── */
    [data-testid="stImage"] img {
        border-radius: 1px !important;
        box-shadow: 0 4px 20px rgba(43,31,31,0.12) !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------
# Paths
# ---------------------
MODEL_PATH = os.path.join("models", "fashion_multitask_model.h5")
LABELS_PATH = os.path.join("models", "multitask_labels.json")
IMG_SIZE = (224, 224)

# ---------------------
# Caching Model & Labels
# ---------------------
@st.cache_resource
def load_fashion_model():
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_data
def load_labels_metadata():
    if not os.path.exists(LABELS_PATH):
        return None
    try:
        with open(LABELS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

model = load_fashion_model()
metadata = load_labels_metadata()

# ---------------------
# Helper Functions
# ---------------------
def preprocess_uploaded_image(pil_img):
    img = np.array(pil_img.convert("RGB"))
    img_resized = cv2.resize(img, IMG_SIZE)
    img_resized = img_resized.astype("float32")
    img_preprocessed = preprocess_input(img_resized)
    img_batch = np.expand_dims(img_preprocessed, axis=0)
    return img_batch

def get_dominant_colors_swatch(pil_img, k=5):
    img = np.array(pil_img.convert("RGB"))
    h, w, _ = img.shape
    cy_start, cy_end = int(h * 0.2), int(h * 0.8)
    cx_start, cx_end = int(w * 0.25), int(w * 0.75)
    crop = img[cy_start:cy_end, cx_start:cx_end]
    crop_resized = cv2.resize(crop, (100, 100))
    pixels = crop_resized.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 15, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, flags)
    counts = np.bincount(labels.flatten())
    percentages = counts / len(labels)
    sorted_indices = np.argsort(percentages)[::-1]
    sorted_centers = centers[sorted_indices].astype(int)
    sorted_percentages = percentages[sorted_indices]
    return sorted_centers, sorted_percentages

def get_styling_recommendations(style, pattern, color):
    recommendations = {
        "co-ord eastern sets": {
            "shoes": "Nude block heels, traditional metallic silver/gold Khussas, or chic flat slides.",
            "bottoms": "Matching co-ord trousers (already included in the set!).",
            "accessories": "Minimalist silver hoop earrings, a sleek metallic watch, or a delicate neck chain.",
            "dupatta": "A sheer organza or chiffon dupatta in a matching tone or white/cream color.",
            "occasions": "Daytime social gatherings, festive brunches, or semi-formal meetups."
        },
        "frock": {
            "shoes": "Strappy sandals, juttis, or platform heels.",
            "bottoms": "Chooridar pajama, straight trousers, or ankle-length leggings depending on the frock length.",
            "accessories": "Traditional jhumkas, glass bangles matching the dominant color, and a statement ring.",
            "dupatta": "Crushed chiffon dupatta or embroidered net dupatta with matching borders.",
            "occasions": "Festive Eid functions, family dinners, weddings, or formal celebrations."
        },
        "fusion wear women": {
            "shoes": "Contemporary mules, ankle boots, or strappy stiletto sandals.",
            "bottoms": "Cigarette pants, palazzo trousers, or premium denim jeans.",
            "accessories": "Modern statement earrings, layered metallic necklaces, or a leather belt to cinch the waist.",
            "dupatta": "Usually worn without a dupatta, or paired with a modern light scarf/stole.",
            "occasions": "Evening parties, dinners, or modern festive events."
        },
        "khaddar dresses": {
            "shoes": "Loafers, leather flats, or traditional Peshawari chappals/khussas.",
            "bottoms": "Warm matching trousers, straight-cut pants, or shalwars.",
            "accessories": "Chunky silver jewelry, terracotta style earrings, or standard studs.",
            "dupatta": "A warm matching wool/khaddar shawl or printed cotton dupatta.",
            "occasions": "Casual winter outings, office wear, or daily run-arounds."
        },
        "long shirts with cigarette pants": {
            "shoes": "Pointed-toe heels, structured mules, or elegant sandals.",
            "bottoms": "Cigarette pants or straight trousers with bottom lace detailing.",
            "accessories": "Pearl earrings, a structured handbag, and a classic metal strap watch.",
            "dupatta": "Medium-weight lawn or silk printed dupatta draped elegantly on one shoulder.",
            "occasions": "Office wear, formal client meetings, or professional gatherings."
        },
        "office wear": {
            "shoes": "Pointed pumps, formal loafers, or closed-toe mules.",
            "bottoms": "Tailored trousers, pencil skirts, or wide-leg trousers.",
            "accessories": "Minimalist pearl/gold studs, a leather belt, and a professional laptop bag.",
            "dupatta": "Usually not applicable; pair with structured blazer or smart trench coat.",
            "occasions": "Corporate settings, professional presentations, job interviews, or client meetings."
        }
    }

    rec = recommendations.get(style.lower(), {
        "shoes": "Universal sandals or elegant pumps.",
        "bottoms": "Neutral tailored trousers.",
        "accessories": "Minimalist classic jewelry.",
        "dupatta": "Solid chiffon dupatta.",
        "occasions": "General smart casual settings."
    })

    pattern_tips = ""
    if pattern == "Printed":
        pattern_tips = "Since the outfit features a **Printed** design, keep accessories solid and minimal to let the print stand out."
    elif pattern == "Embroidered":
        pattern_tips = "The **Embroidered** details add a festive, rich texture. Highlight the embroidery by matching your jewelry or shoes to the embroidery thread color."
    else:
        pattern_tips = "This **Solid** color dress is a perfect canvas for playing with statement accessories. You can easily introduce contrast using a printed dupatta or colorful jewelry."

    color_tips = f"The dominant **{color}** color family pairs beautifully with "
    if color in ["White", "Black", "Grey"]:
        color_tips += "bold contrasting accent colors (like Red, Pink, or Emerald Green) or classic metallic gold and silver."
    elif color in ["Pink", "Red", "Purple"]:
        color_tips += "neutral colors like cream, beige, or gold to keep the look sophisticated yet vibrant."
    elif color in ["Blue", "Green"]:
        color_tips += "silver accessories, white trousers, or earth-tone bags for a balanced color palette."
    else:
        color_tips += "white, cream, or brown tones for a clean, harmonious look."

    return rec, pattern_tips, color_tips

# ---------------------
# Application UI
# ---------------------

# Hero Header
st.markdown("""
<div class="hero-wrap">
    <div class="hero-eyebrow">✦ Powered by Deep Learning ✦</div>
    <h1 class="hero-title">AI <em>Stylist</em></h1>
    <div class="hero-rule">
        <div class="hero-rule-line"></div>
        <div class="hero-rule-diamond"></div>
        <div class="hero-rule-line right"></div>
    </div>
    <p class="hero-sub">Multi‑Task Outfit Classification &amp; Styling Intelligence</p>
</div>
""", unsafe_allow_html=True)

# Check if model exists
if model is None or metadata is None:
    st.warning("⚠️ Multi-task model or label mappings not found. Please train the model first by running `prepare_multitask_data.py` and then `train_multitask.py`.")
    st.info("You can run training in your terminal or wait for the process to complete.")
else:
    # Sidebar options
    st.sidebar.header("Configuration")
    st.sidebar.write("Model loaded successfully!")
    st.sidebar.markdown("""
    **Model Architecture:** 
    MobileNetV2 + Multi-Output Heads
    
    **Classification Attributes:**
    1. **Style Category**
    2. **Pattern Type**
    3. **Dominant Color Family**
    """)

    # Upload and Camera input tabs
    tab1, tab2 = st.tabs(["📤  Upload Image", "📸  Take Photo"])

    image = None
    with tab1:
        uploaded_file = st.file_uploader("Upload an image of an outfit...", type=["jpg", "jpeg", "png", "webp"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)

    with tab2:
        camera_file = st.camera_input("Take a photo of the outfit...")
        if camera_file is not None:
            image = Image.open(camera_file)

    test_path = st.text_input("Or enter local image path for testing:")
    if test_path and os.path.exists(test_path):
        try:
            image = Image.open(test_path)
        except Exception as e:
            st.error(f"Error opening image: {e}")

    if image is not None:
        col1, col2 = st.columns([1, 1.2], gap="large")

        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("🖼️ Input Image")
            st.image(image, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # Color Palette Extraction
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("🎨 Extracted Color Palette")
            st.write("Dominant colors extracted from the clothing area using K-Means:")

            try:
                colors, percentages = get_dominant_colors_swatch(image, k=5)
                swatch_html = '<div class="swatch-grid">'
                for rgb, pct in zip(colors, percentages):
                    hex_color = '#{:02x}{:02x}{:02x}'.format(rgb[0], rgb[1], rgb[2])
                    swatch_html += f'''
                    <div class="swatch-cell">
                        <div class="swatch-block" style="background-color:{hex_color};"></div>
                        <div class="swatch-hex">{hex_color.upper()}</div>
                        <div class="swatch-pct">{pct*100:.1f}%</div>
                    </div>'''
                swatch_html += '</div>'
                st.markdown(swatch_html, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Could not extract color palette: {e}")
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("📊 Multi-Task Attributes Analysis")

            processed_img = preprocess_uploaded_image(image)

            with st.spinner("Analysing outfit attributes…"):
                preds = model.predict(processed_img)

            style_preds, pattern_preds, color_preds = preds

            style_idx   = np.argmax(style_preds[0])
            style_conf  = style_preds[0][style_idx] * 100
            pred_style  = metadata["styles"][style_idx]

            pattern_idx   = np.argmax(pattern_preds[0])
            pattern_conf  = pattern_preds[0][pattern_idx] * 100
            pred_pattern  = metadata["patterns"][pattern_idx]

            color_idx   = np.argmax(color_preds[0])
            color_conf  = color_preds[0][color_idx] * 100
            pred_color  = metadata["colors"][color_idx]

            # Badges
            badge_html = f"""
            <div style="margin-bottom:22px; display:flex; flex-wrap:wrap; gap:6px;">
                <span class="badge badge-style">Style: {pred_style}</span>
                <span class="badge badge-pattern">Pattern: {pred_pattern}</span>
                <span class="badge badge-color">Color: {pred_color}</span>
            </div>"""
            st.markdown(badge_html, unsafe_allow_html=True)

            # Confidence rows
            st.markdown(f'<div class="conf-label">Style Category</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="conf-value">🏷️ {pred_style} &nbsp;<span style="font-family:Jost,sans-serif;font-size:0.78rem;color:#9c8680;">({style_conf:.1f}%)</span></div>', unsafe_allow_html=True)
            st.progress(int(style_conf))

            st.markdown(f'<div class="conf-label" style="margin-top:16px;">Pattern Type</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="conf-value">👕 {pred_pattern} &nbsp;<span style="font-family:Jost,sans-serif;font-size:0.78rem;color:#9c8680;">({pattern_conf:.1f}%)</span></div>', unsafe_allow_html=True)
            st.progress(int(pattern_conf))

            st.markdown(f'<div class="conf-label" style="margin-top:16px;">Dominant Color Family</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="conf-value">🎨 {pred_color} &nbsp;<span style="font-family:Jost,sans-serif;font-size:0.78rem;color:#9c8680;">({color_conf:.1f}%)</span></div>', unsafe_allow_html=True)
            st.progress(int(color_conf))

            st.markdown('</div>', unsafe_allow_html=True)

            # Styling Recommendations Card
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("💡 AI Styling Assistant Suggestions")

            rec, pattern_tip, color_tip = get_styling_recommendations(pred_style, pred_pattern, pred_color)

            st.markdown(f"**🎨 Color Styling Advice:** {color_tip}")
            st.markdown(f"**👕 Pattern Advice:** {pattern_tip}")

            # Elegant recommendation list
            items = [
                ("👠", "Best Footwear Match",      rec['shoes']),
                ("👖", "Trouser Coordination",     rec['bottoms']),
                ("🧣", "Dupatta / Scarf",          rec['dupatta']),
                ("💍", "Accessories Guide",         rec['accessories']),
                ("📅", "Ideal Occasions",           rec['occasions']),
            ]
            rec_html = '<div style="margin-top:18px;">'
            for icon, label, value in items:
                rec_html += f"""
                <div class="rec-item">
                    <div class="rec-icon">{icon}</div>
                    <div>
                        <div class="rec-label">{label}</div>
                        <div class="rec-value">{value}</div>
                    </div>
                </div>"""
            rec_html += '</div>'
            st.markdown(rec_html, unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)