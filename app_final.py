import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
import os
import librosa
import soundfile as sf
import io
import json
import requests
import base64
import streamlit.components.v1 as components
from streamlit_lottie import st_lottie
from supabase import create_client, Client

st.set_page_config(page_title="Parkinson Disease Detection using AI & ML", layout="wide", initial_sidebar_state="expanded")

# --- Database Connection ---
@st.cache_resource
def init_connection() -> Client:
    url = st.secrets["supabase"]["URL"]
    key = st.secrets["supabase"]["KEY"]
    return create_client(url, key)

try:
    supabase = init_connection()
except Exception as e:
    st.error(f"Database connection error. Have you set up your secrets.toml file? Error: {e}")
    supabase = None

# --- Utilities ---
@st.cache_data
def load_lottieurl(url: str):
    try:
        r = requests.get(url, timeout=5)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

@st.cache_data
def get_video_base64(path):
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

def set_local_video_background(video_path, color_mode="blue", theme="Dark"):
    """
    Injects a native HTML5 video element using a base64 encoded local file.
    This GUARANTEES zero UI controls (no YouTube API involved) while maintaining
    the perfect looping, responsive background effect.
    """
    b64_video = get_video_base64(video_path)
    if not b64_video:
        st.error(f"Video file not found at {video_path}")
        return

    # Tinting logic
    if color_mode == "red":
        hue_filter = "sepia(100%) hue-rotate(320deg) saturate(500%)"
    elif color_mode == "purple":
        hue_filter = "sepia(100%) hue-rotate(240deg) saturate(300%)"
    else: # blue/default
        hue_filter = "sepia(100%) hue-rotate(180deg) saturate(300%)"
        
    opacity = "0.75" if theme in ["Dark", "Cyber"] else "0.35"

    js = f"""
    <script>
    const parentDoc = window.parent.document;
    
    // Ensure Streamlit background is completely transparent
    const styleId = 'custom-transparent-bg';
    if (!parentDoc.getElementById(styleId)) {{
        const style = parentDoc.createElement('style');
        style.id = styleId;
        style.innerHTML = `
            .stApp {{ background: transparent !important; background-color: transparent !important; background-image: none !important; overflow-x: hidden; }}
            .local-video-bg {{
                position: fixed; top: 50%; left: 50%;
                min-width: 100vw; min-height: 100vh;
                width: auto; height: auto;
                transform: translate(-50%, -50%);
                z-index: -9999; pointer-events: none;
                object-fit: cover;
                transition: filter 1s ease;
            }}
        `;
        parentDoc.head.appendChild(style);
    }}

    let videoElem = parentDoc.getElementById('local-video-main');
    if (!videoElem) {{
        videoElem = parentDoc.createElement('video');
        videoElem.id = 'local-video-main';
        videoElem.className = 'local-video-bg';
        videoElem.autoplay = true;
        videoElem.loop = true;
        videoElem.muted = true;
        videoElem.playsInline = true;
        parentDoc.body.appendChild(videoElem);
    }}
    
    // Only update src if it changed to prevent stuttering
    const newSrc = 'data:video/mp4;base64,{b64_video}';
    if (videoElem.src !== newSrc) {{
        videoElem.src = newSrc;
    }}
    
    // Apply the color filter based on the mode
    videoElem.style.filter = '{hue_filter} opacity({opacity})';
    </script>
    """
    components.html(js, height=0, width=0)
    
    # Theme configuration
    if theme == "Gold & White":
        text_color = "#1a1a1a"
        card_bg = "rgba(255, 255, 255, 0.85)"
        border_color = "rgba(218, 165, 32, 0.5)"
    elif theme == "Silver & White":
        text_color = "#1a1a1a"
        card_bg = "rgba(255, 255, 255, 0.85)"
        border_color = "rgba(169, 169, 169, 0.5)"
    elif theme == "Light":
        text_color = "#1a1a1a"
        card_bg = "rgba(255, 255, 255, 0.9)"
        border_color = "rgba(0, 0, 0, 0.1)"
    elif theme == "Cyber":
        text_color = "#00ffcc"
        card_bg = "rgba(20, 0, 30, 0.8)"
        border_color = "rgba(0, 255, 204, 0.5)"
    else: # Default Dark
        text_color = "#ffffff"
        card_bg = "rgba(20, 30, 50, 0.75)"
        border_color = "rgba(255, 255, 255, 0.15)"
        
    page_bg_css = f"""
    <style>
    /* Smooth slow Fly-through Transition */
    @keyframes flyThrough {{
        0% {{ opacity: 0; transform: scale(0.95) translateY(20px); }}
        100% {{ opacity: 1; transform: scale(1) translateY(0); }}
    }}
    .block-container {{
        animation: flyThrough 1.2s cubic-bezier(0.2, 0.8, 0.2, 1) forwards;
        background: transparent;
        padding: 40px !important;
        margin-top: 20px;
    }}
    @media (max-width: 768px) {{
        .block-container {{
            padding: 15px !important;
            margin-top: 5px !important;
        }}
    }}
    .stApp, .stApp p, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6, .stApp span, .stApp li {{
        color: {text_color} !important;
    }}
    /* Cards */
    .glass-card, .prediction-box-healthy, .prediction-box-pd {{
        background: {card_bg} !important;
        border: 1px solid {border_color} !important;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3) !important;
        backdrop-filter: blur(16px) !important;
        transition: all 0.3s ease;
        border-radius: 20px;
        padding: 25px;
    }}
    .glass-card:hover {{
        transform: translateY(-5px);
        box-shadow: 0 15px 50px rgba(0, 0, 0, 0.4) !important;
    }}
    /* Buttons */
    div[data-testid="stButton"] button, div[data-testid="stFormSubmitButton"] button {
        background: linear-gradient(45deg, #8b5cf6, #3b82f6, #d946ef) !important;
        background-size: 200% auto !important;
        color: white !important;
        font-weight: bold !important;
        border: none !important;
        border-radius: 12px !important;
        transition: 0.4s !important;
        text-transform: uppercase;
    }
    div[data-testid="stButton"] button *, div[data-testid="stFormSubmitButton"] button * {
        color: white !important;
    }
    div[data-testid="stButton"] button:hover, div[data-testid="stFormSubmitButton"] button:hover {
        background-position: right center !important;
        transform: scale(1.02) !important;
    }
    /* Sidebar */
    [data-testid="stSidebar"] {{
        background: {card_bg} !important;
        backdrop-filter: blur(20px) !important;
        border-right: 1px solid {border_color};
    }}
    </style>
    """
    st.markdown(page_bg_css, unsafe_allow_html=True)

# Lottie Animations
lottie_brain = load_lottieurl("https://lottie.host/8cd28148-f605-498c-a1ba-a0ba23c9ce05/2h9r6j1G1M.json")
lottie_data = load_lottieurl("https://lottie.host/17e2978a-c637-4d7c-bc70-431dd84bc315/sV2w3vW9yS.json")
lottie_wave = load_lottieurl("https://lottie.host/246bcf3f-e889-424a-aece-9eb7c569d675/LqG7yQ1yX2.json")

# --- Base CSS for Headers ---
st.markdown("""
<style>
    .main-header {
        font-size: 56px;
        font-weight: 900;
        background: -webkit-linear-gradient(45deg, #00C9FF, #92FE9D, #FF512F);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0px;
        padding-top: 10px;
    }
    .sub-header {
        font-size: 22px;
        text-align: center;
        margin-bottom: 40px;
        font-weight: 400;
    }
    @media (max-width: 1024px) {
        .main-header {
            font-size: 28px !important;
            padding-top: 5px !important;
            line-height: 1.2 !important;
            word-wrap: break-word !important;
        }
        .sub-header {
            font-size: 16px !important;
            margin-bottom: 20px !important;
        }
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_assets():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    with open('model_mfcc.pkl', 'rb') as f:
        model_mfcc = pickle.load(f)
    with open('scaler_mfcc.pkl', 'rb') as f:
        scaler_mfcc = pickle.load(f)
    with open('feature_names_mfcc.pkl', 'rb') as f:
        feature_names_mfcc = pickle.load(f)
    df = pd.read_csv("parkinson_disease_large.csv")
    if 'id' in df.columns:
        df = df.groupby('id').mean().reset_index()
    with open('background_data.pkl', 'rb') as f:
        background_data = pickle.load(f)
    with open('metrics.json', 'r') as f:
        metrics = json.load(f)
    return model, scaler, feature_names, model_mfcc, scaler_mfcc, feature_names_mfcc, df, metrics, background_data

try:
    model, scaler, feature_names, model_mfcc, scaler_mfcc, feature_names_mfcc, df, metrics, background_data = load_assets()
except Exception as e:
    st.error("Error loading models or dataset. Please make sure train_model.py has been run.")
    st.stop()

# Local video paths provided by user (Now relative for cloud deployment)
signal_video_path = "assets/signal_video.mp4"
neuron_video_path = "assets/neuron_video.mp4"

def login_page():
    set_local_video_background(video_path=neuron_video_path, color_mode="purple", theme="Dark")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<div class="main-header" style="font-size: 38px; background: -webkit-linear-gradient(45deg, #8b5cf6, #3b82f6, #d946ef); -webkit-background-clip: text;">Parkinson Disease Detection using AI & ML</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header" style="color: white;">Sign in to access your dashboard</div>', unsafe_allow_html=True)
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        
        with st.form("login_form"):
            email = st.text_input("Email Address", placeholder="doctor@clinic.com")
            password = st.text_input("Password", type="password", placeholder="••••••••")
            
            c_in, c_up = st.columns(2)
            with c_in:
                submit_in = st.form_submit_button("Sign In", use_container_width=True)
            with c_up:
                submit_up = st.form_submit_button("Create Account", use_container_width=True)
            
            if submit_in:
                if email and password and supabase:
                    try:
                        response = supabase.auth.sign_in_with_password({"email": email, "password": password})
                        st.session_state['logged_in'] = True
                        st.session_state['user_email'] = email
                        st.rerun()
                    except Exception as e:
                        st.error(f"Login failed: Invalid credentials or email not confirmed. Details: {e}")
                else:
                    st.error("Please enter email and password. (Or database is disconnected).")
                    
            if submit_up:
                if email and password and supabase:
                    try:
                        response = supabase.auth.sign_up({"email": email, "password": password})
                        st.success("Account created! If email confirmation is disabled in Supabase, you can Sign In now.")
                    except Exception as e:
                        st.error(f"Signup failed: {e}")
                else:
                    st.error("Please enter email and password.")
                    
        st.markdown('</div>', unsafe_allow_html=True)

def main_app():
    st.sidebar.markdown(f"### 👤 Welcome, Doctor")
    if st.sidebar.button("Log Out"):
        st.session_state['logged_in'] = False
        st.rerun()
    st.sidebar.markdown("---")
    
    st.sidebar.title("Visual Settings")
    theme = st.sidebar.selectbox("Select Theme", ["Dark", "Light", "Gold & White", "Silver & White", "Cyber"])
    st.sidebar.markdown("---")
    
    st.sidebar.title("Input Mode")
    mode = st.sidebar.radio("Select Input Mode:", [
        "Upload Audio File",
        "Demo: Healthy Patient (GFG Model)", 
        "Demo: Parkinson's Patient (GFG Model)", 
        "Dataset Analysis & Performance"
    ])

    # Clear prediction state when switching modes
    if 'current_mode' not in st.session_state or st.session_state['current_mode'] != mode:
        st.session_state['current_mode'] = mode
        if 'prediction' in st.session_state:
            del st.session_state['prediction']

    # Determine background video and color
    if mode == "Upload Audio File":
        bg_video = signal_video_path
        # Signal should go red if disease detected in uploaded file
        if st.session_state.get('prediction') == 1:
            bg_color = "red"
        else:
            bg_color = "blue"
            
    elif mode == "Demo: Healthy Patient (GFG Model)":
        bg_video = neuron_video_path
        bg_color = "blue"
        
    elif mode == "Demo: Parkinson's Patient (GFG Model)":
        bg_video = neuron_video_path
        bg_color = "red" # Neurons should be red in disease detected(demo) page
        
    else:
        bg_video = neuron_video_path
        bg_color = "blue"

    set_local_video_background(video_path=bg_video, color_mode=bg_color, theme=theme)

    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown('<div class="main-header" style="font-size: 38px;">Parkinson Disease Detection using AI & ML</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">Advanced Neuro-Acoustic Machine Learning System</div>', unsafe_allow_html=True)
    with col2:
        if lottie_brain:
            st_lottie(lottie_brain, height=120, key="brain_anim")

    input_data = None
    selected_model = None
    selected_scaler = None
    selected_features = None

    if mode in ["Demo: Healthy Patient (GFG Model)", "Demo: Parkinson's Patient (GFG Model)"]:
        st.sidebar.info(f"Using Logistic Regression and {len(feature_names)} clinical features.")
        selected_model = model
        selected_scaler = scaler
        selected_features = feature_names
        
        if mode == "Demo: Healthy Patient (GFG Model)":
            best_row = df[df['class'] == 0].iloc[0]
            best_prob = 1.0
            for i in range(len(df[df['class'] == 0])):
                row = df[df['class'] == 0].iloc[i]
                x = pd.DataFrame([row[feature_names].values], columns=feature_names)
                prob = model.predict_proba(scaler.transform(x))[0][1]
                if prob < best_prob:
                    best_prob = prob
                    best_row = row
            patient_row = best_row
            st.sidebar.success("Loaded data for a Healthy Patient.")
        else:
            best_row = df[df['class'] == 1].iloc[0]
            best_prob = 0.0
            for i in range(len(df[df['class'] == 1])):
                row = df[df['class'] == 1].iloc[i]
                x = pd.DataFrame([row[feature_names].values], columns=feature_names)
                prob = model.predict_proba(scaler.transform(x))[0][1]
                if prob > best_prob:
                    best_prob = prob
                    best_row = row
            patient_row = best_row
            st.sidebar.error("Loaded data for a Parkinson's Patient.")
            
        input_data = pd.DataFrame([patient_row[feature_names].values], columns=feature_names)
        st.session_state['ready_to_predict'] = True

    elif mode == "Upload Audio File":
        st.sidebar.info("Using the specialized MFCC XGBoost Model for audio files.")
        selected_model = model_mfcc
        selected_scaler = scaler_mfcc
        selected_features = feature_names_mfcc
        
        c1, c2 = st.columns([3, 1])
        with c1:
            st.markdown("### 📁 Acoustic Analysis Portal")
            st.markdown("<p>Upload patient voice recordings (.wav or .mp3) for instantaneous neuro-acoustic feature extraction.</p>", unsafe_allow_html=True)
        with c2:
            if lottie_wave:
                st_lottie(lottie_wave, height=80, key="wave_anim")
        
        audio_file = st.file_uploader("Upload Patient Audio", type=['wav', 'mp3', 'ogg'])
        
        if audio_file is not None:
            # Clear prediction state when a NEW file is uploaded
            if 'last_uploaded_file' not in st.session_state or st.session_state['last_uploaded_file'] != audio_file.name:
                st.session_state['last_uploaded_file'] = audio_file.name
                if 'prediction' in st.session_state:
                    del st.session_state['prediction']
                    st.rerun() # Immediately rerun to clear any red background and results

            st.audio(audio_file)
            y, sr = librosa.load(io.BytesIO(audio_file.read()), sr=None)
            duration = librosa.get_duration(y=y, sr=sr)
            
            if duration < 3.0:
                st.error(f"Audio is too short ({duration:.1f}s). Minimum 3 seconds required.")
                st.session_state['ready_to_predict'] = False
            else:
                is_clipping = np.max(np.abs(y)) >= 0.99
                mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                mfcc_means = np.mean(mfccs, axis=1)
                
                input_data = pd.DataFrame([mfcc_means[1:]], columns=feature_names_mfcc)
                st.success("✅ Successfully extracted acoustic features (Volume-Invariant).")
                
                with st.expander("View Mel Spectrogram Analysis", expanded=False):
                    fig_spec, ax_spec = plt.subplots(figsize=(8, 3))
                    
                    bg_col = 'none' # FIX MATPLOTLIB TRANSPARENCY BUG
                    text_col = 'black' if "White" in theme or theme == "Light" else 'white'
                    
                    fig_spec.patch.set_facecolor(bg_col)
                    ax_spec.set_facecolor(bg_col)
                    ax_spec.xaxis.label.set_color(text_col)
                    ax_spec.yaxis.label.set_color(text_col)
                    ax_spec.tick_params(colors=text_col)
                    
                    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
                    img = librosa.display.specshow(D, y_axis='linear', x_axis='time', sr=sr, ax=ax_spec)
                    fig_spec.colorbar(img, ax=ax_spec, format="%+2.f dB").ax.tick_params(colors=text_col)
                    st.pyplot(fig_spec)
                
                if is_clipping:
                    st.warning("⚠️ Clipping detected. Signal may be saturated.")
                    
                st.session_state['ready_to_predict'] = True
        else:
            st.session_state['ready_to_predict'] = False

    if mode == "Dataset Analysis & Performance":
        st.subheader("📊 Comprehensive Dataset & Theory Intelligence")
        
        t1, t2, t3, t4 = st.tabs(["Parkinson's Disease Theory", "Project Architecture", "Complete Results & Metrics", "Raw Data Explorer"])
        
        with t1:
            st.markdown("### Understanding Parkinson's Pathology")
            st.write("Parkinson's Disease (PD) is a progressive nervous system disorder that affects movement. It develops gradually, sometimes starting with a barely noticeable tremor in just one hand. But while tremor may be the most well-known sign of Parkinson's disease, the disorder also commonly causes stiffness or slowing of movement.")
            st.write("### Why use voice?")
            st.write("Voice impairment (dysarthria) is often one of the earliest indicators of Parkinson's Disease, frequently appearing before severe physical tremors. The acoustic features extracted from sustained vowel phonation (like saying 'Ahhh') provide a robust, non-invasive biomarker for detecting the disease with high accuracy.")
            st.write("Key indicators in vocal biomarkers include:")
            st.markdown("""
            * **Jitter**: Variations in fundamental frequency, leading to a breathy or hoarse voice.
            * **Shimmer**: Variations in amplitude, leading to a noisy, unstable voice.
            * **Harmonics-to-Noise Ratio (HNR)**: Indicates the amount of acoustic noise in the vocal signal.
            """)
        
        with t2:
            st.markdown("### 🏗️ Dual-Model Architecture")
            st.write("This application utilizes a dual-model approach to balance academic clinical datasets with real-world, noisy audio processing:")
            
            st.markdown("#### 1. GeeksForGeeks Clinical Pipeline (Logistic Regression)")
            st.write("Used for processing the rigorous Oxford PD dataset. It utilizes 30 highly correlated clinical features selected via the Chi-Square statistical test. Data balancing is handled via RandomOverSampler to prevent bias.")
            
            st.markdown("#### 2. Live Audio Acoustic Pipeline (XGBoost Classifier)")
            st.write("Used for live `.wav` and `.mp3` uploads. This model uses Mel-Frequency Cepstral Coefficients (MFCCs). Critically, it **drops the 0th MFCC coefficient** (which encodes volume) to prevent the model from misdiagnosing simply based on microphone loudness, drastically reducing false positives.")
            st.write("This model has been **recently fine-tuned with new raw patient telephone recordings** to increase generalization across real-world environments.")
            
        with t3:
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("#### Primary Model (Logistic Regression)")
                st.info(f"Test ROC AUC: {metrics['gfg_test_auc']:.4f}")
                cm = metrics['gfg_cm']
                st.write(f"- True Positives (Healthy): **{cm[0][0]}**")
                st.write(f"- False Positives: **{cm[0][1]}**")
                st.write(f"- False Negatives: **{cm[1][0]}**")
                st.write(f"- True Negatives (Parkinson's): **{cm[1][1]}**")
                
            with col_b:
                st.markdown("#### Acoustic Model (XGBoost)")
                if 'dataset_size' in metrics:
                    st.info(f"**Fine-tuned** on an augmented dataset of **{metrics['dataset_size']}** recordings (including **{metrics.get('new_data_added', 0)}** new patient telephone recordings).")
                st.success(f"Audio Classification Accuracy: {metrics['mfcc_accuracy']*100:.2f}%")
                if lottie_data:
                    st_lottie(lottie_data, height=200, key="data_anim")
                    
        with t4:
            st.markdown("### 🗃️ Raw Data Explorer")
            st.write(f"Displaying {len(df)} patient records with 30+ clinical acoustic features.")
            st.dataframe(df, use_container_width=True)

    else:
        col1, col2 = st.columns([1, 1.5])
        
        with col1:
            st.subheader("Diagnostic Engine")
            st.write("Initialize the neural network prediction.")
            
            if st.session_state.get('ready_to_predict', False):
                if st.button("Initialize Neural Analysis", use_container_width=True, type="primary"):
                    scaled_input = selected_scaler.transform(input_data)
                    prediction = selected_model.predict(scaled_input)[0]
                    probabilities = selected_model.predict_proba(scaled_input)[0]
                    
                    st.session_state['prediction'] = prediction
                    st.session_state['probabilities'] = probabilities
                    st.session_state['scaled_input'] = scaled_input
                    st.session_state['used_model'] = selected_model
                    st.session_state['used_features'] = selected_features
                    st.session_state['is_logistic'] = "GFG Model" in mode
                    st.rerun() # Force rerun to update the background color based on prediction

                # Render prediction results IF they exist
                if 'prediction' in st.session_state:
                    prediction = st.session_state['prediction']
                    probabilities = st.session_state['probabilities']
                    confidence = probabilities[int(prediction)] * 100
                    
                    if prediction == 1:
                        st.markdown(f'<div class="prediction-box-pd">⚠️ Parkinson\'s Disease Detected<br><span style="font-size:16px; color:#fca5a5;">Model Confidence: {confidence:.2f}%</span></div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="prediction-box-healthy">✅ Healthy (No PD Detected)<br><span style="font-size:16px; color:#6ee7b7;">Model Confidence: {confidence:.2f}%</span></div>', unsafe_allow_html=True)
                        
                    st.markdown("---")
                    st.write("**Save Record to Database**")
                    with st.form("save_db_form"):
                        patient_id = st.text_input("Patient ID (e.g., P-1029)", placeholder="P-XXXX")
                        save_btn = st.form_submit_button("Save Result to Cloud Database")
                        if save_btn:
                            if patient_id and supabase:
                                try:
                                    data = {
                                        "doctor_email": st.session_state.get('user_email', 'unknown'),
                                        "patient_id": patient_id,
                                        "prediction_result": "Parkinson's Disease" if prediction == 1 else "Healthy",
                                        "confidence_score": float(confidence)
                                    }
                                    supabase.table("diagnostic_records").insert(data).execute()
                                    st.success("✅ Record securely saved to cloud database!")
                                except Exception as e:
                                    st.error(f"Error saving to database. Ensure you ran the SQL command to create the table. Details: {e}")
                            else:
                                st.error("Please enter a Patient ID (and ensure database is connected).")
            else:
                st.warning("Awaiting input data...")

        with col2:
            st.subheader("Explainable AI (SHAP)")
            if 'scaled_input' in st.session_state and 'prediction' in st.session_state:
                st.write("Feature contribution breakdown determining the model's output.")
                
                used_model = st.session_state['used_model']
                used_features = st.session_state['used_features']
                scaled_in = st.session_state['scaled_input']
                is_logistic = st.session_state.get('is_logistic', False)
                
                fig, ax = plt.subplots(figsize=(8, 5))
                bg_col = 'none' # FIX MATPLOTLIB TRANSPARENCY BUG
                text_col = 'black' if "White" in theme or theme == "Light" else 'white'
                
                fig.patch.set_facecolor(bg_col)
                ax.set_facecolor(bg_col)
                ax.xaxis.label.set_color(text_col)
                ax.yaxis.label.set_color(text_col)
                ax.tick_params(colors=text_col)
                for spine in ax.spines.values():
                    spine.set_edgecolor(text_col)
                
                if is_logistic:
                    masker = shap.maskers.Independent(data=background_data)
                    explainer = shap.LinearExplainer(used_model, masker)
                    shap_values = explainer(scaled_in)
                    shap_values.feature_names = used_features
                    shap.plots.bar(shap_values[0], show=False, max_display=12)
                else:
                    explainer = shap.TreeExplainer(used_model)
                    shap_values = explainer(scaled_in)
                    shap_values.feature_names = used_features
                    shap.plots.bar(shap_values[0], show=False, max_display=12)
                
                plt.xticks(color=text_col)
                plt.yticks(color=text_col)
                plt.xlabel("SHAP value (impact on model output)", color=text_col)
                st.pyplot(fig)
            else:
                st.info("Results pending analysis.")

if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

if not st.session_state['logged_in']:
    login_page()
else:
    main_app()
