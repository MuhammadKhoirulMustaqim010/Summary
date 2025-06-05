import streamlit as st
from transformers import pipeline
import time
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="AI Ringkasan Teks", 
    page_icon="📝", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- CUSTOM CSS ----------
st.markdown("""
    <style>
        /* Main styling */
        .main-header {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 15px;
            margin-bottom: 2rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        .main-title {
            font-size: 3.5em;
            font-weight: bold;
            text-align: center;
            color: white;
            margin: 0;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .main-subtitle {
            font-size: 1.3em;
            text-align: center;
            color: rgba(255,255,255,0.9);
            margin-top: 10px;
        }
        
        /* Cards */
        .metric-card {
            background: white;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            border-left: 4px solid #667eea;
            margin: 1rem 0;
        }
        
        .summary-card {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 2rem;
            border-radius: 15px;
            border: 1px solid #e1e8ed;
            margin: 1rem 0;
        }
        
        /* Buttons */
        .stButton > button {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 25px;
            padding: 0.75rem 2rem;
            font-weight: bold;
            font-size: 1.1em;
            transition: all 0.3s ease;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15);
        }
        
        /* Sidebar */
        .sidebar-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 1rem;
            text-align: center;
            font-weight: bold;
        }
        
        /* Text areas */
        .stTextArea textarea {
            border-radius: 10px;
            border: 2px solid #e1e8ed;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        
        .stTextArea textarea:focus {
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }
        
        /* Progress bar */
        .stProgress > div > div > div > div {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        }
        
        /* Footer */
        .footer {
            text-align: center;
            padding: 2rem;
            color: #666;
            border-top: 1px solid #e1e8ed;
            margin-top: 3rem;
        }
        
        /* Animation */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .fade-in {
            animation: fadeIn 0.6s ease-out;
        }
        
        /* Info boxes */
        .info-box {
            background: #f8f9fa;
            border-left: 4px solid #17a2b8;
            padding: 1rem;
            border-radius: 5px;
            margin: 1rem 0;
        }
        
        .warning-box {
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 1rem;
            border-radius: 5px;
            margin: 1rem 0;
        }
        
        .success-box {
            background: #d4edda;
            border-left: 4px solid #28a745;
            padding: 1rem;
            border-radius: 5px;
            margin: 1rem 0;
        }
    </style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
st.markdown("""
    <div class="main-header fade-in">
        <div class="main-title">🤖 AI Text Summarizer</div>
        <div class="main-subtitle">Ringkas teks panjang menjadi inti sari dengan teknologi AI terdepan</div>
    </div>
""", unsafe_allow_html=True)

# ---------- SIDEBAR ----------
with st.sidebar:
    st.markdown('<div class="sidebar-header">⚙️ Pengaturan Model</div>', unsafe_allow_html=True)
    
    # Model selection
    model_options = {
        "BART (Facebook)": "facebook/bart-large-cnn",
        "T5 Base": "t5-base",
        "DistilBART": "sshleifer/distilbart-cnn-12-6",
        "Pegasus": "google/pegasus-xsum"
    }
    
    selected_model = st.selectbox(
        "🎯 Pilih Model AI",
        list(model_options.keys()),
        index=0,
        help="Pilih model AI untuk meringkas teks"
    )
    
    # Parameters
    st.markdown("### 📊 Parameter Ringkasan")
    max_len = st.slider("📏 Panjang Maksimal", 50, 500, 150, 10)
    min_len = st.slider("📐 Panjang Minimal", 10, 100, 30, 5)
    
    # Advanced settings
    with st.expander("🔧 Pengaturan Lanjutan"):
        do_sample = st.checkbox("Sampling", value=False, help="Gunakan sampling untuk variasi hasil")
        temperature = st.slider("🌡️ Temperature", 0.1, 2.0, 1.0, 0.1, help="Kreativitas model")
        top_p = st.slider("🎯 Top-p", 0.1, 1.0, 0.9, 0.1, help="Nucleus sampling")
    
    # Statistics
    st.markdown("---")
    st.markdown("### 📈 Statistik Sesi")
    if 'summary_count' not in st.session_state:
        st.session_state.summary_count = 0
    if 'total_chars_processed' not in st.session_state:
        st.session_state.total_chars_processed = 0
    
    st.metric("Ringkasan Dibuat", st.session_state.summary_count)
    st.metric("Total Karakter", f"{st.session_state.total_chars_processed:,}")

# ---------- LOAD MODEL ----------
@st.cache_resource(show_spinner=False)
def load_model(model_name):
    try:
        return pipeline("summarization", model=model_name)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# ---------- HELPER FUNCTIONS ----------
def analyze_text(text):
    """Analyze text statistics"""
    words = len(text.split())
    chars = len(text)
    sentences = text.count('.') + text.count('!') + text.count('?')
    paragraphs = len([p for p in text.split('\n') if p.strip()])
    
    return {
        'words': words,
        'characters': chars,
        'sentences': sentences,
        'paragraphs': paragraphs
    }

def estimate_reading_time(text):
    """Estimate reading time (average 200 words per minute)"""
    words = len(text.split())
    return max(1, round(words / 200))

# ---------- MAIN CONTENT ----------
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("## 📝 Input Teks")
    
    # Input options
    input_method = st.radio(
        "Pilih metode input:",
        ["✍️ Ketik Manual", "📄 Upload File", "🌐 Contoh Teks"],
        horizontal=True
    )
    
    text_input = ""
    
    if input_method == "✍️ Ketik Manual":
        text_input = st.text_area(
            "Masukkan teks yang ingin diringkas:",
            height=200,
            placeholder="Tempel atau ketik teks panjang di sini...",
            help="Masukkan teks minimal 50 kata untuk hasil terbaik"
        )
    
    elif input_method == "📄 Upload File":
        uploaded_file = st.file_uploader(
            "Upload file teks (.txt)",
            type=['txt'],
            help="Upload file teks untuk diringkas"
        )
        if uploaded_file:
            text_input = str(uploaded_file.read(), "utf-8")
            st.success(f"File berhasil diupload: {uploaded_file.name}")
    
    elif input_method == "🌐 Contoh Teks":
        sample_texts = {
            "Artikel Teknologi": """
            Artificial Intelligence (AI) telah mengalami perkembangan yang sangat pesat dalam beberapa dekade terakhir. 
            Teknologi ini telah merevolusi berbagai aspek kehidupan manusia, mulai dari cara kita berkomunikasi, bekerja, 
            hingga cara kita memecahkan masalah kompleks. Machine learning, sebagai subset dari AI, telah memungkinkan 
            komputer untuk belajar dari data tanpa perlu diprogram secara eksplisit untuk setiap tugas spesifik.
            
            Dalam bidang kesehatan, AI telah membantu dokter dalam mendiagnosis penyakit dengan akurasi yang tinggi. 
            Sistem AI dapat menganalisis gambar medis seperti CT scan dan MRI dengan kecepatan dan presisi yang 
            mengagumkan. Di bidang transportasi, mobil otonom yang didukung AI mulai diuji coba di berbagai negara, 
            menjanjikan masa depan transportasi yang lebih aman dan efisien.
            
            Namun, perkembangan AI juga menimbulkan berbagai tantangan etis dan sosial. Kekhawatiran tentang 
            penggantian pekerjaan manusia oleh robot dan AI, privasi data, serta bias dalam algoritma menjadi 
            isu-isu penting yang perlu diselesaikan. Oleh karena itu, pengembangan AI harus dilakukan dengan 
            mempertimbangkan aspek-aspek etis dan dampaknya terhadap masyarakat.
            """,
            "Artikel Pendidikan": """
            Sistem pendidikan di era digital mengalami transformasi yang signifikan. Pandemi COVID-19 telah 
            mempercepat adopsi teknologi dalam pendidikan, memaksa institusi pendidikan untuk beradaptasi dengan 
            pembelajaran online. E-learning platform dan virtual classroom menjadi solusi utama untuk melanjutkan 
            proses belajar mengajar di tengah pembatasan fisik.
            
            Teknologi seperti Learning Management System (LMS), video conferencing, dan aplikasi pembelajaran 
            interaktif telah menjadi bagian integral dari proses pendidikan modern. Guru dan siswa harus 
            mempelajari cara menggunakan berbagai tools digital untuk memastikan efektivitas pembelajaran.
            
            Meskipun memberikan fleksibilitas dan aksesibilitas yang lebih besar, pembelajaran online juga 
            menimbulkan tantangan baru. Kesenjangan digital antara siswa yang memiliki akses teknologi dengan 
            yang tidak, serta kurangnya interaksi tatap muka yang penting untuk perkembangan sosial siswa, 
            menjadi isu yang perlu diperhatikan dalam implementasi pendidikan digital.
            """
        }
        
        selected_sample = st.selectbox("Pilih contoh teks:", list(sample_texts.keys()))
        text_input = sample_texts[selected_sample]
        st.text_area("Preview teks:", value=text_input, height=150, disabled=True)

with col2:
    if text_input:
        st.markdown("## 📊 Analisis Teks")
        
        stats = analyze_text(text_input)
        reading_time = estimate_reading_time(text_input)
        
        # Display metrics in cards
        st.markdown(f"""
        <div class="metric-card">
            <h4>📖 Statistik Teks</h4>
            <p><strong>Kata:</strong> {stats['words']:,}</p>
            <p><strong>Karakter:</strong> {stats['characters']:,}</p>
            <p><strong>Kalimat:</strong> {stats['sentences']}</p>
            <p><strong>Paragraf:</strong> {stats['paragraphs']}</p>
            <p><strong>Waktu Baca:</strong> ~{reading_time} menit</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Text length visualization
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = stats['words'],
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Jumlah Kata"},
            gauge = {
                'axis': {'range': [None, 1000]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 250], 'color': "lightgray"},
                    {'range': [250, 500], 'color': "gray"},
                    {'range': [500, 1000], 'color': "lightgreen"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 800
                }
            }
        ))
        fig.update_layout(height=200, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

# ---------- SUMMARIZATION ----------
st.markdown("---")

if st.button("✨ Ringkas Teks Sekarang", use_container_width=True):
    if not text_input.strip():
        st.markdown("""
        <div class="warning-box">
            ⚠️ <strong>Peringatan:</strong> Silakan masukkan teks terlebih dahulu!
        </div>
        """, unsafe_allow_html=True)
    
    elif len(text_input.split()) < 20:
        st.markdown("""
        <div class="warning-box">
            ⚠️ <strong>Peringatan:</strong> Teks terlalu pendek. Masukkan minimal 20 kata untuk hasil yang optimal.
        </div>
        """, unsafe_allow_html=True)
    
    else:
        # Load model
        with st.spinner("🔄 Memuat model AI..."):
            model_name = model_options[selected_model]
            summarizer = load_model(model_name)
        
        if summarizer:
            # Create progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Update progress
                for i in range(100):
                    progress_bar.progress(i + 1)
                    if i < 30:
                        status_text.text("🔍 Menganalisis teks...")
                    elif i < 70:
                        status_text.text("🤖 Memproses dengan AI...")
                    else:
                        status_text.text("📝 Menyiapkan ringkasan...")
                    time.sleep(0.01)
                
                # Generate summary
                summary_params = {
                    "max_length": max_len,
                    "min_length": min_len,
                    "do_sample": do_sample
                }
                
                if do_sample:
                    summary_params.update({
                        "temperature": temperature,
                        "top_p": top_p
                    })
                
                summary = summarizer(text_input, **summary_params)
                summary_text = summary[0]['summary_text']
                
                # Clear progress
                progress_bar.empty()
                status_text.empty()
                
                # Update session statistics
                st.session_state.summary_count += 1
                st.session_state.total_chars_processed += len(text_input)
                
                # Display results
                st.markdown("""
                <div class="success-box">
                    ✅ <strong>Berhasil!</strong> Ringkasan telah dibuat dengan model AI.
                </div>
                """, unsafe_allow_html=True)
                
                # Summary comparison
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 📄 Teks Asli")
                    st.text_area("", value=text_input, height=200, disabled=True)
                    
                    original_stats = analyze_text(text_input)
                    st.metric("Kata", original_stats['words'])
                
                with col2:
                    st.markdown("### 📋 Ringkasan")
                    st.markdown(f"""
                    <div class="summary-card">
                        <p style="font-size: 1.1em; line-height: 1.6; margin: 0;">
                            {summary_text}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    summary_stats = analyze_text(summary_text)
                    st.metric("Kata", summary_stats['words'])
                    
                    # Compression ratio
                    compression_ratio = (1 - summary_stats['words'] / original_stats['words']) * 100
                    st.metric("Kompresi", f"{compression_ratio:.1f}%")
                
                # Download button
                st.download_button(
                    label="💾 Download Ringkasan",
                    data=f"RINGKASAN TEKS\n{'='*50}\n\nTeks Asli ({original_stats['words']} kata):\n{text_input}\n\n{'='*50}\n\nRingkasan ({summary_stats['words']} kata):\n{summary_text}\n\n{'='*50}\nDibuat dengan AI Text Summarizer\nTanggal: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    file_name=f"ringkasan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
                
                # Comparison chart
                st.markdown("### 📊 Perbandingan Statistik")
                
                comparison_data = pd.DataFrame({
                    'Metrik': ['Kata', 'Karakter', 'Kalimat'],
                    'Teks Asli': [original_stats['words'], original_stats['characters'], original_stats['sentences']],
                    'Ringkasan': [summary_stats['words'], summary_stats['characters'], summary_stats['sentences']]
                })
                
                fig = px.bar(comparison_data, x='Metrik', y=['Teks Asli', 'Ringkasan'], 
                           barmode='group', title="Perbandingan Teks Asli vs Ringkasan")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Terjadi kesalahan: {str(e)}")
                st.info("💡 Coba dengan teks yang lebih pendek atau pilih model yang berbeda.")

# ---------- FOOTER ----------
st.markdown("---")
st.markdown("""
<div class="footer">
    <div style="display: flex; justify-content: center; align-items: center; gap: 20px; flex-wrap: wrap;">
        <div>🤖 Powered by Hugging Face Transformers</div>
        <div>•</div>
        <div>⚡ Built with Streamlit</div>
        <div>•</div>
        <div> Made by Khoirul Mustaqim</div>
    </div>
    <div style="margin-top: 10px; font-size: 0.9em;">
        v2.0 - Enhanced UI/UX Edition
    </div>
</div>
""", unsafe_allow_html=True)

# ---------- ADDITIONAL FEATURES ----------
with st.sidebar:
    st.markdown("---")
    st.markdown("### 🆘 Bantuan")
    
    with st.expander("📖 Cara Penggunaan"):
        st.markdown("""
        1. **Pilih Model**: Pilih model AI yang sesuai
        2. **Atur Parameter**: Sesuaikan panjang ringkasan
        3. **Input Teks**: Masukkan teks yang ingin diringkas
        4. **Klik Ringkas**: Tunggu proses selesai
        5. **Download**: Simpan hasil ringkasan
        """)
    
    with st.expander("🔍 Tips & Trik"):
        st.markdown("""
        - **Teks Optimal**: 100-1000 kata untuk hasil terbaik
        - **Model BART**: Bagus untuk artikel berita
        - **Model T5**: Fleksibel untuk berbagai jenis teks
        - **Parameter**: Sesuaikan sesuai kebutuhan
        """)
    
    if st.button("🔄 Reset Statistik"):
        st.session_state.summary_count = 0
        st.session_state.total_chars_processed = 0
        st.success("Statistik telah direset!")
        st.experimental_rerun()