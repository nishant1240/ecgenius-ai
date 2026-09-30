# ECGENIUS AI — Local Prototype

AI-assisted ECG screening and health-assistant prototype. **Educational/research
prototype only — not a medical diagnostic device.**

## 1. Setup

```bash
# (recommended) create a virtual environment first
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

## 2. Run

```bash
streamlit run app.py
```

Streamlit will open the app in your browser at `http://localhost:8501`.
The first run trains the ECG classifier on a synthetic dataset (a few
seconds) and caches it, so later runs start instantly.

## 3. What works out of the box (no extra installs)

- ECG demo library (5 built-in cases) + CSV/TXT upload
- Signal filtering + R-peak detection (SciPy)
- ML classification (scikit-learn RandomForest) with confidence + risk level
- Rule-based Explainable AI
- Weekly progress reports (SQLite + Markdown export)
- Full English/Hindi UI
- Live simulation mode

## 4. Optional upgrades (each is independent — install only what you want)

| Feature | Install | Notes |
|---|---|---|
| Stronger classifier | `pip install xgboost` | Auto-detected and used in place of RandomForest |
| Quantitative XAI | `pip install shap` | Adds SHAP feature-attribution chart |
| Speech-to-text | `pip install faster-whisper` | Powers the voice-question upload in Voice Assistant |
| Offline text-to-speech | `pip install pyttsx3` | "Speak result" button |
| Online text-to-speech | `pip install edge-tts` | Used if `pyttsx3` isn't installed |
| Real PhysioNet data | `pip install wfdb` | Load real MIT-BIH/PTB records via `modules/signal_utils.load_wfdb_record` |
| PDF reports | `pip install reportlab` | Weekly report exports as a real PDF instead of Markdown |
| Local LLM chatbot | Install [Ollama](https://ollama.com), then `ollama pull deepseek-r1:8b` and `pip install requests` | Chatbot uses it automatically when the Ollama server is running |

## 5. Project structure

```
ecgenius/
├── app.py                  # Streamlit app — one section per spec feature
├── requirements.txt
├── data/                   # created at runtime: ecgenius.db (SQLite history)
└── modules/
    ├── signal_utils.py     # synthetic ECG generation, filtering, peak detection, file loading
    ├── classifier.py       # feature extraction + model training/inference
    ├── explainability.py   # rule-based + optional SHAP explanations
    ├── voice_assistant.py  # speech-to-text / text-to-speech
    ├── chatbot.py          # Ollama LLM + rule-based fallback
    ├── reports.py           # SQLite logging, weekly summary, PDF/Markdown export
    ├── demo_library.py     # 5 built-in demo cases
    └── i18n.py             # English/Hindi UI strings
```

## 6. Presenting at an ideathon

Suggested demo flow: **Overview → Demo Library (load "Atrial Fibrillation") →
Visualization → Classification & Alerts → Explainable AI → Voice Assistant
("Speak result") → Chatbot → Weekly Reports → Future Expansion.**

Be upfront that the ML model is trained on synthetic signals for this
prototype (real MIT-BIH training is one `pip install wfdb` + a data loader
away — see `signal_utils.load_wfdb_record`), and that the chatbot/voice
sections light up further with the optional installs above.
