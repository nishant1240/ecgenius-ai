"""
AI Health Assistant chatbot.

Tries a locally-running Ollama server (spec: DeepSeek 8B) first. If Ollama
isn't running/installed, falls back to a small rule-based educational Q&A so
the section still works out of the box.

To enable the full local LLM:
  1. Install Ollama: https://ollama.com
  2. Run:  ollama pull deepseek-r1:8b   (or your preferred 8B model)
  3. Make sure `ollama serve` is running (default http://localhost:11434)
"""

SYSTEM_PROMPT = (
    "You are ECGENIUS AI, an educational cardiac-health assistant. You explain "
    "ECG concepts, heart-rhythm terminology, and general symptoms in plain "
    "language. You must NEVER diagnose a specific person, prescribe medication, "
    "or replace professional medical advice. Always include a brief reminder to "
    "consult a qualified healthcare professional for any personal concern."
)

RULE_BASED = {
    "en": {
        "arrhythmia": "Arrhythmia means an irregular heartbeat — it can beat too fast, too slow, or erratically. Many types are harmless, but some need medical attention.",
        "heart rate": "Heart rate is the number of times your heart beats per minute (bpm). A typical resting adult range is 60–100 bpm.",
        "atrial fibrillation": "Atrial fibrillation (AFib) is a common irregular heart rhythm originating in the atria, causing an erratic pulse.",
        "afib": "Atrial fibrillation (AFib) is a common irregular heart rhythm originating in the atria, causing an erratic pulse.",
        "bradycardia": "Bradycardia means a resting heart rate slower than 60 bpm.",
        "tachycardia": "Tachycardia means a resting heart rate faster than 100 bpm.",
        "explain": "I analyze R-R intervals and beat morphology from the ECG to estimate heart rate and rhythm regularity, then a trained model compares those features against learned patterns to suggest a category.",
        "default": "I can explain ECG terms like arrhythmia, heart rate, AFib, bradycardia or tachycardia — ask away! (Educational information only, not a diagnosis.)",
    },
    "hi": {
        "arrhythmia": "अतालता का अर्थ है अनियमित दिल की धड़कन — यह बहुत तेज़, धीमी या अनियमित हो सकती है।",
        "अतालता": "अतालता का अर्थ है अनियमित दिल की धड़कन — यह बहुत तेज़, धीमी या अनियमित हो सकती है।",
        "heart rate": "हृदय गति प्रति मिनट धड़कनों की संख्या है। सामान्य विश्राम सीमा 60–100 bpm है।",
        "हृदय गति": "हृदय गति प्रति मिनट धड़कनों की संख्या है। सामान्य विश्राम सीमा 60–100 bpm है।",
        "atrial fibrillation": "आलिंद फिब्रिलेशन (AFib) एक सामान्य अनियमित हृदय लय है।",
        "afib": "आलिंद फिब्रिलेशन (AFib) एक सामान्य अनियमित हृदय लय है।",
        "bradycardia": "ब्रैडीकार्डिया का अर्थ है 60 bpm से धीमी हृदय गति।",
        "tachycardia": "टैचीकार्डिया का अर्थ है 100 bpm से तेज़ हृदय गति।",
        "explain": "मैं R-R अंतराल और धड़कन आकृति का विश्लेषण करके हृदय गति और लय नियमितता का अनुमान लगाता हूँ।",
        "default": "आप मुझसे अतालता, हृदय गति, AFib, ब्रैडीकार्डिया या टैचीकार्डिया के बारे में पूछ सकते हैं! (केवल शैक्षिक जानकारी।)",
    },
}


def rule_based_answer(question: str, lang: str = "en") -> str:
    q = question.lower()
    table = RULE_BASED.get(lang, RULE_BASED["en"])
    for key, ans in table.items():
        if key != "default" and key in q:
            return ans
    return table["default"]


def ollama_chat(question: str, lang: str = "en", model: str = "deepseek-r1:8b", host: str = "http://localhost:11434"):
    """Returns (answer, used_llm: bool). Falls back to rule-based on any failure."""
    try:
        import requests
        resp = requests.post(
            f"{host}/api/chat",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT + (
                        " Respond in Hindi." if lang == "hi" else " Respond in English.")},
                    {"role": "user", "content": question},
                ],
                "stream": False,
            },
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        answer = data.get("message", {}).get("content", "").strip()
        if answer:
            return answer, True
    except Exception:
        pass
    return rule_based_answer(question, lang), False
