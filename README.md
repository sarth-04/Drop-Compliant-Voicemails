
# 📞 Intelligent Voicemail Drop Detection System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Deepgram](https://img.shields.io/badge/Deepgram-Transcription-green)](https://deepgram.com/)
[![Gemini](https://img.shields.io/badge/AI-Google%20Gemini-blueviolet)](https://deepmind.google/technologies/gemini/)


An AI-powered system designed to accurately detect the precise moment a voicemail greeting ends. By combining energy-based silence detection with Large Language Model (LLM) intent analysis, this system determines the optimal timestamp to "drop" a pre-recorded message, avoiding awkward pauses or interruptions.



## 🚀 Features

* **Hybrid Detection Logic:** Uses a combination of RMS energy levels (silence detection) and Semantic Analysis (LLM) for high accuracy.
* Real-Time Simulation: Simulates live audio streaming by processing `.wav` files in 50ms chunks.
* **Low Latency Transcription:** Integrates **Deepgram API** for fast, streaming speech-to-text.
* **Multi-LLM Support:** Compatible with both **Google Gemini** and **OpenAI GPT-4o** for intent recognition.
* **Adaptive Wait Times:**
    * *Standard Silence:* 2.2 seconds (default).
    * *AI-Confirmed Silence:* 0.6 seconds (fast drop upon intent confirmation).

## 📂 Project Structure

```text
├── dataset/                # Folder containing audio files (.wav)
├── ground_truth.csv        # Validation data with correct timestamps & accuracy scores
├── llm_new.py              # Main processing engine & simulation script
├── .env                    # API keys configuration (not committed to repo)
└── README.md               # Project documentation

```

## 🛠️ Prerequisites

Before running the code, ensure you have the following installed:

* **Python 3.8+**
* **API Keys:**
* [Deepgram API Key](https://console.deepgram.com/) (Required for transcription)
* [Google Gemini API Key](https://aistudio.google.com/) **OR** [OpenAI API Key](https://platform.openai.com/) (Required for intent detection)



## 📦 Installation

1. **Clone the repository:**
```bash
git clone [https://github.com/yourusername/voicemail-drop-detection.git](https://github.com/yourusername/voicemail-drop-detection.git)
cd voicemail-drop-detection

```


2. **Install dependencies:**
```bash
pip install deepgram-sdk==3.* google-generativeai openai python-dotenv numpy scipy pandas

```


3. **Environment Setup:**
Create a file named `.env` in the root directory of the project. Add your API keys as shown below. You only need one of the LLM keys (Gemini or OpenAI), but Deepgram is mandatory.
```ini
# .env file content

# Required for Speech-to-Text
DEEPGRAM_API_KEY=your_deepgram_api_key_here

# Option 1: Use Google Gemini (Recommended)
GEMINI_API_KEY=your_gemini_api_key_here

# Option 2: Use OpenAI
OPENAI_API_KEY=your_openai_api_key_here

```



## ⚙️ Configuration

The system sensitivity in the `CONFIG` dictionary located at the top of `llm_new.py`:

```python
CONFIG = {
    "CHUNK_MS": 50,                  # Audio chunk size (simulates streaming)
    "SILENCE_RMS_THRESH": 100,       # Energy threshold for silence
    "DEFAULT_WAIT_TIME": 2.2,        # Wait time without AI confirmation
    "LLM_CONFIRMED_WAIT_TIME": 0.6,  # Wait time WITH AI confirmation (Fast Drop)
    "GEMINI_MODEL": "gemini-2.5-flash",
    "OPENAI_MODEL": "gpt-4o-mini",
}

```

## 🚀 Usage

### 1. Prepare Data

Ensure your audio files are placed in the `dataset/` folder. The system expects `.wav` files.

### 2. Run the Simulation

To test the detection on a specific file (e.g., `test1.wav`), open `llm_new.py` and modify the `__main__` block at the bottom:

```python
if __name__ == "__main__":
    run_simulation("dataset/test1.wav")

```

Then run the script in your terminal:

```bash
python llm_new.py

```

### 3. Output

The console will show the real-time transcription and the final drop decision:

```text
📞 Dialing dataset/test1.wav...
   Audio Format: 8000Hz | 1 Channel(s)
   🗣️  Transcript: Hi you have reached the voicemail of...
   🧠 [AI] Intent Detected: 'Greeting Finished'.
   ✅ DROP TRIGGER: AI + Short Silence (0.6s)
🚀 ACTION: DROP MESSAGE at 14.40s

```

## 📊 Logic & Workflow

1. **Audio Streaming:** The system reads the `.wav` file in 50ms chunks to mimic a phone network.
2. **Transcription:** Audio is sent to Deepgram; text is returned via WebSocket.
3. **Intent Analysis:** The text is accumulated and sent to the LLM (Gemini/GPT). The LLM checks for phrases indicating the end of a greeting (e.g., "Leave a message").
4. **Decision Engine:**
* If **Audio Energy > Threshold**: The silence timer resets.
* If **Silence Detected**:
* The system checks if the silence duration exceeds the `Limit`.
* The `Limit` dynamically switches from **2.2s** to **0.6s** if the LLM detects the greeting is finished.





## 📈 Results

The system has been evaluated against a dataset of 17 distinct voicemail scenarios, achieving an accuracy of **88.23%**.
| audio name | starting timestamp | ending timestamp | pred timestamp | score           |
| :---       | :---               | :---             | :---           | :---            |
| test1      | 13                 | 15               | 14.4           | 1               |
| test2      | 13                 | 15               | 14.4           | 1               |
| test3      | 5                  | 7                | 6.95           | 1               |
| test4      | 1                  | 3                | 2.9            | 1               |
| test5      | 3                  | 5                | 4.05           | 1               |
| test6      | 3                  | 5                | 5.65           | 0               |
| test7      | 23                 | 25               | 24.05          | 1               |
| test8      | 6                  | 8                | 7.55           | 1               |
| test9      | 6                  | 8                | 7.25           | 1               |
| test10     | 6                  | 8                | 7.05           | 1               |
| vm1_output | 10                 | 12               | 11.35          | 1               |
| vm2_output | 9                  | 11               | 9.75           | 1               |
| vm3_output | 10                 | 12               | 10.45          | 1               |
| vm4_output | 5                  | 7                | 7.25           | 0               |
| vm5_output | 15                 | 17               | 15.15          | 1               |
| vm6_output | 6                  | 8                | 6.25           | 1               |
| vm7_output | 12                 | 14               | 13.15          | 1               |
| **Total** |                    |                  |                | **Accuracy=88.23%** |

*(See `ground_truth.csv` for full validation data)*



