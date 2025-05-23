# 🎯 SarcastiSense - Real-Time Sarcasm Detector

**Detect sarcasm, passive-aggression, and mock politeness in real-time during voice conversations!**

SarcastiSense combines speech-to-text transcription, emotion detection, and AI-powered sarcasm analysis to identify when someone might be saying one thing but meaning another.

## 🌟 Features

- **Real-time Speech Processing**: Continuous audio capture and transcription using OpenAI Whisper
- **AI-Powered Sarcasm Detection**: Uses Google Gemini AI to analyze text for sarcasm patterns
- **Emotion Analysis**: Detects emotional tone from voice characteristics
- **Tone-Text Mismatch Detection**: Identifies when emotional tone contradicts spoken words
- **Live Scoring**: Real-time sarcasm percentage with confidence levels
- **Color-Coded Output**: Visual feedback with different colors for sarcasm levels
- **Detection History**: Tracks and analyzes patterns over time

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Microphone    │───▶│  Speech-to-Text │───▶│  Text Analysis  │
│     Input       │    │   (Whisper)     │    │   (Gemini AI)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       │                       ▼
┌─────────────────┐              │              ┌─────────────────┐
│ Voice Emotion   │              │              │ Sarcasm Score   │
│   Detection     │              │              │   & Analysis    │
└─────────────────┘              │              └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 ▼
                    ┌─────────────────────────┐
                    │  Tone-Text Mismatch     │
                    │     Detection           │
                    └─────────────────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │    Live Display &       │
                    │   Results Output        │
                    └─────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Microphone access
- Google Gemini API key
- Internet connection for AI analysis

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/real-time-sarcasm-detector.git
   cd real-time-sarcasm-detector
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up API key**
   - Get a Google Gemini API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Replace `YOUR_API_KEY_HERE` in the code with your actual API key

4. **Run SarcastiSense**
   ```bash
   python Sarcasm_Live.py
   ```

### Usage

1. **Start the application**
   ```bash
   python sarcastisense.py
   ```

2. **Begin speaking** into your microphone

3. **View real-time results** with color-coded sarcasm levels:
   - 🟢 **Green (0-39%)**: Genuine/Sincere
   - 🟡 **Yellow (40-69%)**: Potentially sarcastic
   - 🔴 **Red (70-100%)**: Highly sarcastic

4. **Press 'q'** to quit and see statistics

## 📊 Example Output

```
============================================================
[14:32:15] 📝 "Oh great, another meeting. Just what I needed today."
😊 Emotion: angry (0.8)
🔥 Sarcasm Level: 87% - sarcastic
🎭 TONE MISMATCH DETECTED!
💭 Detected sarcastic language with contradictory positive words and negative tone [TONE MISMATCH DETECTED]
============================================================
```

## 🎛️ Configuration Options

### Audio Settings
```python
CHUNK = 1024          # Audio buffer size
RATE = 16000          # Sample rate (Hz)
RECORD_SECONDS = 3    # Processing chunk duration
```

### Detection Sensitivity
- Adjust sarcasm thresholds in `display_results()` method
- Modify tone-mismatch boost in `apply_tone_mismatch_boost()`

## 🧪 Testing Examples

Try these phrases to test SarcastiSense:

**Highly Sarcastic (Expected 70%+)**
- "Oh great, another meeting. Just what I needed today."
- "Wow, you're such a genius."
- "I love working overtime on weekends."

**Moderately Sarcastic (Expected 40-70%)**
- "That's absolutely wonderful news!"
- "Sure, that makes perfect sense."
- "I'm sooo happy for you, really."

**Genuine (Expected 0-39%)**
- "Thank you for your help with this project."
- "I really appreciate your feedback."
- "Have a great day!"

## 🔧 Troubleshooting

### Common Issues

**"No module named 'pyaudio'"**
```bash
# On Windows
pip install pyaudio

# On macOS
brew install portaudio
pip install pyaudio

# On Linux
sudo apt-get install portaudio19-dev python3-pyaudio
pip install pyaudio
```

**"Permission denied" for microphone**
- Check system microphone permissions
- Run as administrator if needed
- Ensure no other applications are using the microphone

**API errors**
- Verify your Gemini API key is correct
- Check internet connection
- Ensure API quotas aren't exceeded

### Performance Tips
- Use headphones to prevent audio feedback
- Speak clearly and at normal volume
- Ensure good microphone quality for best results

## 📁 Project Structure

```
sarcastisense/
├── sarcastisense.py          # Main application
├── requirements.txt          # Dependencies
├── README.md                # This file
├── examples/
│   ├── test_phrases.txt     # Sample phrases for testing
│   └── demo_output.txt      # Example output
└── docs/
    ├── architecture.md      # Detailed architecture
    └── api_setup.md        # API setup guide
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📋 Technical Details

### Speech-to-Text
- **Engine**: OpenAI Whisper (base model)
- **Language**: English (configurable)
- **Processing**: 3-second audio chunks
- **Format**: 16kHz, mono, 16-bit

### Sarcasm Analysis
- **AI Model**: Google Gemini 1.5 Flash
- **Features**: Text pattern analysis, keyword detection
- **Output**: JSON with score, confidence, type, explanation

### Emotion Detection
- **Method**: Audio characteristic analysis
- **Features**: Volume, pitch variance, tone analysis
- **Integration**: Tone-text mismatch detection

## 🎯 Future Enhancements

- [ ] Web interface for remote monitoring
- [ ] Multiple language support
- [ ] Advanced emotion models (SpeechBrain integration)
- [ ] Real-time visualizations and charts
- [ ] Export detection reports
- [ ] Voice response with sarcastic replies
- [ ] Integration with video calls (Zoom, Teams)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenAI Whisper for speech recognition
- Google Gemini for AI-powered analysis
- SpeechBrain community for emotion detection research
- The open-source audio processing community

## 📞 Support

For questions or issues:
- 📧 Email: safeer@supportiyo.support
- 🐛 GitHub Issues: [Create an issue](https://github.com/yourusername/real-time-sarcasm-detector/issues)

---

**Made with ❤️ for better human-AI communication**
