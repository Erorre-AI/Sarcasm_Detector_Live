"""
SarcastiSense - Real-Time Sarcasm Detection System
Combines speech-to-text, emotion analysis, and AI-powered sarcasm detection
"""

import os
import time
import json
import threading
import queue
import numpy as np
import pyaudio
import wave
import whisper
import sounddevice as sd
from scipy.io.wavfile import write
import google.generativeai as genai
from speechbrain.inference.interfaces import foreign_class
import keyboard
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
from datetime import datetime

# Configuration
GOOGLE_API_KEY = "AIzaSyB2SL5vL7XeHPtwOzZDcDmIVgQ1yLL3B0U"  # Replace with your actual API key
AUDIO_CONFIG = {
    'chunk': 1024,
    'format': pyaudio.paInt16,
    'channels': 1,
    'rate': 16000,
    'record_seconds': 5
}

@dataclass
class SarcasmResult:
    """Data class for sarcasm detection results"""
    timestamp: str
    text: str
    sarcasm_score: int
    confidence: int
    sarcasm_type: str
    explanation: str
    keywords: List[str]
    emotion: str
    emotion_score: float

class RealTimeSarcasmDetector:
    """Main class for real-time sarcasm detection"""
    
    def __init__(self):
        self.whisper_model = None
        self.emotion_classifier = None
        self.gemini_model = None
        self.is_running = False
        self.results_queue = queue.Queue()
        self.audio_temp_file = "temp_audio.wav"
        
    def initialize_models(self):
        """Initialize all AI models"""
        print("🚀 Initializing SarcastiSense...")
        
        # Initialize Whisper for speech-to-text
        print("📝 Loading Whisper model...")
        self.whisper_model = whisper.load_model("base")
        print("✅ Whisper model loaded!")
        
        # Initialize emotion classifier
        print("😊 Loading emotion classifier...")
        try:
            self.emotion_classifier = foreign_class(
                source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
                pymodule_file="custom_interface.py",
                classname="CustomEncoderWav2vec2Classifier"
            )
            print("✅ Emotion classifier loaded!")
        except Exception as e:
            print(f"⚠️ Emotion classifier failed to load: {e}")
            self.emotion_classifier = None
        
        # Initialize Gemini for sarcasm detection
        print("🧠 Initializing Gemini AI...")
        try:
            genai.configure(api_key=GOOGLE_API_KEY)
            self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
            print("✅ Gemini AI initialized!")
        except Exception as e:
            print(f"⚠️ Gemini AI failed to initialize: {e}")
            self.gemini_model = None
    
    def capture_audio_chunk(self, duration: int = 3) -> np.ndarray:
        """Capture audio chunk from microphone"""
        audio = pyaudio.PyAudio()
        
        try:
            stream = audio.open(
                format=AUDIO_CONFIG['format'],
                channels=AUDIO_CONFIG['channels'],
                rate=AUDIO_CONFIG['rate'],
                input=True,
                frames_per_buffer=AUDIO_CONFIG['chunk']
            )
            
            frames = []
            for _ in range(0, int(AUDIO_CONFIG['rate'] / AUDIO_CONFIG['chunk'] * duration)):
                data = stream.read(AUDIO_CONFIG['chunk'])
                frames.append(data)
            
            stream.stop_stream()
            stream.close()
            
            # Convert to numpy array and normalize
            audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
            audio_data = audio_data.astype(np.float32) / 32768.0
            
            return audio_data
            
        finally:
            audio.terminate()
    
    def save_audio_for_emotion(self, audio_data: np.ndarray):
        """Save audio data for emotion analysis"""
        # Convert back to int16 for saving
        audio_int16 = (audio_data * 32768).astype(np.int16)
        write(self.audio_temp_file, AUDIO_CONFIG['rate'], audio_int16)
    
    def transcribe_audio(self, audio_data: np.ndarray) -> str:
        """Convert audio to text using Whisper"""
        if self.whisper_model is None:
            return ""
        
        try:
            result = self.whisper_model.transcribe(audio_data, language="en")
            return result["text"].strip()
        except Exception as e:
            print(f"❌ Transcription error: {e}")
            return ""
    
    def analyze_emotion(self) -> Tuple[str, float]:
        """Analyze emotion from audio file"""
        if self.emotion_classifier is None or not os.path.exists(self.audio_temp_file):
            return "unknown", 0.0
        
        try:
            out_prob, score, index, text_lab = self.emotion_classifier.classify_file(self.audio_temp_file)
            # Handle case where text_lab might be a list
            if isinstance(text_lab, list):
                emotion_label = text_lab[0] if text_lab else "unknown"
            else:
                emotion_label = str(text_lab)
            return emotion_label, float(score)
        except Exception as e:
            print(f"❌ Emotion analysis error: {e}")
            return "unknown", 0.0
    
    def create_sarcasm_prompt(self, text: str, emotion: str = None) -> str:
        """Create prompt for sarcasm detection"""
        emotion_context = f"\nVOICE EMOTION DETECTED: {emotion}" if emotion and emotion != "unknown" else ""
        
        prompt = f"""
You are an expert at detecting sarcasm, passive-aggression, and mock politeness in human speech.
Analyze the following text and provide a detailed assessment.

TEXT TO ANALYZE: "{text}"{emotion_context}

Please respond with a JSON object containing:
1. "sarcasm_score": A number from 0-100 indicating how sarcastic this text is
2. "confidence": A number from 0-100 indicating your confidence in this assessment  
3. "sarcasm_type": One of ["genuine", "sarcastic", "passive_aggressive", "mock_polite"]
4. "explanation": A brief explanation of why you classified it this way
5. "keywords": List of words/phrases that indicate sarcasm or insincerity

DETECTION GUIDELINES:
- Sarcastic: Saying the opposite of what is meant, often with irony
- Passive Aggressive: Indirect expression of negative feelings
- Mock Polite: Fake politeness or overly formal language hiding negativity
- Genuine: Sincere, honest communication

Consider:
- Exaggerated positivity ("sooo happy", "absolutely wonderful")
- Contradictory tone indicators
- Overly formal language in casual contexts
- Repeated emphasis or punctuation
- Context clues and common sarcastic patterns
- Voice emotion as additional context

Respond ONLY with valid JSON format.
"""
        return prompt
    
    def detect_sarcasm(self, text: str, emotion: str = None) -> Optional[Dict]:
        """Detect sarcasm using Gemini AI"""
        if self.gemini_model is None or not text.strip():
            return None
        
        try:
            prompt = self.create_sarcasm_prompt(text, emotion)
            response = self.gemini_model.generate_content(prompt)
            
            # Parse JSON response
            response_text = response.text.strip()
            if response_text.startswith('```json'):
                response_text = response_text[7:-3]
            elif response_text.startswith('```'):
                response_text = response_text[3:-3]
            
            return json.loads(response_text)
            
        except Exception as e:
            print(f"❌ Sarcasm detection error: {e}")
            return None
    
    def process_audio_chunk(self) -> Optional[SarcasmResult]:
        """Process a single audio chunk through the entire pipeline"""
        try:
            # Capture audio
            audio_data = self.capture_audio_chunk()
            
            # Check if there's actual speech (not silence)
            if np.max(np.abs(audio_data)) < 0.01:
                return None
            
            # Save audio for emotion analysis
            self.save_audio_for_emotion(audio_data)
            
            # Transcribe speech to text
            text = self.transcribe_audio(audio_data)
            if not text.strip():
                return None
            
            # Analyze emotion
            emotion, emotion_score = self.analyze_emotion()
            
            # Detect sarcasm
            sarcasm_result = self.detect_sarcasm(text, emotion)
            if sarcasm_result is None:
                return None
            
            # Create result object
            result = SarcasmResult(
                timestamp=datetime.now().strftime("%H:%M:%S"),
                text=text,
                sarcasm_score=sarcasm_result.get('sarcasm_score', 0),
                confidence=sarcasm_result.get('confidence', 0),
                sarcasm_type=sarcasm_result.get('sarcasm_type', 'unknown'),
                explanation=sarcasm_result.get('explanation', ''),
                keywords=sarcasm_result.get('keywords', []),
                emotion=emotion,
                emotion_score=emotion_score
            )
            
            return result
            
        except Exception as e:
            print(f"❌ Processing error: {e}")
            return None
    
    def display_result(self, result: SarcasmResult):
        """Display analysis result with color coding"""
        # Color coding based on sarcasm score
        if result.sarcasm_score >= 70:
            color = "\033[91m"  # Red - High sarcasm
            indicator = "🔥 HIGH SARCASM"
        elif result.sarcasm_score >= 40:
            color = "\033[93m"  # Yellow - Medium sarcasm
            indicator = "⚠️  MEDIUM SARCASM"
        else:
            color = "\033[92m"  # Green - Low/No sarcasm
            indicator = "✅ LOW SARCASM"
        
        # Safely handle emotion display
        emotion_display = str(result.emotion).upper() if result.emotion else "UNKNOWN"
        
        print(f"\n{color}{'='*60}")
        print(f"[{result.timestamp}] {indicator}")
        print(f"{'='*60}\033[0m")
        print(f"📢 Text: \"{result.text}\"")
        print(f"🎯 Sarcasm Score: {result.sarcasm_score}/100 (Confidence: {result.confidence}%)")
        print(f"🏷️  Type: {result.sarcasm_type.upper()}")
        print(f"😊 Emotion: {emotion_display} ({result.emotion_score:.2f})")
        print(f"💡 Explanation: {result.explanation}")
        if result.keywords:
            print(f"🔍 Key Indicators: {', '.join(result.keywords)}")
        print(f"{color}{'='*60}\033[0m\n")
    
    def run_detection_loop(self):
        """Main detection loop"""
        print("\n🎙️  SarcastiSense is now listening...")
        print("💬 Speak naturally - the system will analyze your speech for sarcasm")
        print("⌨️  Press 'q' to quit\n")
        print("🔄 Listening for speech...")
        
        while self.is_running:
            try:
                if keyboard.is_pressed('q'):
                    break
                
                result = self.process_audio_chunk()
                if result:
                    self.display_result(result)
                else:
                    print("🔊 Listening... (silence or no speech detected)")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ Error in detection loop: {e}")
                time.sleep(1)
    
    def cleanup(self):
        """Clean up temporary files"""
        if os.path.exists(self.audio_temp_file):
            os.remove(self.audio_temp_file)
    
    def run(self):
        """Main entry point"""
        try:
            self.initialize_models()
            self.is_running = True
            self.run_detection_loop()
        except KeyboardInterrupt:
            print("\n🛑 Detection stopped by user")
        finally:
            self.is_running = False
            self.cleanup()
            print("👋 SarcastiSense shutdown complete")

def test_individual_components():
    """Test each component individually"""
    print("🧪 Testing individual components...\n")
    
    detector = RealTimeSarcasmDetector()
    
    # Test audio capture
    print("1. Testing audio capture...")
    try:
        audio_data = detector.capture_audio_chunk(2)
        print(f"✅ Audio captured: shape {audio_data.shape}")
    except Exception as e:
        print(f"❌ Audio capture failed: {e}")
        return False
    
    # Test Whisper
    print("2. Testing speech-to-text...")
    try:
        detector.whisper_model = whisper.load_model("base")
        print("✅ Whisper model loaded")
    except Exception as e:
        print(f"❌ Whisper failed: {e}")
    
    # Test Gemini
    print("3. Testing sarcasm detection...")
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel('gemini-1.5-flash')
        result = detector.detect_sarcasm("Oh great, another meeting!", "neutral")
        if result:
            print("✅ Sarcasm detection working")
        else:
            print("⚠️ Sarcasm detection returned no result")
    except Exception as e:
        print(f"❌ Sarcasm detection failed: {e}")
    
    print("\n🧪 Component testing complete!")
    return True

def main():
    """Main function"""
    print("🎯 SarcastiSense - Real-Time Sarcasm Detection System")
    print("=" * 60)
    
    # Check if user wants to run tests first
    run_tests = input("🧪 Run component tests first? (y/N): ").lower().strip() == 'y'
    
    if run_tests:
        if not test_individual_components():
            print("❌ Some tests failed. Continue anyway? (y/N): ", end="")
            if input().lower().strip() != 'y':
                return
    
    print("\n🚀 Starting SarcastiSense...")
    detector = RealTimeSarcasmDetector()
    detector.run()

if __name__ == "__main__":
    main()