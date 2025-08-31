from fastapi import FastAPI, Response
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from rag_pipeline import rag_query

app = FastAPI(title="AI Tutor Mascot Backend")

# Root endpoint for homepage
@app.get("/", response_class=HTMLResponse)
def read_root():
    return "<h2>AI Tutor Mascot Backend is running.</h2>"

# Favicon endpoint to prevent 404
@app.get("/favicon.ico")
def favicon():
    return Response(content="", media_type="image/x-icon")

class QueryRequest(BaseModel):
    question: str
    history: list = []

class ResponseModel(BaseModel):
    text: str
    emotion: str

@app.post("/query", response_model=ResponseModel)
def query(request: QueryRequest):
    answer, emotion = rag_query(request.question, request.history)
    return {"text": answer, "emotion": emotion}

@app.post("/chat", response_model=ResponseModel)
def chat(request: QueryRequest):
    answer, emotion = rag_query(request.question, request.history)
    return {"text": answer, "emotion": emotion}
import sys
import os
import requests
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QTextEdit, QHBoxLayout
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QPixmap, QMovie

# Add the parent directory to the system path to allow imports from stt_tts and backend
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stt_tts.stt import listen_and_transcribe
from stt_tts.tts import speak

# Constants
BACKEND_URL = "http://127.0.0.1:8000"


def make_api_call(user_question, chat_history):
    """
    Sends a query to the FastAPI backend and returns the response.
    """
    try:
        # Prepare the conversation history for the API call
        history_payload = [
            {"question": msg["text"], "answer": msg["text"]}
            for msg in chat_history if
            "text" in msg and ("role" in msg and (msg["role"] == "user" or msg["role"] == "ai"))
        ]

        payload = {
            "question": user_question,
            "history": history_payload
        }

        response = requests.post(
            f"{BACKEND_URL}/chat",  # Use the chat endpoint for multi-turn conversation
            json=payload,
            headers={"Content-Type": "application/json"}
        )

        response.raise_for_status()  # Raise an exception for bad status codes
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error making API call to backend: {e}")
        return None


# Worker thread for handling API and TTS calls to prevent UI freezing
class WorkerThread(QThread):
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, user_text, chat_history):
        super().__init__()
        self.user_text = user_text
        self.chat_history = chat_history

    def run(self):
        # API call
        response = make_api_call(self.user_text, self.chat_history)
        if response:
            # TTS call (This might still be blocking, but is in a separate thread)
            speak(response["text"], response["emotion"])
            self.finished.emit(response)
        else:
            self.error.emit("Failed to get response from backend.")


class MascotUI(QWidget):
    def __init__(self):
        super().__init__()
        self.chat_history = []
        self.init_ui()
        self.worker = None

    def init_ui(self):
        self.setWindowTitle("AI Tutor Mascot")
        self.setGeometry(100, 100, 800, 600)

        main_layout = QHBoxLayout()

        # Left Panel (Mascot and Mic Button)
        left_panel = QVBoxLayout()
        left_panel.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.mascot_label = QLabel()
        self.mascot_label.setFixedSize(200, 200)
        self.mascot_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.update_mascot("neutral")
        left_panel.addWidget(self.mascot_label)

        self.mic_button = QPushButton("🗣️ Start Speaking")
        self.mic_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border-radius: 25px;
                padding: 10px 20px;
                font-size: 16px;
                border: none;
                box-shadow: 0 4px #999;
            }
            QPushButton:pressed {
                box-shadow: 0 2px #666;
                transform: translateY(2px);
            }
        """)
        self.mic_button.clicked.connect(self.start_listening)
        left_panel.addWidget(self.mic_button)

        main_layout.addLayout(left_panel)

        # Right Panel (Conversation)
        right_panel = QVBoxLayout()
        self.conversation_area = QTextEdit()
        self.conversation_area.setReadOnly(True)
        self.conversation_area.setHtml("<h3>Conversation</h3><hr>")
        right_panel.addWidget(self.conversation_area)

        main_layout.addLayout(right_panel)

        self.setLayout(main_layout)

    def update_mascot(self, emotion):
        """Updates the mascot image based on the emotion state."""
        # Using a simple image change for now. For animations, you'd use a QMovie.
        image_path = f"path/to/mascot_{emotion}.png"  # Placeholder
        if not os.path.exists(image_path):
            image_path = "path/to/mascot_neutral.png"  # Fallback

        self.mascot_label.setPixmap(
            QPixmap(image_path).scaled(self.mascot_label.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                       Qt.TransformationMode.SmoothTransformation))

    @pyqtSlot()
    def start_listening(self):
        self.mic_button.setEnabled(False)
        self.update_mascot("curious")
        self.conversation_area.append("<br><b>You:</b> Listening...")

        try:
            user_text = listen_and_transcribe()
        except Exception as e:
            print(f"Exception in listen_and_transcribe: {e}")
            self.conversation_area.append(f"Error during speech recognition: {e}")
            user_text = None

        if user_text:
            self.chat_history.append({"role": "user", "text": user_text})
            self.conversation_area.append(user_text)

            self.conversation_area.append("<br><b>Mascot:</b> Thinking...")
            self.update_mascot("explaining")

            try:
                self.worker = WorkerThread(user_text, self.chat_history)
                self.worker.finished.connect(self.on_api_response)
                self.worker.error.connect(self.on_api_error)
                self.worker.start()
            except Exception as e:
                print(f"Exception in WorkerThread: {e}")
                self.conversation_area.append(f"Error during response generation: {e}")
                self.mic_button.setEnabled(True)
                self.update_mascot("neutral")
        else:
            self.conversation_area.append("I didn't catch that. Please try again.")
            self.mic_button.setEnabled(True)
            self.update_mascot("neutral")

    @pyqtSlot(dict)
    def on_api_response(self, response):
        ai_text = response.get("text", "I'm sorry, I don't have an answer.")
        ai_emotion = response.get("emotion", "neutral")

        self.chat_history.append({"role": "ai", "text": ai_text, "emotion": ai_emotion})
        self.conversation_area.append(f"<br><b>Mascot:</b> {ai_text}")
        self.update_mascot(ai_emotion)
        self.mic_button.setEnabled(True)

    @pyqtSlot(str)
    def on_api_error(self, error_message):
        self.conversation_area.append(f"<br><b>Mascot:</b> {error_message}")
        self.update_mascot("sad")
        self.mic_button.setEnabled(True)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MascotUI()
    window.show()
    sys.exit(app.exec())
