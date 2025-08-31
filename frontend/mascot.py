import sys
import os
import requests
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QTextEdit, QHBoxLayout, QLineEdit, QFrame, QSizePolicy
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, pyqtSlot, QTimer
from PyQt6.QtGui import QPixmap, QMovie, QIcon, QFont, QColor, QPainter, QBrush

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
        self.worker = None
        self.status = "ready"  # ready, listening, thinking, error
        self.thinking_timer = None
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("AI Tutor Mascot")
        self.setGeometry(100, 100, 900, 650)
        self.setStyleSheet("""
            QWidget {
                background-color: #181A20;
                color: #F3F3F3;
                font-family: 'Segoe UI', 'Arial', sans-serif;
            }
        """)

        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Left Panel
        left_panel = QVBoxLayout()
        left_panel.setAlignment(Qt.AlignmentFlag.AlignTop)
        left_panel.setContentsMargins(40, 40, 20, 40)

        # Header with icon and title
        header_layout = QHBoxLayout()
        mascot_icon = QLabel()
        mascot_icon.setPixmap(QPixmap("assets/mascot/mascot_happy.gif").scaled(40, 40, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        header_layout.addWidget(mascot_icon)
        title = QLabel("<span style='font-size:28pt; font-weight:700; font-family:Segoe UI,Arial,sans-serif; letter-spacing:1px;'>AI Tutor Mascot</span>")
        title.setStyleSheet("color: #A084E8;")
        header_layout.addWidget(title)
        header_layout.addStretch()
        left_panel.addLayout(header_layout)
        left_panel.addSpacing(20)

        # Mascot in circular frame with dynamic glow
        self.mascot_frame = MascotGlowFrame(self)
        self.mascot_frame.setFixedSize(240, 240)
        left_panel.addWidget(self.mascot_frame, alignment=Qt.AlignmentFlag.AlignHCenter)
        left_panel.addSpacing(20)

        # Controls: Start Speaking button and settings icon
        controls_layout = QHBoxLayout()
        self.mic_button = QPushButton("🗣️ Start Speaking")
        self.mic_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border-radius: 30px;
                padding: 16px 32px;
                font-size: 20px;
                font-weight: 600;
                border: none;
                box-shadow: 0 4px #999;
            }
            QPushButton:pressed {
                box-shadow: 0 2px #666;
                transform: translateY(2px);
            }
        """)
        self.mic_button.clicked.connect(self.start_listening)
        controls_layout.addWidget(self.mic_button)
        # Settings icon
        self.settings_button = QPushButton()
        self.settings_button.setIcon(QIcon.fromTheme("settings", QIcon("assets/mascot/mascot_happy.gif")))
        self.settings_button.setFixedSize(44, 44)
        self.settings_button.setStyleSheet("""
            QPushButton {
                background: transparent;
                border: none;
            }
            QPushButton:hover {
                background: #222;
                border-radius: 22px;
            }
        """)
        controls_layout.addWidget(self.settings_button)
        left_panel.addLayout(controls_layout)
        left_panel.addStretch()

        main_layout.addLayout(left_panel, 2)

        # Right Panel
        right_panel = QVBoxLayout()
        right_panel.setContentsMargins(0, 40, 40, 40)

        # Conversation header with optional search/history icon
        conv_header = QHBoxLayout()
        conv_title = QLabel("<span style='font-size:22pt; font-weight:600;'>Conversation</span>")
        conv_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        conv_title.setStyleSheet("color: #A084E8;")
        conv_header.addWidget(conv_title)
        # Optional search/history icon (use a static icon or placeholder)
        search_icon = QLabel()
        # Try to use a static search icon from the system, fallback to a Unicode magnifier
        search_icon.setPixmap(QIcon.fromTheme("edit-find").pixmap(28, 28))
        if search_icon.pixmap() is None or search_icon.pixmap().isNull():
            search_icon.setText("🔍")
            search_icon.setStyleSheet("font-size: 22px; color: #A084E8;")
        conv_header.addWidget(search_icon)
        right_panel.addLayout(conv_header)

        # Chat display area (bubbles) inside a scroll area
        from PyQt6.QtWidgets import QScrollArea
        self.chat_scroll = QScrollArea()
        self.chat_scroll.setWidgetResizable(True)
        self.chat_scroll.setStyleSheet("background: #23243A; border-radius: 18px; padding: 0px;"
                                       "QScrollBar:vertical, QScrollBar:horizontal {width:0px; height:0px; background:transparent;}")
        self.chat_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.chat_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.chat_area = ChatBubbleArea()
        self.chat_area.setStyleSheet("background: transparent; border-radius: 18px; padding: 18px;")
        self.chat_scroll.setWidget(self.chat_area)
        right_panel.addWidget(self.chat_scroll, 8)

        # Typing/thinking indicator
        self.thinking_indicator = ThinkingIndicator()
        self.thinking_indicator.setVisible(False)
        right_panel.addWidget(self.thinking_indicator)

        # Input area at bottom
        input_layout = QHBoxLayout()
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Type your message here...")
        self.input_field.setStyleSheet("""
            QLineEdit {
                background: #23243A;
                color: #F3F3F3;
                border-radius: 18px;
                padding: 12px 18px;
                font-size: 16px;
                border: 2px solid #A084E8;
            }
        """)
        input_layout.addWidget(self.input_field, 8)
        # Microphone icon
        self.mic_icon = QPushButton()
        self.mic_icon.setIcon(QIcon.fromTheme("microphone", QIcon("assets/mascot/mascot_happy.gif")))
        self.mic_icon.setFixedSize(44, 44)
        self.mic_icon.setStyleSheet("""
            QPushButton {
                background: #4CAF50;
                border-radius: 22px;
                color: white;
                font-size: 22px;
            }
            QPushButton:pressed {
                background: #388E3C;
            }
        """)
        self.mic_icon.clicked.connect(self.start_listening)
        input_layout.addWidget(self.mic_icon)
        # Optional send button
        self.send_button = QPushButton()
        self.send_button.setIcon(QIcon.fromTheme("send"))
        self.send_button.setFixedSize(44, 44)
        self.send_button.setStyleSheet("""
            QPushButton {
                background: #A084E8;
                border-radius: 22px;
                color: white;
                font-size: 22px;
            }
            QPushButton:pressed {
                background: #7A5AE0;
            }
        """)
        self.send_button.clicked.connect(self.send_text_input)
        input_layout.addWidget(self.send_button)
        right_panel.addLayout(input_layout)

        main_layout.addLayout(right_panel, 5)

        # Set mascot to neutral at start
        self.update_mascot("neutral")

    def update_mascot(self, emotion):
        """Updates the mascot animation and frame glow based on the emotion state using GIFs."""
        gif_map = {
            "happy": "assets/mascot/mascot_happy.gif",
            "sad": "assets/mascot/mascot_sad.gif",
            "explaining": "assets/mascot/mascot_think.gif",
            "curious": "assets/mascot/mascot_think.gif",
            "neutral": "assets/mascot/mascot_think.gif"
        }
        gif_path = gif_map.get(emotion, "assets/mascot/mascot_think.gif")
        self.mascot_frame.set_gif(gif_path)
        # Set glow color based on status
        if emotion in ["happy", "curious"]:
            self.status = "listening"
        elif emotion == "explaining":
            self.status = "thinking"
        elif emotion == "sad":
            self.status = "error"
        else:
            self.status = "ready"
        self.mascot_frame.set_status(self.status)

    def add_chat_bubble(self, text, sender="user", status=None):
        # Remove any previous temporary bubbles if present
        self.chat_area.remove_temp_bubbles()
        self.chat_area.add_bubble(text, sender, status)
        # Auto-scroll to bottom
        scroll_bar = self.chat_scroll.verticalScrollBar()
        if scroll_bar is not None:
            scroll_bar.setValue(scroll_bar.maximum())

    @pyqtSlot()
    def start_listening(self):
        self.mic_button.setEnabled(False)
        self.mic_icon.setEnabled(False)
        self.update_mascot("curious")
        self.add_chat_bubble("Listening...", sender="user", status="listening")
        self.thinking_indicator.setVisible(False)
        self.listening_bubble_active = True

        try:
            user_text = listen_and_transcribe()
        except Exception as e:
            print(f"Exception in listen_and_transcribe: {e}")
            self.add_chat_bubble(f"Error during speech recognition: {e}", sender="mascot", status="error")
            user_text = None

        if user_text:
            self.chat_history.append({"role": "user", "text": user_text})
            # Remove listening bubble before adding user text
            self.chat_area.remove_temp_bubbles()
            self.add_chat_bubble(user_text, sender="user")
            self.add_chat_bubble("Thinking...", sender="mascot", status="thinking")
            self.thinking_bubble_active = True
            self.update_mascot("explaining")
            self.thinking_indicator.setVisible(True)
            self.thinking_indicator.start()
            try:
                self.worker = WorkerThread(user_text, self.chat_history)
                self.worker.finished.connect(self.on_api_response)
                self.worker.error.connect(self.on_api_error)
                self.worker.start()
            except Exception as e:
                print(f"Exception in WorkerThread: {e}")
                self.add_chat_bubble(f"Error during response generation: {e}", sender="mascot", status="error")
                self.mic_button.setEnabled(True)
                self.mic_icon.setEnabled(True)
                self.update_mascot("neutral")
                self.thinking_indicator.setVisible(False)
        else:
            self.add_chat_bubble("I didn't catch that. Please try again.", sender="mascot", status="error")
            self.mic_button.setEnabled(True)
            self.mic_icon.setEnabled(True)
            self.update_mascot("neutral")
            self.thinking_indicator.setVisible(False)

    def send_text_input(self):
        user_text = self.input_field.text().strip()
        if not user_text:
            return
        self.input_field.clear()
        self.mic_button.setEnabled(False)
        self.mic_icon.setEnabled(False)
        self.update_mascot("curious")
        # Remove listening bubble before adding user text
        self.chat_area.remove_temp_bubbles()
        self.add_chat_bubble(user_text, sender="user")
        self.add_chat_bubble("Thinking...", sender="mascot", status="thinking")
        self.thinking_bubble_active = True
        self.thinking_indicator.setVisible(True)
        self.thinking_indicator.start()
        try:
            self.worker = WorkerThread(user_text, self.chat_history)
            self.worker.finished.connect(self.on_api_response)
            self.worker.error.connect(self.on_api_error)
            self.worker.start()
        except Exception as e:
            print(f"Exception in WorkerThread: {e}")
            self.add_chat_bubble(f"Error during response generation: {e}", sender="mascot", status="error")
            self.mic_button.setEnabled(True)
            self.mic_icon.setEnabled(True)
            self.update_mascot("neutral")
            self.thinking_indicator.setVisible(False)

    @pyqtSlot(dict)
    def on_api_response(self, response):
        ai_text = response.get("text", "I'm sorry, I don't have an answer.")
        ai_emotion = response.get("emotion", "neutral")
        self.chat_history.append({"role": "ai", "text": ai_text, "emotion": ai_emotion})
        # Remove thinking bubble before adding mascot response
        self.chat_area.remove_temp_bubbles()
        self.add_chat_bubble(ai_text, sender="mascot")
        self.update_mascot(ai_emotion)
        self.mic_button.setEnabled(True)
        self.mic_icon.setEnabled(True)
        self.thinking_indicator.setVisible(False)
        self.thinking_indicator.stop()

    @pyqtSlot(str)
    def on_api_error(self, error_message):
        # Remove thinking bubble before adding error
        self.chat_area.remove_temp_bubbles()
        self.add_chat_bubble(error_message, sender="mascot", status="error")
        self.update_mascot("sad")
        self.mic_button.setEnabled(True)
        self.mic_icon.setEnabled(True)
        self.thinking_indicator.setVisible(False)
        self.thinking_indicator.stop()


# --- Custom Widgets ---
class MascotGlowFrame(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.status = "ready"

        # Create a child QLabel for the mascot GIF
        self.mascot_label = QLabel(self)
        self.mascot_label.setScaledContents(True)
        self.mascot_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        # Set up a layout for the glow frame to center the mascot label
        layout = QVBoxLayout(self)
        layout.addWidget(self.mascot_label, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.setContentsMargins(10, 10, 10, 10)  # Add padding for the glow

        self.gif_path = "assets/mascot/mascot_think.gif"
        self.mascot_movie = QMovie(self.gif_path)
        self.mascot_label.setMovie(self.mascot_movie)
        self.mascot_movie.start()


    def set_gif(self, gif_path):
        if self.gif_path != gif_path:
            self.gif_path = gif_path
            self.mascot_movie.stop()
            self.mascot_movie = QMovie(self.gif_path)
            self.mascot_label.setMovie(self.mascot_movie)
            self.mascot_movie.start()

    def set_status(self, status):
        self.status = status
        self.update()

    def paintEvent(self, event):
        # No glow or border, just default QLabel painting
        super().paintEvent(event)


class ChatBubbleArea(QFrame):
    def __init__(self):
        super().__init__()
        self.bubble_layout = QVBoxLayout(self)
        self.bubble_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.bubble_layout.setSpacing(10)
        self.bubble_layout.setContentsMargins(10, 10, 10, 10)
        self.temp_bubbles = []  # Track temporary bubbles (listening/thinking)

    def add_bubble(self, text, sender="user", status=None):
        bubble = ChatBubble(text, sender, status)
        self.bubble_layout.addWidget(bubble)
        # Track temp bubbles for removal
        if status in ("listening", "thinking"):
            self.temp_bubbles.append(bubble)

    def remove_temp_bubbles(self):
        for bubble in self.temp_bubbles:
            self.bubble_layout.removeWidget(bubble)
            bubble.deleteLater()
        self.temp_bubbles.clear()


class ChatBubble(QLabel):
    def __init__(self, text, sender="user", status=None):
        super().__init__()
        self.setWordWrap(True)
        self.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        if sender == "user":
            color = "#4FC3F7" if status != "listening" else "#4CAF50"
            align = Qt.AlignmentFlag.AlignRight
            style = f"background: {color}; color: #181A20; border-radius: 18px; padding: 12px 18px; font-size: 16px; margin-left: 80px;"
        else:
            color = "#A084E8" if status != "error" else "#E84A5F"
            align = Qt.AlignmentFlag.AlignLeft
            style = f"background: {color}; color: #F3F3F3; border-radius: 18px; padding: 12px 18px; font-size: 16px; margin-right: 80px;"
        self.setStyleSheet(style)
        self.setAlignment(align)
        self.setText(text)


class ThinkingIndicator(QLabel):
    def __init__(self):
        super().__init__()
        self.setFixedHeight(36)
        self.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.dots = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.animate)
        self.setStyleSheet("font-size: 22px; color: #A084E8; padding-left: 12px;")

    def start(self):
        self.dots = 0
        self.setText("<b>Mascot is thinking</b> ")
        self.timer.start(500)

    def stop(self):
        self.timer.stop()
        self.setText("")

    def animate(self):
        self.dots = (self.dots + 1) % 4
        self.setText("<b>Mascot is thinking</b> " + "." * self.dots)



if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MascotUI()
    window.show()
    sys.exit(app.exec())