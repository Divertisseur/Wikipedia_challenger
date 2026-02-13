"""
Wikipedia Challenger – PyQt6 GUI
A modern dark-themed desktop interface for the Wikipedia path-finder bot.
"""

import sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QComboBox, QPushButton, QTextEdit, QFrame,
    QGroupBox, QSplitter, QSizePolicy, QTextBrowser, QCheckBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt6.QtGui import QFont, QColor, QIcon, QTextCursor

from wikipedia_bot import WikipediaChallenger

# ── Language catalogue ──────────────────────────────────────────────
LANGUAGES = [
    ("English",    "en"),
    ("Français",   "fr"),
    ("Deutsch",    "de"),
    ("Español",    "es"),
    ("Italiano",   "it"),
    ("Português",  "pt"),
    ("Nederlands", "nl"),
    ("日本語",      "ja"),
    ("中文",        "zh"),
    ("Русский",    "ru"),
]


# ── Worker thread ───────────────────────────────────────────────────
class SearchWorker(QThread):
    """Runs the BFS search on a background thread."""
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(dict)   # emits the result dict
    error_signal = pyqtSignal(str)

    def __init__(self, start: str, end: str, lang: str, mode: str, use_metadata: bool = False):
        super().__init__()
        self.start_page = start
        self.end_page = end
        self.lang = lang
        self.mode = mode
        self.use_metadata = use_metadata
        self.bot: WikipediaChallenger | None = None

    def run(self):
        try:
            self.bot = WikipediaChallenger(
                lang=self.lang,
                log_callback=self.log_signal.emit,
            )
            result = self.bot.find_shortest_path(
                self.start_page, self.end_page, mode=self.mode, use_metadata=self.use_metadata
            )
            if result:
                self.finished_signal.emit(result)
            else:
                self.error_signal.emit("No path found (or search was cancelled).")
        except Exception as exc:
            self.error_signal.emit(str(exc))

    def cancel(self):
        if self.bot:
            self.bot.cancel()


# ── Colour palette ──────────────────────────────────────────────────
BG           = "#f1f5f9"
SURFACE      = "#ffffff"
SURFACE_ALT  = "#f8fafc"
ACCENT       = "#4f46e5"
HIGHLIGHT    = "#6366f1"
TEXT         = "#0f172a"
TEXT_DIM     = "#64748b"
SUCCESS      = "#10b981"
BORDER       = "#e2e8f0"


# ── Stylesheet ──────────────────────────────────────────────────────
STYLESHEET = f"""
QMainWindow {{
    background-color: {BG};
}}
QWidget {{
    color: {TEXT};
    font-family: 'Segoe UI', 'Inter', 'Roboto', sans-serif;
    font-size: 13px;
}}

/* ── Group boxes ───────────────────────────────── */
QGroupBox {{
    background-color: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 16px;
    margin-top: 14px;
    padding: 20px 16px 16px 16px;
    font-weight: 600;
    font-size: 14px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 20px;
    padding: 0 8px;
    color: {ACCENT};
}}

/* ── Inputs ────────────────────────────────────── */
QLineEdit {{
    background-color: {SURFACE_ALT};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 14px;
    color: {TEXT};
    selection-background-color: {HIGHLIGHT};
}}
QLineEdit:focus {{
    border-color: {HIGHLIGHT};
    background-color: {SURFACE};
}}

QComboBox {{
    background-color: {SURFACE_ALT};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 10px 14px;
    min-width: 150px;
}}
QComboBox::drop-down {{
    border: none;
    width: 28px;
}}
QComboBox QAbstractItemView {{
    background-color: {SURFACE};
    selection-background-color: {SURFACE_ALT};
    selection-color: {ACCENT};
    border: 1px solid {BORDER};
    border-radius: 8px;
}}

/* ── Buttons ───────────────────────────────────── */
QPushButton {{
    border: none;
    border-radius: 8px;
    padding: 12px 32px;
    font-weight: 700;
    font-size: 14px;
}}
QPushButton#startBtn {{
    background-color: {ACCENT};
    color: #ffffff;
}}
QPushButton#startBtn:hover {{
    background-color: #4338ca;
}}
QPushButton#startBtn:pressed {{
    background-color: #3730a3;
}}
QPushButton#startBtn:disabled {{
    background-color: #e2e8f0;
    color: #94a3b8;
}}
QPushButton#cancelBtn {{
    background-color: {SURFACE};
    border: 1px solid {BORDER};
    color: {TEXT_DIM};
}}
QPushButton#cancelBtn:hover {{
    background-color: {SURFACE_ALT};
    color: {TEXT};
}}
QPushButton#cancelBtn:pressed {{
    background-color: {BORDER};
}}
QPushButton#cancelBtn:disabled {{
    background-color: {SURFACE};
    color: {BORDER};
}}

/* ── Log area ──────────────────────────────────── */
QTextEdit {{
    background-color: {SURFACE_ALT};
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 12px;
    font-family: 'Cascadia Code', 'Consolas', 'Courier New', monospace;
    font-size: 12px;
    color: {TEXT};
}}

/* ── Labels ────────────────────────────────────── */
QLabel {{
    font-size: 13px;
    color: {TEXT_DIM};
}}
QLabel#titleLabel {{
    font-size: 28px;
    font-weight: 900;
    color: {ACCENT};
    padding: 6px 0;
}}
QLabel#subtitleLabel {{
    font-size: 13px;
    color: {TEXT_DIM};
    padding-bottom: 12px;
}}
QLabel#resultTitle {{
    font-size: 18px;
    font-weight: 700;
    color: {SUCCESS};
}}
QLabel#clicksLabel, QLabel#timeLabel, QLabel#pagesLabel {{
    font-size: 36px;
    font-weight: 800;
    color: {HIGHLIGHT};
    padding: 4px;
    min-height: 50px;
}}
QLabel#timeLabel {{
    color: {ACCENT};
}}
QLabel#pagesLabel {{
    color: {SUCCESS};
}}

/* ── Frames / splitters ────────────────────────── */
QSplitter::handle {{
    background-color: {BORDER};
    height: 2px;
}}
"""


# ── Main window ─────────────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Wikipedia Challenger")
        self.setMinimumSize(QSize(720, 700))
        self.resize(800, 750)
        self.worker: SearchWorker | None = None

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(20, 20, 20, 20)
        root.setSpacing(16)

        # ── Header ─────────────────────────────────
        header = QHBoxLayout()
        title = QLabel("🌐  Wikipedia Challenger")
        title.setObjectName("titleLabel")
        header.addWidget(title)
        header.addStretch()
        
        subtitle = QLabel("Shortest path bot")
        subtitle.setObjectName("subtitleLabel")
        header.addWidget(subtitle)
        root.addLayout(header)

        # ── Main Body (Side-by-Side) ──────────────
        body = QHBoxLayout()
        body.setSpacing(20)

        # Left Column: Settings
        left_col = QVBoxLayout()
        left_col.setSpacing(12)
        
        settings_group = QGroupBox("Search Settings")
        settings_group.setFixedWidth(300)
        settings_layout = QVBoxLayout(settings_group)
        settings_layout.setSpacing(15)

        # Language
        lang_label = QLabel("Language:")
        self.lang_combo = QComboBox()
        for display, code in LANGUAGES:
            self.lang_combo.addItem(f"{display} ({code})", code)
        settings_layout.addWidget(lang_label)
        settings_layout.addWidget(self.lang_combo)

        # Start page
        start_label = QLabel("Start page:")
        self.start_input = QLineEdit()
        self.start_input.setPlaceholderText("e.g. France")
        settings_layout.addWidget(start_label)
        settings_layout.addWidget(self.start_input)

        # End page
        end_label = QLabel("End page:")
        self.end_input = QLineEdit()
        self.end_input.setPlaceholderText("e.g. Europe")
        settings_layout.addWidget(end_label)
        settings_layout.addWidget(self.end_input)

        # Method
        method_label = QLabel("Search Method:")
        self.method_combo = QComboBox()
        self.method_combo.addItem("Likely Path (Semantic AI)", "semantic")
        self.method_combo.addItem("Fast Guess (Word Match)", "lexical")
        self.method_combo.addItem("Brute Force (BFS)", "bruteforce")
        settings_layout.addWidget(method_label)
        settings_layout.addWidget(self.method_combo)

        # Toggle
        self.intelligent_check = QCheckBox("Intelligent Context")
        self.intelligent_check.setStyleSheet(f"color: {TEXT_DIM}; font-size: 12px;")
        settings_layout.addWidget(self.intelligent_check)

        settings_layout.addStretch()

        # Buttons
        self.start_btn = QPushButton("⛏  Start")
        self.start_btn.setObjectName("startBtn")
        self.start_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.start_btn.clicked.connect(self._on_start)

        self.cancel_btn = QPushButton("✕  Cancel")
        self.cancel_btn.setObjectName("cancelBtn")
        self.cancel_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._on_cancel)

        settings_layout.addWidget(self.start_btn)
        settings_layout.addWidget(self.cancel_btn)

        left_col.addWidget(settings_group)
        body.addLayout(left_col)

        # Right Column: Results & Logs (Splitter)
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.setHandleWidth(4)

        # Result panel
        self.result_group = QGroupBox("Results & Statistics")
        result_layout = QVBoxLayout(self.result_group)
        
        self.result_title = QLabel("")
        self.result_title.setObjectName("resultTitle")
        result_layout.addWidget(self.result_title)

        stats_row = QHBoxLayout()
        stats_row.setSpacing(30)

        def create_stat_box(label_obj, caption):
            box = QVBoxLayout()
            box.addWidget(label_obj)
            cap = QLabel(caption)
            cap.setAlignment(Qt.AlignmentFlag.AlignCenter)
            box.addWidget(cap)
            return box

        self.clicks_label = QLabel("–")
        self.clicks_label.setObjectName("clicksLabel")
        self.clicks_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.time_label = QLabel("–")
        self.time_label.setObjectName("timeLabel")
        self.time_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.pages_label = QLabel("–")
        self.pages_label.setObjectName("pagesLabel")
        self.pages_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        stats_row.addLayout(create_stat_box(self.clicks_label, "clicks"))
        stats_row.addLayout(create_stat_box(self.time_label, "seconds"))
        stats_row.addLayout(create_stat_box(self.pages_label, "pages"))
        stats_row.addStretch()
        result_layout.addLayout(stats_row)

        self.path_display = QTextBrowser()
        self.path_display.setReadOnly(True)
        self.path_display.setOpenExternalLinks(True)
        self.path_display.setPlaceholderText("The shortest path will appear here…")
        result_layout.addWidget(self.path_display, 1) # Give it stretch

        splitter.addWidget(self.result_group)

        # Log panel
        log_group = QGroupBox("Live Activity")
        log_layout = QVBoxLayout(log_group)
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        log_layout.addWidget(self.log_area)
        splitter.addWidget(log_group)

        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        
        body.addWidget(splitter, 1) # Rigth side stretches
        root.addLayout(body)

        # Enter key triggers search
        self.start_input.returnPressed.connect(self._on_start)
        self.end_input.returnPressed.connect(self._on_start)

    # ── Slots ──────────────────────────────────────
    def _on_start(self):
        start = self.start_input.text().strip()
        end = self.end_input.text().strip()
        if not start or not end:
            self._append_log("⚠  Please enter both a start and end page.")
            return

        lang = self.lang_combo.currentData()
        mode = self.method_combo.currentData()
        use_metadata = self.intelligent_check.isChecked()

        # Reset UI
        self.log_area.clear()
        self.path_display.clear()
        self.result_title.setText("")
        self.clicks_label.setText("–")
        self.time_label.setText("–")
        self.pages_label.setText("–")
        self._set_running(True)

        self._append_log(f"Starting search: \"{start}\" → \"{end}\" ({lang})")
        self._append_log(f"Method: {self.method_combo.currentText()}")

        self.worker = SearchWorker(start, end, lang, mode, use_metadata=use_metadata)
        self.worker.log_signal.connect(self._append_log)
        self.worker.finished_signal.connect(self._on_finished)
        self.worker.error_signal.connect(self._on_error)
        self.worker.finished.connect(lambda: self._set_running(False))
        self.worker.start()

    def _on_cancel(self):
        if self.worker:
            self.worker.cancel()
        self._append_log("⏹  Cancel requested…")

    def _on_finished(self, result: dict):
        path = result["path"]
        clicks = result["clicks"]
        elapsed = result["time"]
        searched = result.get("searched", 0)

        self.result_title.setText("🎯  Target Found!")
        self.clicks_label.setText(str(clicks))
        self.time_label.setText(f"{elapsed:.2f}")
        self.pages_label.setText(str(searched))

        # Build path tree with HTML links
        lang = self.lang_combo.currentData()
        html_lines = []
        for i, page in enumerate(path):
            prefix = "&nbsp;&nbsp;" * i + ("└─ " if i > 0 else "")
            url_safe = page.replace(" ", "_")
            url = f"https://{lang}.wikipedia.org/wiki/{url_safe}"
            link_html = f'<a href="{url}" style="color: {HIGHLIGHT}; text-decoration: none;">{page}</a>'
            html_lines.append(f"<div>{prefix}{link_html}</div>")
        
        self.path_display.setHtml(f'<div style="color: {TEXT_DIM};">{"".join(html_lines)}</div>')

        self._append_log(f"\n✅  Done! {clicks} click(s) in {elapsed:.2f}s ({searched} pages explored)")

    def _on_error(self, msg: str):
        self.result_title.setText("❌  Search Failed")
        self.result_title.setStyleSheet(f"color: {HIGHLIGHT};")
        self._append_log(f"Error: {msg}")

    # ── Helpers ────────────────────────────────────
    def _append_log(self, text: str):
        self.log_area.append(text)
        # Auto-scroll to bottom
        cursor = self.log_area.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.log_area.setTextCursor(cursor)

    def _set_running(self, running: bool):
        self.start_btn.setEnabled(not running)
        self.cancel_btn.setEnabled(running)
        self.start_input.setEnabled(not running)
        self.end_input.setEnabled(not running)
        self.lang_combo.setEnabled(not running)


# ── Entry point ─────────────────────────────────────────────────────
def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")       # cross-platform consistent look
    app.setStyleSheet(STYLESHEET)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
