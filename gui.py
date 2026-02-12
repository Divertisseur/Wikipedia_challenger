"""
Wikipedia Challenger – PyQt6 GUI
A modern dark-themed desktop interface for the Wikipedia path-finder bot.
"""

import sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QComboBox, QPushButton, QTextEdit, QFrame,
    QGroupBox, QSplitter, QSizePolicy, QTextBrowser
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

    def __init__(self, start: str, end: str, lang: str, mode: str):
        super().__init__()
        self.start_page = start
        self.end_page = end
        self.lang = lang
        self.mode = mode
        self.bot: WikipediaChallenger | None = None

    def run(self):
        try:
            self.bot = WikipediaChallenger(
                lang=self.lang,
                log_callback=self.log_signal.emit,
            )
            result = self.bot.find_shortest_path(
                self.start_page, self.end_page, mode=self.mode
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
BG           = "#0f0f17"
SURFACE      = "#1a1a2e"
SURFACE_ALT  = "#16213e"
ACCENT       = "#0f3460"
HIGHLIGHT    = "#e94560"
TEXT         = "#eaeaea"
TEXT_DIM     = "#8a8a9a"
SUCCESS      = "#00e676"
BORDER       = "#2a2a4a"


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
    border-radius: 10px;
    margin-top: 14px;
    padding: 16px 12px 12px 12px;
    font-weight: 600;
    font-size: 14px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 16px;
    padding: 0 6px;
    color: {HIGHLIGHT};
}}

/* ── Inputs ────────────────────────────────────── */
QLineEdit {{
    background-color: {SURFACE_ALT};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 8px 12px;
    font-size: 14px;
    color: {TEXT};
    selection-background-color: {HIGHLIGHT};
}}
QLineEdit:focus {{
    border-color: {HIGHLIGHT};
}}

QComboBox {{
    background-color: {SURFACE_ALT};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 8px 12px;
    min-width: 150px;
}}
QComboBox::drop-down {{
    border: none;
    width: 28px;
}}
QComboBox QAbstractItemView {{
    background-color: {SURFACE};
    selection-background-color: {ACCENT};
    border: 1px solid {BORDER};
    border-radius: 4px;
}}

/* ── Buttons ───────────────────────────────────── */
QPushButton {{
    border: none;
    border-radius: 6px;
    padding: 10px 28px;
    font-weight: 700;
    font-size: 14px;
}}
QPushButton#startBtn {{
    background-color: {HIGHLIGHT};
    color: #ffffff;
}}
QPushButton#startBtn:hover {{
    background-color: #ff5a75;
}}
QPushButton#startBtn:pressed {{
    background-color: #c8354d;
}}
QPushButton#startBtn:disabled {{
    background-color: #5a2030;
    color: #888;
}}
QPushButton#cancelBtn {{
    background-color: {ACCENT};
    color: {TEXT};
}}
QPushButton#cancelBtn:hover {{
    background-color: #1a4a80;
}}
QPushButton#cancelBtn:disabled {{
    background-color: #0a1a30;
    color: #555;
}}

/* ── Log area ──────────────────────────────────── */
QTextEdit {{
    background-color: {SURFACE_ALT};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 10px;
    font-family: 'Cascadia Code', 'Consolas', 'Courier New', monospace;
    font-size: 12px;
    color: {TEXT_DIM};
}}

/* ── Labels ────────────────────────────────────── */
QLabel {{
    font-size: 13px;
    color: {TEXT_DIM};
}}
QLabel#titleLabel {{
    font-size: 26px;
    font-weight: 800;
    color: {HIGHLIGHT};
    padding: 4px 0;
}}
QLabel#subtitleLabel {{
    font-size: 13px;
    color: {TEXT_DIM};
    padding-bottom: 8px;
}}
QLabel#resultTitle {{
    font-size: 16px;
    font-weight: 700;
    color: {SUCCESS};
}}
QLabel#clicksLabel {{
    font-size: 36px;
    font-weight: 800;
    color: {HIGHLIGHT};
}}
QLabel#timeLabel {{
    font-size: 14px;
    color: {TEXT_DIM};
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
        self.setMinimumSize(QSize(720, 640))
        self.resize(800, 700)
        self.worker: SearchWorker | None = None

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(24, 18, 24, 18)
        root.setSpacing(12)

        # ── Header ─────────────────────────────────
        title = QLabel("🌐  Wikipedia Challenger")
        title.setObjectName("titleLabel")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(title)

        subtitle = QLabel("Find the shortest path between any two Wikipedia pages")
        subtitle.setObjectName("subtitleLabel")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(subtitle)

        # ── Settings group ─────────────────────────
        settings_group = QGroupBox("Search Settings")
        settings_layout = QVBoxLayout(settings_group)
        settings_layout.setSpacing(10)

        # Row 1: Language
        lang_row = QHBoxLayout()
        lang_label = QLabel("Language:")
        lang_label.setFixedWidth(90)
        self.lang_combo = QComboBox()
        for display, code in LANGUAGES:
            self.lang_combo.addItem(f"{display}  ({code})", code)
        lang_row.addWidget(lang_label)
        lang_row.addWidget(self.lang_combo, 1)
        settings_layout.addLayout(lang_row)

        # Row 2: Start page
        start_row = QHBoxLayout()
        start_label = QLabel("Start page:")
        start_label.setFixedWidth(90)
        self.start_input = QLineEdit()
        self.start_input.setPlaceholderText("e.g.  France")
        start_row.addWidget(start_label)
        start_row.addWidget(self.start_input, 1)
        settings_layout.addLayout(start_row)

        # Row 3: End page
        end_row = QHBoxLayout()
        end_label = QLabel("End page:")
        end_label.setFixedWidth(90)
        self.end_input = QLineEdit()
        self.end_input.setPlaceholderText("e.g.  Europe")
        end_row.addWidget(end_label)
        end_row.addWidget(self.end_input, 1)
        settings_layout.addLayout(end_row)

        # Row 4: Search Method
        method_row = QHBoxLayout()
        method_label = QLabel("Method:")
        method_label.setFixedWidth(90)
        self.method_combo = QComboBox()
        self.method_combo.addItem("Likely Path (Semantic AI)", "semantic")
        self.method_combo.addItem("Fast Guess (Word Match)", "lexical")
        self.method_combo.addItem("Brute Force (BFS)", "bruteforce")
        method_row.addWidget(method_label)
        method_row.addWidget(self.method_combo, 1)
        settings_layout.addLayout(method_row)

        # Row 5: Buttons
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self.start_btn = QPushButton("⛏  Start Mining")
        self.start_btn.setObjectName("startBtn")
        self.start_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.start_btn.clicked.connect(self._on_start)

        self.cancel_btn = QPushButton("✕  Cancel")
        self.cancel_btn.setObjectName("cancelBtn")
        self.cancel_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._on_cancel)

        btn_row.addWidget(self.start_btn)
        btn_row.addWidget(self.cancel_btn)
        btn_row.addStretch()
        settings_layout.addLayout(btn_row)

        root.addWidget(settings_group)

        # ── Results / Log splitter ─────────────────
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.setHandleWidth(4)

        # Result panel
        self.result_group = QGroupBox("Result")
        result_layout = QVBoxLayout(self.result_group)
        self.result_title = QLabel("")
        self.result_title.setObjectName("resultTitle")
        self.result_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        result_layout.addWidget(self.result_title)

        stats_row = QHBoxLayout()
        stats_row.setSpacing(20)

        # Clicks box
        clicks_box = QVBoxLayout()
        clicks_box.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.clicks_label = QLabel("–")
        self.clicks_label.setObjectName("clicksLabel")
        self.clicks_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        clicks_caption = QLabel("clicks")
        clicks_caption.setAlignment(Qt.AlignmentFlag.AlignCenter)
        clicks_box.addWidget(self.clicks_label)
        clicks_box.addWidget(clicks_caption)

        # Time box
        time_box = QVBoxLayout()
        time_box.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.time_label = QLabel("–")
        self.time_label.setObjectName("timeLabel")
        self.time_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.time_label.setStyleSheet(f"font-size: 36px; font-weight: 800; color: {ACCENT};")
        time_caption = QLabel("seconds")
        time_caption.setAlignment(Qt.AlignmentFlag.AlignCenter)
        time_box.addWidget(self.time_label)
        time_box.addWidget(time_caption)

        # Pages box
        pages_box = QVBoxLayout()
        pages_box.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.pages_label = QLabel("–")
        self.pages_label.setObjectName("pagesLabel")
        self.pages_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.pages_label.setStyleSheet(f"font-size: 36px; font-weight: 800; color: {SUCCESS};")
        pages_caption = QLabel("pages")
        pages_caption.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pages_box.addWidget(self.pages_label)
        pages_box.addWidget(pages_caption)

        stats_row.addStretch()
        stats_row.addLayout(clicks_box)
        stats_row.addLayout(time_box)
        stats_row.addLayout(pages_box)
        stats_row.addStretch()
        result_layout.addLayout(stats_row)

        # Path display
        self.path_display = QTextBrowser()
        self.path_display.setReadOnly(True)
        self.path_display.setOpenExternalLinks(True)
        self.path_display.setMaximumHeight(150)
        self.path_display.setPlaceholderText("Path will appear here after a successful search…")
        result_layout.addWidget(self.path_display)

        splitter.addWidget(self.result_group)

        # Log panel
        log_group = QGroupBox("Live Log")
        log_layout = QVBoxLayout(log_group)
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setPlaceholderText("Search activity log…")
        log_layout.addWidget(self.log_area)
        splitter.addWidget(log_group)

        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)
        root.addWidget(splitter, 1)

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

        self.worker = SearchWorker(start, end, lang, mode)
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
