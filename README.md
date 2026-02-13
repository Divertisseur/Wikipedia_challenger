# 🌐 Wikipedia Challenger

Wikipedia Challenger is an optimized pathfinder bot designed to find the shortest path between any two Wikipedia pages using various search strategies, including semantic AI.

## 🚀 Features

- **Advanced Search Modes**:
  - **Semantic AI (Likely Path)**: Uses `sentence-transformers` (all-MiniLM-L6-v2) to understand the meaning of page titles and prioritize links that are semantically close to the target.
  - **Lexical Match (Fast Guess)**: Uses stemmed word overlap and hub weighting to navigate via common topics.
  - **Brute Force (BFS)**: A traditional Breadth-First Search for finding the guaranteed shortest path (though slower).
- **Fast & Efficient**: Utilizes `ThreadPoolExecutor` for concurrent API requests.
- **Detailed Reporting**:
  - **Total Clicks**: Shows how many clicks it took to reach the destination.
  - **Time Taken**: Real-time performance tracking.
  - **Pages Searched**: New counter showing the total number of pages explored.
  - **Visual Path**: A tree view of the path taken, including clickable direct links in the GUI.
- **Modern GUI**: A beautiful dark-themed PyQt6 desktop interface with optional **Intelligent Context** mode.

## 🛠️ Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/MaelK/Wikipedia_challenger.git
    cd Wikipedia_challenger
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## 📖 Usage

### Desktop GUI (Recommended)
Launch the modern interface with:
```bash
python gui.py
```

### CLI Bot
Run the search directly from the terminal:
```bash
python wikipedia_bot.py "France" "Europe" --lang fr --intelligent
```
*Note: Use `--intelligent` to improve celebrity searches (might be slower).*

### Debug & Testing
Run a quick test script:
```bash
python debug_bot.py
```

## 🧪 Requirements

- Python 3.8+
- `PyQt6` (for GUI)
- `sentence-transformers`, `torch`, `numpy` (for Semantic AI)
- `nltk` (for lexical scoring)
- `requests`, `tqdm`

## 📝 Recent Improvements
- Added **Pages Searched** counter to track search depth.
- Implemented **Clickable Links** in the GUI path display for instant navigation.
- Switched to `QTextBrowser` for enhanced link interaction.
- Synchronized CLI reports with new metrics and URLs.
