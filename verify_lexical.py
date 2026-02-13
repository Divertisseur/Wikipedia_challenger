
import difflib
import re
from nltk.stem import SnowballStemmer

_WORD_RE = re.compile(r"[a-zA-Z0-9\u00C0-\u024F]+", re.UNICODE)
stemmer = SnowballStemmer("french")

def _get_stemmed_words(text: str):
    words = _WORD_RE.findall(text.lower())
    return {stemmer.stem(w) for w in words}

def calculate_lexical_score(title, target_title):
    target_words_stemmed = _get_stemmed_words(target_title)
    current_words = _get_stemmed_words(title)
    
    if target_words_stemmed and current_words:
        intersection = current_words & target_words_stemmed
        reach = len(intersection) / len(target_words_stemmed)
        precision = len(intersection) / len(current_words)
        char_sim = difflib.SequenceMatcher(None, title.lower(), target_title.lower()).ratio()
        
        score = (0.4 * reach) + (0.4 * precision) + (0.2 * char_sim)
        return {
            "score": score,
            "reach": reach,
            "precision": precision,
            "char_sim": char_sim
        }
    return {"score": 0.0, "reach": 0, "precision": 0, "char_sim": 0}

target = "France"
titles = ["France", "France en 1988", "Île-de-France", "Français"]

with open("verify_lexical_results.txt", "w", encoding="utf-8") as f:
    f.write(f"Target: {target}\n")
    for t in titles:
        res = calculate_lexical_score(t, target)
        f.write(f"Title: {t:20} Score: {res['score']:.3f} (R: {res['reach']:.2f}, P: {res['precision']:.2f}, C: {res['char_sim']:.2f})\n")
