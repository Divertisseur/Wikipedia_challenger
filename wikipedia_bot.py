"""
Wikipedia Challenger – Optimised pathfinder
───────────────────────────────────────────
• Concurrent link fetching  (ThreadPoolExecutor, 8 workers)
• Advanced Similarity Search:
    - Semantic Similarity (sentence-transformers)
    - Stemmed Lexical Overlap (nltk)
    - Hub/Popularity Weighting (Title heuristics)
"""

import dataclasses
import difflib
import heapq
import re
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Callable, Set, Any, Tuple

import numpy as np
import nltk
from nltk.stem import SnowballStemmer
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
from wiki_api import WikipediaAPI

# ── Tunables ────────────────────────────────────────────────────────
MAX_WORKERS = 8        # concurrent API requests per BFS level
SEMANTIC_WEIGHT = 0.50 # weight for semantic embedding similarity
LEXICAL_WEIGHT = 0.30  # weight for stemmed lexical overlap
HUB_WEIGHT = 0.20      # weight for hub/popularity bonus

_WORD_RE = re.compile(r"[a-zA-Z0-9\u00C0-\u024F]+", re.UNICODE)


# ── Advanced Similarity Engine ──────────────────────────────────────
# ── Advanced Similarity Engine ──────────────────────────────────────
class AdvancedSimilarityEngine:
    def __init__(self, lang_code: str = "en", use_semantic: bool = True):
        self.use_semantic = use_semantic
        
        # Map Wikipedia code to NLTK language name
        lang_map = {
            "en": "english", "fr": "french", "de": "german",
            "es": "spanish", "it": "italian", "pt": "portuguese",
            "nl": "dutch", "ru": "russian"
        }
        nltk_lang = lang_map.get(lang_code, "english")

        try:
            self.stemmer = SnowballStemmer(nltk_lang)
        except Exception:
            self.stemmer = SnowballStemmer("english")

        # Load semantic model only if requested
        self.model = None
        if self.use_semantic:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        self.target_embedding = None
        self.target_title = ""
        self.target_words_stemmed = set()

    def set_target(self, target_title: str):
        self.target_title = target_title
        self.target_words_stemmed = self._get_stemmed_words(target_title)
        if self.use_semantic and self.model:
            self.target_embedding = self.model.encode(target_title, convert_to_tensor=True)

    def _get_stemmed_words(self, text: str) -> Set[str]:
        words = _WORD_RE.findall(text.lower())
        return {self.stemmer.stem(w) for w in words}

    def score_batch(self, titles: List[str], descriptions: Optional[Dict[str, str]] = None) -> np.ndarray:
        """Calculate scores for a batch of titles based on configuration."""
        if not titles:
            return np.array([])
        
        count = len(titles)
        descriptions = descriptions or {}
        
        # 1. Semantic Scores
        if self.use_semantic and self.model and self.target_embedding is not None:
             # Enhance titles with descriptions for richer context
             text_to_encode = []
             for t in titles:
                 desc = descriptions.get(t, "")
                 if desc:
                     text_to_encode.append(f"{t}: {desc}")
                 else:
                     text_to_encode.append(t)
                     
             embeddings = self.model.encode(text_to_encode, convert_to_tensor=True)
             semantic_scores = util.cos_sim(embeddings, self.target_embedding).flatten().cpu().numpy()
        else:
             semantic_scores = np.zeros(count)

        combined_scores = []
        for i, title in enumerate(titles):
            # 2. Lexical Score (Stemmed Overlap)
            current_words = self._get_stemmed_words(title)
            if self.target_words_stemmed:
                lexical_score = len(current_words & self.target_words_stemmed) / len(self.target_words_stemmed)
            else:
                lexical_score = 0.0

            # 3. Hub / Popularity Weighting
            hub_bonus = 0.0
            title_len = len(title)
            if title_len < 10: hub_bonus += 0.5
            elif title_len < 20: hub_bonus += 0.2
            
            hubs = {"history", "science", "mathematics", "geography", "politics", "culture"}
            if any(h in title.lower() for h in hubs):
                hub_bonus += 0.3

            # Combine
            # If semantic is off, we scale up lexical/hub to act as the primary score
            if self.use_semantic:
                score = (
                    SEMANTIC_WEIGHT * semantic_scores[i] +
                    LEXICAL_WEIGHT * lexical_score +
                    HUB_WEIGHT * min(hub_bonus, 1.0)
                )
            else:
                # "Lexical only" mode re-weights: 70% lexical, 30% hub
                score = 0.7 * lexical_score + 0.3 * min(hub_bonus, 1.0)
                
            combined_scores.append(score)

        return np.array(combined_scores)


@dataclasses.dataclass(order=True)
class PrioritizedItem:
    priority: Tuple[float, int]
    item: Any = dataclasses.field(compare=False)


class WikipediaChallenger:
    def __init__(
        self,
        lang: str = "en",
        log_callback: Optional[Callable[[str], None]] = None,
    ):
        self.api = WikipediaAPI(lang=lang)
        self.log_callback = log_callback
        self.lang = lang
        self._cancelled = False
        self.similarity_engine = None

    def _log(self, message: str):
        if self.log_callback:
            self.log_callback(message)
        else:
            print(message)

    def cancel(self):
        self._cancelled = True

    def _init_similarity_engine(self, target_title: str, mode: str):
        # Re-init if mode changed or engine missing
        use_semantic = (mode == "semantic")
        
        # If we need semantic but current engine is lexical-only, force re-init
        if self.similarity_engine and self.similarity_engine.use_semantic != use_semantic:
            self.similarity_engine = None
            
        if not self.similarity_engine:
            if use_semantic:
                self._log("Loading semantic AI model…")
            self.similarity_engine = AdvancedSimilarityEngine(self.lang, use_semantic=use_semantic)
            
        self.similarity_engine.set_target(target_title)

    def find_shortest_path(
        self, start_title: str, end_title: str, mode: str = "semantic", use_metadata: bool = False
    ) -> Optional[Dict]:
        self._cancelled = False
        start_time = time.time()

        self._log("Validating pages…")
        start_norm = self.api.validate_page(start_title)
        end_norm = self.api.validate_page(end_title)

        if not start_norm:
            self._log(f"Error: Start page '{start_title}' not found.")
            return None
        if not end_norm:
            self._log(f"Error: End page '{end_title}' not found.")
            return None

        self._log(f"Searching from '{start_norm}' to '{end_norm}'…")
        self._log(f"Mode: {mode.upper()} · {MAX_WORKERS} threads")
        
        # Initialize engine unless bruteforce
        if mode != "bruteforce":
            self._init_similarity_engine(end_norm, mode)

        if start_norm == end_norm:
            return {"path": [start_norm], "clicks": 0,
                    "time": time.time() - start_time,
                    "searched": 0}

        counter = 0
        heap: list = []
        # For bruteforce, priority is 0 (FIFO behavior via counter)
        # For others, priority is -score
        init_prio = 0.0 if mode == "bruteforce" else -1.0
        
        heapq.heappush(
            heap,
            PrioritizedItem((init_prio, counter), (start_norm, [start_norm]))
        )
        visited: Set[str] = {start_norm}

        use_tqdm = self.log_callback is None
        pbar = tqdm(desc="Exploring pages", unit="page") if use_tqdm else None
        pages_explored = 0

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            while heap:
                if self._cancelled:
                    self._log("Search cancelled by user.")
                    if use_tqdm: pbar.close()
                    return None

                batch: list = []
                while heap and len(batch) < MAX_WORKERS:
                    wrapper = heapq.heappop(heap)
                    batch.append(wrapper.item)

                future_map = {}
                for page, path in batch:
                    future = pool.submit(self.api.get_links, page)
                    future_map[future] = (page, path)

                for future in as_completed(future_map):
                    if self._cancelled:
                        self._log("Search cancelled by user.")
                        if use_tqdm: pbar.close()
                        return None

                    page, path = future_map[future]
                    pages_explored += 1
                    if use_tqdm: pbar.update(1)
                    else: self._log(f"[{pages_explored}] Exploring: {page}")

                    try:
                        links = future.result()
                    except Exception as e:
                        self._log(f"Skipping {page}: {e}")
                        continue

                    valid_links = []
                    for link in links:
                        if link == end_norm:
                            total_time = time.time() - start_time
                            if use_tqdm: pbar.close()
                            return {
                                "path": path + [end_norm],
                                "clicks": len(path),
                                "time": total_time,
                                "searched": pages_explored,
                            }
                        if link not in visited:
                            visited.add(link)
                            valid_links.append(link)

                    # Strategy:
                    # 1. Bruteforce: Add all with priority=0 (FIFO by counter)
                    # 2. Similarity: Calculate scores, add with priority=-score
                    
                    if mode == "bruteforce":
                        for link in valid_links:
                            counter += 1
                            heapq.heappush(
                                heap,
                                PrioritizedItem((0.0, counter), (link, path + [link]))
                            )
                    elif valid_links:
                        # Batch fetch descriptions only if requested
                        descriptions = {}
                        if use_metadata:
                            for i in range(0, len(valid_links), 50):
                                batch_links = valid_links[i:i+50]
                                descriptions.update(self.api.get_metadata_batch(batch_links))
                            
                        scores = self.similarity_engine.score_batch(valid_links, descriptions)
                        for link, score in zip(valid_links, scores):
                            counter += 1
                            heapq.heappush(
                                heap,
                                PrioritizedItem(
                                    (-score, counter),
                                    (link, path + [link])
                                )
                            )

        if use_tqdm: pbar.close()
        return None

    def print_report(self, result: Optional[Dict]):
        if not result:
            self._log("\nNo path found between the pages.")
            return

        self._log("\n" + "=" * 40)
        self._log("WIKIPEDIA CHALLENGER REPORT")
        self._log("=" * 40)
        self._log("Status: Target Found!")
        self._log(f"Start Page: {result['path'][0]}")
        self._log(f"End Page: {result['path'][-1]}")
        self._log(f"Total Clicks: {result['clicks']}")
        self._log(f"Time Taken: {result['time']:.2f} seconds")
        self._log(f"Pages Searched: {result.get('searched', 0)}")
        self._log("\nPath Traversed:")
        for i, page in enumerate(result["path"]):
            prefix = "  " * i + ("└─ " if i > 0 else "")
            url_safe = page.replace(" ", "_")
            url = f"https://{self.lang}.wikipedia.org/wiki/{url_safe}"
            self._log(f"{prefix}{page} ({url})")
        self._log("=" * 40)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Wikipedia Challenger Bot")
    parser.add_argument("start", help="Starting Wikipedia page title", nargs="?")
    parser.add_argument("end", help="Ending Wikipedia page title", nargs="?")
    parser.add_argument("--lang", help="Wikipedia language code (default: en)", default="en")
    parser.add_argument("--intelligent", help="Enable intelligent context (slower)", action="store_true")

    args = parser.parse_args()
    start, end, lang = args.start, args.end, args.lang

    if not start or not end:
        start = input("Enter starting page: ")
        end = input("Enter ending page: ")
        lang_input = input("Enter language code (default: en): ")
        if lang_input: lang = lang_input

    bot = WikipediaChallenger(lang=lang)
    result = bot.find_shortest_path(start, end, use_metadata=args.intelligent)
    bot.print_report(result)



if __name__ == "__main__":
    main()
