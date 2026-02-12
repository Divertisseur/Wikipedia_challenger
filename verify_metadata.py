
from wiki_api import WikipediaAPI
from wikipedia_bot import AdvancedSimilarityEngine
import numpy as np

def verify_metadata():
    with open("verify_log.txt", "w", encoding="utf-8") as f:
        # 1. Test API Metadata Batch Fetching
        api = WikipediaAPI()
        titles = ["Barack Obama", "Paris", "Python (programming language)"]
        metadata = api.get_metadata_batch(titles)
        f.write(f"Metadata Fetch: {'SUCCESS' if len(metadata) > 0 else 'FAILED'}\n")
        for t, d in metadata.items():
            f.write(f" - {t}: {d}\n")

        # 2. Test Scoring with Descriptions
        engine = AdvancedSimilarityEngine(use_semantic=True)
        target = "United States"
        engine.set_target(target)
        
        candidates = ["Barack Obama", "Eiffel Tower"]
        descriptions = {
            "Barack Obama": "44th president of the United States",
            "Eiffel Tower": "Tower in Paris, France"
        }
        
        scores_no = engine.score_batch(candidates, descriptions=None)
        scores_yes = engine.score_batch(candidates, descriptions=descriptions)
        
        f.write(f"\nTarget: {target}\n")
        for i, t in enumerate(candidates):
            f.write(f"{t}:\n")
            f.write(f"  No desc: {scores_no[i]:.4f}\n")
            f.write(f"  With desc: {scores_yes[i]:.4f}\n")
            f.write(f"  Diff: {scores_yes[i] - scores_no[i]:.4f}\n")

if __name__ == "__main__":
    verify_metadata()
