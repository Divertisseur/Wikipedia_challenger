
import json
from wiki_api import WikipediaAPI

def inspect_page(title, lang="en"):
    api = WikipediaAPI(lang=lang)
    norm_title = api.validate_page(title)
    if not norm_title:
        return {"error": "Page not found"}
    metadata = api.get_metadata_batch([norm_title])
    desc = metadata.get(norm_title, "")
    model_input = f"{norm_title}: {desc}" if desc else norm_title
    return {
        "lang": lang,
        "normalized": norm_title,
        "description": desc,
        "model_input": model_input
    }

if __name__ == "__main__":
    results = [
        inspect_page("Tadej Pogacar", lang="en"),
        inspect_page("Tadej Pogacar", lang="fr")
    ]
    with open("pogacar_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
