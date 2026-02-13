
from wiki_api import WikipediaAPI

def inspect_page(title, lang="en"):
    api = WikipediaAPI(lang=lang)
    print(f"\n--- Inspection of '{title}' (lang={lang}) ---")
    
    # 1. Validation & Normalization
    norm_title = api.validate_page(title)
    print(f"Normalized Title: {norm_title}")
    
    if not norm_title:
        print("Page not found!")
        return

    # 2. Description
    metadata = api.get_metadata_batch([norm_title])
    desc = metadata.get(norm_title, "")
    print(f"Description (Metadata): {repr(desc)}")
    
    # 3. Final string sent to model
    if desc:
        model_input = f"{norm_title}: {desc}"
    else:
        model_input = norm_title
    print(f"Exact string sent to model: {repr(model_input)}")

if __name__ == "__main__":
    inspect_page("Tadej Pogacar", lang="en")
    inspect_page("Tadej Pogacar", lang="fr")
