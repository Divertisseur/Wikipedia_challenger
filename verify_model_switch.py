
import sys
from wikipedia_bot import WikipediaChallenger

try:
    print("Initializing bot...")
    bot = WikipediaChallenger(lang="en")

    # Test Case 1: use_metadata=False (Default)
    # Should use 'all-MiniLM-L6-v2'
    print("\n--- Test Case 1: use_metadata=False ---")
    bot._init_similarity_engine("Target", "semantic", use_metadata=False)
    
    current_model = bot.similarity_engine.model
    # SentenceTransformer objects might not have a simple 'name' attribute publicly accessible in all versions,
    # but we can check if it loaded. We can also check bot.current_model_name which we added.
    print(f"Current model name in bot: {bot.current_model_name}")
    
    if bot.current_model_name == 'all-MiniLM-L6-v2':
        print("PASS: Correct model for no metadata.")
    else:
        print(f"FAIL: Expected 'all-MiniLM-L6-v2', got '{bot.current_model_name}'")

    # Test Case 2: use_metadata=True
    # Should use 'all-mpnet-base-v2'
    print("\n--- Test Case 2: use_metadata=True ---")
    bot._init_similarity_engine("Target", "semantic", use_metadata=True)
    
    print(f"Current model name in bot: {bot.current_model_name}")
    
    if bot.current_model_name == 'all-mpnet-base-v2':
        print("PASS: Correct model for metadata enabled.")
    else:
        print(f"FAIL: Expected 'all-mpnet-base-v2', got '{bot.current_model_name}'")

    # Test Case 3: Switch back to False
    print("\n--- Test Case 3: Switch back to use_metadata=False ---")
    bot._init_similarity_engine("Target", "semantic", use_metadata=False)
    print(f"Current model name in bot: {bot.current_model_name}")
    
    if bot.current_model_name == 'all-MiniLM-L6-v2':
        print("PASS: Correctly switched back.")
    else:
        print(f"FAIL: Expected 'all-MiniLM-L6-v2', got '{bot.current_model_name}'")

except Exception as e:
    print(f"\nERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
