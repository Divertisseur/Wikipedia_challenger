import traceback
import sys
from wikipedia_bot import WikipediaChallenger

try:
    bot = WikipediaChallenger(lang="en")
    result = bot.find_shortest_path("France", "Europe")
    bot.print_report(result)
except Exception:
    traceback.print_exc()
