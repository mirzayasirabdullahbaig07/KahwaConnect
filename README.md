# ☕ KahwaConnect — Small AI for Development (Tourism track)

Offline-first Streamlit tool that helps Noor, a coffee farmer running informal farm tours, to:
1. **Sort visitor messages** (English / German / Roman Urdu / Urdu) into price, directions, booking, tour info — or "Not sure — ask a person".
2. **Draft replies** in the visitor's language from templates filled with Noor's own facts, with an Urdu back-meaning so she can check it. Noor approves; nothing is sent automatically.
3. **Summarise reviews** into what visitors love, what they struggle with, and one idea to test — with the exact sentences as evidence.

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```
## Deploy
Push to GitHub → share.streamlit.io → New app → select repo, main file `app.py`.

## Architecture
`core.py`: TF-IDF (char n-grams) + logistic regression, trained at start-up on a synthetic labelled set (~80 messages); language detection by word lists; safety-word escalation; rule-based review theme/sentiment analysis. `app.py`: Streamlit UI. No external API calls.

## Data & limits
Synthetic messages written by the team (en/de/Roman Urdu). Not covered: real visitor traffic, Punjabi, voice, long messages. Urdu templates need native-speaker review. Not a legal or financial advisor.
