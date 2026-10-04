    import json
    import pandas as pd
    import streamlit as st
    import core

    st.set_page_config(page_title="KahwaConnect", page_icon="☕", layout="wide")
    st.write("")  # ensures first render paints immediately

    @st.cache_resource
    def get_model():
        return core.train()

    try:
        model = get_model()
    except Exception as e:  # show a visible error instead of a blank page
        st.error(f"Model failed to load: {e}")
        st.stop()

    SAMPLES = [
        "Hello! How much does the farm tour cost for 2 people?",
        "Hallo, wir möchten am Samstag eine Führung buchen. Wir sind 4 Personen.",
        "Wie kommen wir von der Stadt zu eurer Farm?",
        "What is included in the tour and how long does it take?",
        "I have a nut allergy, is the tasting safe for me?",
        "Can you recommend a hotel in the city?",
    ]
    SAMPLE_REVIEWS = """The coffee tasting was amazing and Noor was so friendly and welcoming.
    Beautiful views of the highlands. But the road was muddy and hard to find, we got lost twice.
    Loved the fresh coffee and the tea. Wish there was a translation card, we could not understand the guide.
    Great value for the price. The walk felt a bit rushed though.
    Noor's family made us feel at home. No shade or toilet near the coffee plot, that could be better.
    Delicious lunch from the farm. The directions were unclear and the signal was poor.
    Best coffee I have ever tasted. Beautiful scenery, very kind host.
    The road to the farm is difficult but the view is worth it."""

    if "outbox" not in st.session_state:
        st.session_state.outbox = []

    # ---------------- Sidebar: Noor's settings ----------------
    with st.sidebar:
        st.header("☕ Noor's settings")
        st.caption("Noor fills these once. Replies only use these facts, so the tool cannot invent a price or place.")
        cfg = {
            "price": st.text_input("Price per person", "PKR 2,000 (example)"),
            "duration": st.text_input("Tour duration", "2 hours"),
            "includes": st.text_input("What is included", "coffee plot walk, tasting, farm tea"),
            "meet_point": st.text_input("Meeting point", "the Ondera Coffee Cooperative gate"),
            "phone": st.text_input("Phone", "+92 000 0000000"),
        }
        st.divider()
        st.caption("Mode: **offline core**. The classifier is a ~100 KB TF-IDF + logistic regression model that runs on-device. No internet is needed for any feature on this page.")

    st.title("☕ KahwaConnect")
    st.caption("Small AI for development · Tourism track · Human-in-the-loop · Not legal or financial advice")

    tab1, tab2, tab3 = st.tabs(["📨 Visitor inbox", "⭐ Review insights", "📊 Evidence & limits"])

    # ---------------- Tab 1 ----------------
    with tab1:
        c1, c2 = st.columns([1, 1])
        with c1:
            pick = st.selectbox("Load a sample message (or type your own below)", ["(custom)"] + SAMPLES)
            msg = st.text_area("Visitor message", value="" if pick == "(custom)" else pick, height=120,
                            placeholder="Paste a message in English, German or Urdu...")
        if msg.strip():
            lang = core.detect_lang(msg)
            intent, conf, order = core.classify(model, msg)
            ask, why = core.needs_person(msg, intent, conf, lang)
            ents = core.entities(msg)
            langname = {"en": "English", "de": "German", "ur": "Urdu", "unknown": "Unknown"}[lang]
            with c1:
                st.markdown(f"**Language:** {langname} &nbsp;|&nbsp; **Topic:** {core.INTENT_LABEL[intent]} &nbsp;|&nbsp; **Confidence:** {conf:.0%}")
                st.progress(min(conf, 1.0))
                if ents:
                    st.markdown("**Details found:** " + ", ".join(f"{k}: {v}" for k, v in ents.items()))
            with c2:
                st.markdown("**اردو میں خلاصہ (Urdu summary for Noor)**")
                gist = core.URDU_GIST["other"] if ask else core.URDU_GIST[intent]
                st.markdown(f"<div dir='rtl' style='font-size:1.4rem;line-height:2;background:#f3e6d3;padding:12px;border-radius:8px'>{gist}</div>",
                            unsafe_allow_html=True)
                st.caption("Summary comes from the topic label, not a free translation, so it cannot hallucinate. Always read the original message.")
            st.divider()
            if ask:
                st.warning(f"🙋 **Not sure — ask a person.** {why}")
                st.caption("No draft was written. Noor reads the message and replies herself.")
            else:
                draft, ur = core.draft_reply(intent, lang, cfg)
                st.subheader("Draft reply (edit if needed)")
                reply = st.text_area(f"Reply in visitor's language ({langname if lang != 'unknown' else 'English'})", draft, height=110)
                st.markdown("**Urdu meaning of this reply, so Noor can check what she is sending:**")
                st.markdown(f"<div dir='rtl' style='font-size:1.2rem;line-height:2'>{ur}</div>", unsafe_allow_html=True)
                st.caption("Nothing is sent automatically. Tapping Approve only saves it to the outbox.")
                if st.button("✅ Approve — queue to send when signal is available", type="primary"):
                    st.session_state.outbox.append({"visitor_message": msg, "reply": reply, "topic": intent,
                                                    "status": "queued (store-and-forward)"})
                    st.success("Saved to outbox.")
        st.divider()
        st.subheader(f"📤 Outbox ({len(st.session_state.outbox)})")
        if st.session_state.outbox:
            st.dataframe(pd.DataFrame(st.session_state.outbox))
            st.download_button("Download outbox (JSON)", json.dumps(st.session_state.outbox, ensure_ascii=False, indent=2),
                            "outbox.json", "application/json")
            if st.button("Clear outbox"):
                st.session_state.outbox = []
                st.rerun()
        else:
            st.caption("Nothing queued yet.")

    # ---------------- Tab 2 ----------------
    with tab2:
        st.write("Paste visitor reviews, one per line. The tool counts what visitors keep praising and what they struggle with, and shows the exact sentence as evidence.")
        txt = st.text_area("Reviews", SAMPLE_REVIEWS, height=220)
        reviews = [r.strip() for r in txt.splitlines() if r.strip()]
        st.caption(f"{len(reviews)} reviews")
        if len(reviews) < 3:
            st.warning("🙋 Not sure — too few reviews for a pattern. Collect at least 5.")
        else:
            loved, issues, idea = core.analyse_reviews(reviews)
            a, b = st.columns(2)
            with a:
                st.subheader("💚 Visitors love")
                for t, ev in loved:
                    st.markdown(f"**{t}** — {len(ev)} mention(s)")
                    for e in ev[:2]: st.caption(f"“{e}”")
            with b:
                st.subheader("🟠 Visitors struggle with")
                for t, ev in issues:
                    st.markdown(f"**{t}** — {len(ev)} mention(s)")
                    for e in ev[:2]: st.caption(f"“{e}”")
            if idea:
                st.success(f"💡 **Idea to test next:** {idea[1]}  \n*(based on: {idea[0]})*")
                st.caption("This is a suggestion from a fixed idea list, not a prediction. Noor decides.")
            if len(reviews) < 8:
                st.info("Small sample: treat these as hints, not conclusions.")

    # ---------------- Tab 3 ----------------
    with tab3:
        acc, rows = core.evaluate(model)
        asked = sum(r["ask_a_person"] for r in rows)
        wrong_auto = sum((not r["correct"]) and (not r["ask_a_person"]) for r in rows)
        st.subheader("Held-out test (20 messages the model never saw)")
        m1, m2, m3 = st.columns(3)
        m1.metric("Topic accuracy", f"{acc:.0%}")
        m2.metric("Sent to a person", f"{asked}/20")
        m3.metric("Wrong AND auto-answered", f"{wrong_auto}/20")
        st.dataframe(pd.DataFrame(rows))
        st.markdown("""
    **Why AI, not just SMS or a search?** Visitors write in English, German and Roman Urdu, with typos and mixed words. A keyword list breaks on this; a small character-level model generalises across spellings and languages, and gives a confidence that powers the "ask a person" fail-safe.

    **Guardrails.** (1) Fixed list of answers: only 4 topics, replies come from templates filled with Noor's own facts. (2) Low confidence, sensitive words (allergy, refund, injury) or unknown language → *Not sure — ask a person*. (3) Nothing is sent without Noor's approval. (4) No visitor data leaves the device; the outbox is stored only in this session and exported by Noor.

    **Data.** Training and test messages are **synthetic**, written by the team (≈16 per topic, English / German / Roman Urdu). Sector context: Annex C of the World Bank concept note; MASSIVE (Amazon) and Wikivoyage are the intended next sources for scale-up.

    **What the data does NOT cover.** Real visitor messages; Punjabi and other local languages; voice input; long or multi-question messages; slang. Accuracy on real traffic will be lower than shown here. Urdu replies were written by the team and should be reviewed by a native speaker before real use.

    **Offline.** Everything on this page runs without internet. Approved replies wait in the outbox until Noor has signal (store-and-forward).

    **Localizing AI means** building around the language, device and daily routine of the person using it: small models, her facts, her approval, and a tool that admits when it does not know.
    """)
