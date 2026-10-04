"""Core logic for KahwaConnect. Runs fully offline (no API calls)."""
import re
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

CONF_THRESHOLD = 0.35  # below this -> "Not sure - ask a person"

# ---- Training data: SYNTHETIC, hand-written by the team (en / de / Roman Urdu) ----
TRAIN = {
 "price": [
  "How much does the farm tour cost?", "What is the price per person?", "How much for 2 people?",
  "Is there a discount for children?", "What do you charge for a coffee tasting?", "Do you have group prices?",
  "Is the tour expensive?", "What are your rates?", "Wie viel kostet die Führung?", "Was kostet es pro Person?",
  "Gibt es einen Preis für Kinder?", "Wie teuer ist die Kaffeeverkostung?", "kitne paise lagenge?",
  "tour ki qeemat kya hai?", "ek bande ka kitna kharcha hai?", "Combien coûte la visite?"],
 "directions": [
  "How do I get to your farm?", "Where do we meet?", "Where exactly are you located?",
  "Can you send the location?", "Is there a bus from the town to the farm?", "Which road should we take?",
  "We are lost, where are you?", "How far is the farm from the district town?", "Wie kommen wir zu eurer Farm?",
  "Wo ist der Treffpunkt?", "Wo genau liegt der Hof?", "Gibt es einen Bus dorthin?", "farm tak kaise pohanchein?",
  "aap ki jagah kahan hai?", "rasta kaun sa hai?", "Où est la ferme?"],
 "booking": [
  "I would like to book a tour for Saturday.", "Can we reserve for 4 people next week?", "Are you available on 15 October?",
  "We want to join your tour tomorrow.", "Do you have free spots this weekend?", "Please book us for two adults.",
  "Is Sunday morning possible?", "Can I make a reservation?", "Wir möchten eine Führung am Samstag buchen.",
  "Habt ihr nächste Woche noch Plätze frei?", "Können wir für 3 Personen reservieren?", "Ist morgen noch etwas frei?",
  "mujhe tour book karni hai", "kya hum kal aa sakte hain?", "hafte ko 4 log aana chahte hain", "Nous voulons réserver pour dimanche."],
 "tour_info": [
  "What is included in the tour?", "How long does the tour take?", "Do I need to wear special shoes?",
  "Will we taste the coffee?", "Is the tour suitable for children?", "What will we see on the farm?",
  "Do you offer lunch?", "What should I bring?", "Was ist in der Führung enthalten?", "Wie lange dauert die Tour?",
  "Gibt es etwas zu essen?", "Ist es für Kinder geeignet?", "tour mein kya kya shamil hai?",
  "tour kitni der ki hai?", "kya khana milega?", "Qu'est-ce qui est inclus?"],
 "other": [
  "Hello", "Thank you so much!", "Great, see you soon", "Hi Noor", "Guten Tag", "Danke schön",
  "ok", "Good morning", "Can you recommend a hotel in the city?", "What is the weather forecast?",
  "Do you sell coffee beans to export?", "I am a journalist, can I call you?", "Hallo zusammen",
  "shukriya", "assalam o alaikum", "Is the internet good there?"],
}

# ---- Held-out test set (different wording) for the evidence slide ----
TEST = [
 ("What's the fee for one adult?", "price"), ("Wie viel müssen wir zahlen?", "price"), ("kitna paisa dena hoga", "price"),
 ("Is it cheaper for a group of six?", "price"), ("Where should we park?", "directions"),
 ("Wo finden wir euch?", "directions"), ("Can you share a map pin?", "directions"),
 ("hum aap ke ghar kaise aayen?", "directions"), ("Can we come on Friday afternoon?", "booking"),
 ("Wir würden gerne am Sonntag kommen, geht das?", "booking"), ("I want to reserve a spot for my family", "booking"),
 ("hamein kal ka slot chahiye", "booking"), ("How many hours is the walk?", "tour_info"),
 ("Gibt es eine Kaffeeprobe?", "tour_info"), ("Do we get to try the coffee?", "tour_info"),
 ("tour mein chai bhi milegi?", "tour_info"), ("Thanks!", "other"), ("Vielen Dank und bis bald", "other"),
 ("Can you tell me a good restaurant nearby?", "other"), ("walaikum assalam", "other"),
]

SAFETY_WORDS = ["allerg", "sick", "injur", "pregnan", "wheelchair", "medical", "doctor", "refund", "complain",
                "krank", "unfall", "schwanger", "rollstuhl", "arzt", "beschwerde", "emergency", "notfall",
                "bimar", "hadsa", "shikayat"]

def train():
    X, y = [], []
    for k, v in TRAIN.items():
        X += v; y += [k] * len(v)
    model = make_pipeline(TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), lowercase=True),
                          LogisticRegression(C=40, max_iter=3000))
    model.fit(X, y)
    return model

DE = set("ich wir und der die das ist was wie kostet können möchten hallo bitte wo für uns eine einen gibt wann ihr euch habt wäre würden gerne danke".split())
EN = set("the is are what how do you we can i a to for of your with please there it much".split())
ROMAN_UR = set("kya hai mein aap hum kitna kitne kaise kahan tour ki ka ko se aur nahi mujhe hamein".split())

def detect_lang(text):
    if re.search(r"[\u0600-\u06FF]", text):
        return "ur"
    words = re.findall(r"[a-zäöüß']+", text.lower())
    de = sum(w in DE for w in words) + (1 if re.search(r"[äöüß]", text.lower()) else 0)
    en = sum(w in EN for w in words)
    ur = sum(w in ROMAN_UR for w in words)
    best = max(de, en, ur)
    if best == 0:
        return "unknown"
    if ur == best and ur > en and ur > de:
        return "ur"
    return "de" if de > en else "en"

def classify(model, text):
    probs = model.predict_proba([text])[0]
    classes = list(model.classes_)
    order = sorted(zip(classes, probs), key=lambda t: -t[1])
    return order[0][0], float(order[0][1]), order

def needs_person(text, intent, conf, lang):
    low = text.lower()
    hit = [w for w in SAFETY_WORDS if w in low]
    if hit:
        return True, "Sensitive topic (health / complaint / refund) - a person must reply."
    if len(text.strip()) < 4:
        return True, "Message too short to understand."
    if conf < CONF_THRESHOLD:
        return True, f"Model confidence {conf:.0%} is below {CONF_THRESHOLD:.0%}."
    if intent == "other":
        return True, "Not one of the 4 things this tool can answer."
    if lang == "unknown":
        return True, "Language not recognised (tool supports English, German, Urdu)."
    return False, ""

def entities(text):
    out = {}
    m = re.search(r"(\d+)\s*(people|persons|guests|adults|kids|children|personen|leute|log|bande)", text.lower())
    if m: out["Group size"] = m.group(1)
    m = re.search(r"\b(\d{1,2}[/.]\d{1,2}([/.]\d{2,4})?|\d{1,2}\s*(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*)", text.lower())
    if m: out["Date"] = m.group(0)
    for d in ["monday","tuesday","wednesday","thursday","friday","saturday","sunday","tomorrow","next week","weekend",
              "montag","dienstag","mittwoch","donnerstag","freitag","samstag","sonntag","morgen","wochenende","kal"]:
        if d in text.lower():
            out["Day"] = d; break
    return out

URDU_GIST = {
 "price": "مہمان ٹور کی قیمت پوچھ رہا ہے۔",
 "directions": "مہمان فارم تک پہنچنے کا راستہ / ملنے کی جگہ پوچھ رہا ہے۔",
 "booking": "مہمان ٹور بُک کرنا چاہتا ہے۔",
 "tour_info": "مہمان ٹور کے بارے میں معلومات (دورانیہ / کیا شامل ہے) پوچھ رہا ہے۔",
 "other": "پیغام واضح نہیں، براہ کرم خود پڑھ کر جواب دیں۔",
}
INTENT_LABEL = {"price": "💰 Price", "directions": "📍 Directions", "booking": "📅 Booking",
                "tour_info": "ℹ️ Tour info", "other": "❓ Other"}

REPLIES = {
 "price": {
  "en": "Hello! Our farm tour costs {price} per person and takes about {duration}. It includes: {includes}.",
  "de": "Hallo! Unsere Farmführung kostet {price} pro Person und dauert etwa {duration}. Inklusive: {includes}.",
  "ur": "السلام علیکم! ہمارے فارم ٹور کی قیمت {price} فی کس ہے اور دورانیہ تقریباً {duration} ہے۔ شامل ہے: {includes}۔"},
 "directions": {
  "en": "Hello! Please meet us at {meet_point}. If you can't find us, call {phone}.",
  "de": "Hallo! Bitte treffen Sie uns an diesem Ort: {meet_point}. Wenn Sie uns nicht finden, rufen Sie {phone} an.",
  "ur": "السلام علیکم! براہ کرم {meet_point} پر ملیں۔ اگر راستہ نہ ملے تو {phone} پر کال کریں۔"},
 "booking": {
  "en": "Thank you for your interest! Please tell us your preferred date and number of guests. Noor will check availability and confirm.",
  "de": "Vielen Dank für Ihr Interesse! Bitte nennen Sie uns Ihr Wunschdatum und die Personenzahl. Noor prüft die Verfügbarkeit und bestätigt.",
  "ur": "آپ کی دلچسپی کا شکریہ! براہ کرم اپنی پسندیدہ تاریخ اور مہمانوں کی تعداد بتائیں۔ نور دستیابی دیکھ کر تصدیق کرے گی۔"},
 "tour_info": {
  "en": "Hello! The tour includes: {includes}. It takes about {duration} and starts at {meet_point}.",
  "de": "Hallo! Die Tour beinhaltet: {includes}. Sie dauert etwa {duration} und beginnt bei {meet_point}.",
  "ur": "السلام علیکم! ٹور میں شامل ہے: {includes}۔ دورانیہ تقریباً {duration} ہے اور آغاز {meet_point} سے ہوگا۔"},
}

def draft_reply(intent, lang, cfg):
    l = lang if lang in ("en", "de", "ur") else "en"
    return REPLIES[intent][l].format(**cfg), REPLIES[intent]["ur"].format(**cfg)

# ---------------- Review insights ----------------
THEMES = {
 "Coffee tasting": ["coffee", "tasting", "brew", "roast", "beans", "cup"],
 "Guide & hospitality": ["noor", "guide", "host", "welcom", "friendly", "kind", "hospitab", "family"],
 "Views & nature": ["view", "scenery", "landscape", "beautiful", "mountain", "highland", "nature", "walk"],
 "Food & tea": ["food", "lunch", "meal", "snack", "tea", "breakfast"],
 "Price / value": ["price", "value", "expens", "cheap", "worth", "cost"],
 "Road & finding the farm": ["road", "muddy", "find", "directions", "far", "transport", "lost", "signal"],
 "Language barrier": ["language", "translat", "understand", "communicat", "english"],
 "Duration & pace": ["long", "short", "hours", "rushed", "wait", "late"],
 "Facilities": ["toilet", "bathroom", "seat", "shade", "clean", "water"],
}
POS = ["love", "great", "amazing", "beautiful", "friendly", "delicious", "best", "wonderful", "excellent", "lovely",
       "enjoy", "fantastic", "perfect", "worth", "kind", "warm", "good", "fresh", "highlight", "recommend"]
NEG = ["bad", "dirty", "hard", "difficult", "lost", "muddy", "disappoint", "wish", "rushed", "expensive", "late",
       "confus", "problem", "poor", "lack", "missing", "could be", "wasn't", "couldn't", "no ", "not ", "too ", "unclear", "slippery"]
IDEAS = {
 "Road & finding the farm": "Send visitors a pin + a 3-photo route card by message after booking, and offer a pickup point in the district town.",
 "Language barrier": "Make a bilingual picture card (local language + English/German) for each stop on the tour.",
 "Facilities": "Add a shaded rest spot with water and a simple toilet sign before the next season.",
 "Duration & pace": "Offer two formats: a 90-minute short tour and a half-day tour with a meal.",
 "Price / value": "Create a clear price card showing what is included, plus a family/group bundle.",
 "Food & tea": "Add a paid lunch or tea add-on using produce from the farm.",
 "Coffee tasting": "Sell a 'harvest-to-cup' workshop as a premium add-on.",
 "Guide & hospitality": "Offer a 'meet the family' evening meal as a premium experience.",
 "Views & nature": "Offer a sunrise highland walk as a separate product.",
}

def analyse_reviews(reviews):
    pos, neg = defaultdict(list), defaultdict(list)
    for r in reviews:
        for s in re.split(r"(?<=[.!?;])\s+|\bbut\b", r):
            s = s.strip()
            if len(s) < 8: continue
            sl = s.lower()
            p = sum(w in sl for w in POS); n = sum(w in sl for w in NEG)
            themes = [t for t, kws in THEMES.items() if any(k in sl for k in kws)]
            for t in themes:
                if n > p:
                    neg[t].append(s)
                elif p > 0:
                    pos[t].append(s)
    top = lambda d: sorted(d.items(), key=lambda kv: -len(kv[1]))[:3]
    loved, issues = top(pos), top(neg)
    idea = None
    if issues:
        idea = (issues[0][0], IDEAS.get(issues[0][0], ""))
    elif loved:
        idea = (loved[0][0], IDEAS.get(loved[0][0], ""))
    return loved, issues, idea

def evaluate(model):
    rows, ok, flagged = [], 0, 0
    for text, true in TEST:
        intent, conf, _ = classify(model, text)
        np_, _ = needs_person(text, intent, conf, detect_lang(text))
        correct = intent == true
        ok += correct; flagged += np_
        rows.append({"message": text, "true": true, "predicted": intent, "confidence": round(conf, 2),
                     "correct": correct, "ask_a_person": np_})
    return ok / len(TEST), rows
