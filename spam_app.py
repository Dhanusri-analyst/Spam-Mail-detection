import re
import html
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# =====================================================================
# Core logic (no Streamlit calls in this section)
# =====================================================================

def clean_text(text):
    """Must match the cleaning function used when the model was trained."""
    text = str(text).lower()
    text = re.sub(r"(https?://\S+|www\.\S+)", " urltoken ", text)
    text = re.sub(r"\S+@\S+", " emailtoken ", text)
    text = re.sub(r"[£$€]\s?\d*[\d,.]*", " moneytoken ", text)
    text = re.sub(r"\b\d{5,}\b", " longnumtoken ", text)
    text = re.sub(r"\d+", " numtoken ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def compute_scores(model, messages):
    """Return a spam score between 0 and 1 for each message."""
    cleaned = [clean_text(m) for m in messages]
    if hasattr(model, "predict_proba"):
        return model.predict_proba(cleaned)[:, 1]
    d = model.decision_function(cleaned)          # e.g. LinearSVC
    return 1 / (1 + np.exp(-2 * d))               # squash to 0-1, 0.5 = decision boundary


def build_weights(model):
    """Word -> weight (positive = spam-like, negative = ham-like). Empty if unsupported."""
    try:
        vec = model.named_steps["tfidf"]
        clf = model.named_steps["clf"]
    except Exception:
        return {}
    if hasattr(clf, "coef_"):
        w = clf.coef_[0]
    elif hasattr(clf, "feature_log_prob_"):
        w = clf.feature_log_prob_[1] - clf.feature_log_prob_[0]
    else:
        return {}
    return dict(zip(vec.get_feature_names_out(), w))


def highlight_html(message, weights, strength=0.10):
    """Colour words that push the model towards spam (red) or ham (green)."""
    if not weights:
        return html.escape(message)
    limit = strength * max(abs(v) for v in weights.values())
    parts = []
    for tok in re.findall(r"\S+|\s+", message):
        if tok.isspace():
            parts.append(tok)
            continue
        cleaned_tok = clean_text(tok)
        w = 0.0 if cleaned_tok == "numtoken" else weights.get(cleaned_tok, 0.0)  # ignore plain small numbers
        safe = html.escape(tok)
        if w >= limit:
            parts.append(f"<span style='background:#ffd6d0;color:#8a1c0d;padding:1px 4px;"
                         f"border-radius:4px;font-weight:600'>{safe}</span>")
        elif w <= -limit:
            parts.append(f"<span style='background:#d8f0dc;color:#14532d;padding:1px 4px;"
                         f"border-radius:4px'>{safe}</span>")
        else:
            parts.append(safe)
    return "".join(parts)


def label_from_score(score, threshold):
    return "SPAM" if score >= threshold else "HAM"


# =====================================================================
# UI
# =====================================================================
st.set_page_config(page_title="Spam Mail Detector", page_icon="📧", layout="wide")


@st.cache_resource
def load_model():
    return joblib.load("spam_classifier.joblib")


model = load_model()
weights = build_weights(model)

EXAMPLES = {
    "🎁 Prize scam": "Congratulations! You have won a £1000 prize. Call 09061701461 now to claim your reward!",
    "🔗 Phishing link": "URGENT! Your account has been selected for a free iPhone. Click www.claim-prize.com now",
    "💬 Normal chat": "Hey, are we still meeting for lunch at 1 pm tomorrow?",
    "📚 College note": "Can you send me the notes from yesterday's class?",
}

if "history" not in st.session_state:
    st.session_state.history = []
if "msg" not in st.session_state:
    st.session_state.msg = ""
if "last" not in st.session_state:
    st.session_state.last = None


def set_example(text):
    st.session_state.msg = text


def clear_text():
    st.session_state.msg = ""
    st.session_state.last = None


# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("⚙️ Settings")
    threshold = st.slider(
        "Spam threshold", 0.10, 0.90, 0.50, 0.05,
        help="Lower = stricter filter (catches more spam but may flag real mail). "
             "Higher = more lenient.",
    )
    show_words = st.checkbox("Highlight influential words", value=True)
    st.divider()
    h = st.session_state.history
    st.subheader("Session stats")
    c1, c2 = st.columns(2)
    c1.metric("Checked", len(h))
    c2.metric("Spam found", sum(1 for r in h if r["Result"] == "SPAM"))

st.title("📧 Spam Mail Detector")
st.caption("TF-IDF + machine learning model trained on 5,000+ labelled SMS/email messages.")

tab_single, tab_batch, tab_about = st.tabs(["🔍 Check a message", "📂 Check a CSV file", "ℹ️ About the model"])

# ---------------- Tab 1: single message ----------------
with tab_single:
    st.write("**Try an example or type your own:**")
    cols = st.columns(len(EXAMPLES))
    for col, (name, text) in zip(cols, EXAMPLES.items()):
        col.button(name, on_click=set_example, args=(text,), use_container_width=True)

    st.text_area("Message", key="msg", height=150,
                 placeholder="Type or paste an SMS / email message here...")

    b1, b2, _ = st.columns([1, 1, 4])
    if b1.button("Analyze", type="primary", use_container_width=True):
        if st.session_state.msg.strip():
            st.session_state.last = st.session_state.msg
            score = float(compute_scores(model, [st.session_state.msg])[0])
            st.session_state.history.insert(0, {
                "Message": st.session_state.msg[:80] + ("…" if len(st.session_state.msg) > 80 else ""),
                "Result": label_from_score(score, threshold),
                "Spam score": round(score, 3),
            })
        else:
            st.warning("Please enter a message first.")
    b2.button("Clear", on_click=clear_text, use_container_width=True)

    # The result is recomputed on every rerun, so moving the slider updates it live.
    if st.session_state.last:
        msg = st.session_state.last
        score = float(compute_scores(model, [msg])[0])
        verdict = label_from_score(score, threshold)

        st.divider()
        left, right = st.columns([1, 2])
        with left:
            if verdict == "SPAM":
                st.error("🚫 SPAM")
            else:
                st.success("✅ HAM (not spam)")
            st.metric("Spam score", f"{score * 100:.0f}%")
            st.progress(min(max(score, 0.0), 1.0))
            st.caption(f"Flagged as spam when the score is ≥ {threshold * 100:.0f}% (change this in the sidebar).")
        with right:
            if show_words and weights:
                st.markdown("**Why?** &nbsp; 🟥 pushes towards spam &nbsp; 🟩 pushes towards ham")
                st.markdown(highlight_html(msg, weights), unsafe_allow_html=True)
            elif show_words:
                st.info("Word highlighting is not available for this model type.")

    if st.session_state.history:
        st.divider()
        st.subheader("History")
        hist_df = pd.DataFrame(st.session_state.history)
        st.dataframe(hist_df, use_container_width=True, hide_index=True)
        if st.button("Clear history"):
            st.session_state.history = []
            st.rerun()

# ---------------- Tab 2: batch CSV ----------------
with tab_batch:
    st.write("Upload a CSV file with a column of messages and the model will label every row.")
    up = st.file_uploader("Choose a CSV file", type="csv")
    if up is not None:
        try:
            data = pd.read_csv(up)
        except Exception as e:
            st.error(f"Could not read the file: {e}")
            data = None

        if data is not None and not data.empty:
            default = list(data.columns).index("Message") if "Message" in data.columns else 0
            col = st.selectbox("Which column contains the messages?", data.columns, index=default)
            texts = data[col].fillna("").astype(str).tolist()
            scores = compute_scores(model, texts)

            out = data.copy()
            out["Spam score"] = np.round(scores, 3)
            out["Prediction"] = ["SPAM" if s >= threshold else "HAM" for s in scores]

            n_spam = int((out["Prediction"] == "SPAM").sum())
            m1, m2, m3 = st.columns(3)
            m1.metric("Messages", len(out))
            m2.metric("Spam", n_spam)
            m3.metric("Spam share", f"{n_spam / len(out) * 100:.1f}%")

            st.bar_chart(out["Prediction"].value_counts())

            view = st.radio("Show", ["All", "SPAM only", "HAM only"], horizontal=True)
            if view == "SPAM only":
                shown = out[out["Prediction"] == "SPAM"]
            elif view == "HAM only":
                shown = out[out["Prediction"] == "HAM"]
            else:
                shown = out
            st.dataframe(shown, use_container_width=True, hide_index=True)

            st.download_button("⬇️ Download results as CSV", out.to_csv(index=False).encode("utf-8"),
                               file_name="spam_predictions.csv", mime="text/csv")
        elif data is not None:
            st.warning("The file is empty.")

# ---------------- Tab 3: about ----------------
with tab_about:
    st.subheader("How it works")
    st.markdown(
        """
1. **Clean** the text: lowercase, replace URLs / emails / money / phone numbers with tokens, remove punctuation.
2. **TF-IDF** turns each message into numbers (single words and word pairs).
3. A **machine learning classifier** scores how spam-like the message is.
4. The **threshold** in the sidebar decides where "spam" starts.
        """
    )
    st.subheader("Test-set results")
    st.dataframe(pd.DataFrame({
        "Metric": ["Accuracy", "Spam precision", "Spam recall", "Spam F1"],
        "Score": ["98.7%", "96.8%", "93.0%", "0.948"],
    }), hide_index=True)
    st.caption("Accuracy alone is misleading because only about 13% of messages are spam, "
               "so precision and recall on the spam class matter more.")
    st.warning("This is a learning project. Do not rely on it as your only protection against phishing.")