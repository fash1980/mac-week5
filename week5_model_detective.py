"""MAC Week 5 | 90-minute interactive Data Mining tutorial.

Run: streamlit run week5_model_detective.py
Requirements: streamlit, supabase, reportlab
Run week5_tutorial_setup.sql in Supabase first. Keep service-role credentials in
Streamlit Secrets, never in this file or a public GitHub repository.

Notes: Student-ID-only identification is not secure authentication. Browser CSS
can discourage text copying/dragging, but screenshots/AI tools cannot be blocked.
"""
from __future__ import annotations

import hashlib
import hmac
import random
import re
from datetime import datetime, timedelta, timezone
from io import BytesIO
from xml.sax.saxutils import escape

import streamlit as st
from supabase import create_client
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether


TITLE = "Week 5 | The Model Detective Challenge"
TABLE = "mac_week5_tutorial_attempts"
LIMIT_MINUTES = 90
st.set_page_config(page_title=TITLE, page_icon="🧠", layout="wide")
st.markdown("""
<style>
.block-container {max-width:1100px;padding-top:1.7rem}
.block-container, .block-container * { -webkit-user-select:none; user-select:none; }
input, textarea { -webkit-user-select:text !important; user-select:text !important; }
img, svg { -webkit-user-drag:none; user-drag:none; pointer-events:none; }
[data-testid="stToolbar"] {visibility:hidden}
div[data-testid="stForm"] {border-radius:14px}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def client(url: str, key: str):
    return create_client(url, key)


try:
    db = client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_SERVICE_ROLE_KEY"])
    seed_secret = str(st.secrets.get("WEEK5_SEED_SECRET", "week5-classroom-fixed-seed"))
except (KeyError, FileNotFoundError):
    st.error("The tutorial is not configured yet. Please contact your lecturer.")
    st.stop()


def utc_now():
    return datetime.now(timezone.utc)


def parsed_time(value: str):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def questions_for(student_id: str):
    """Generate 30 reproducible questions, differing numerically by Student ID."""
    digest = hmac.new(seed_secret.encode(), student_id.encode(), hashlib.sha256).digest()
    r = random.Random(int.from_bytes(digest[:16], "big"))
    qs = []

    def add(topic, prompt, choices, correct, why, kind="select", diagram=None):
        number = len(qs) + 1
        assert correct in choices if kind == "select" else isinstance(correct, (tuple, list))
        if kind == "select":
            assert len(choices) == 4 and len(set(choices)) == 4, f"Duplicate options in Q{number}"
            choices = list(choices)
            r.shuffle(choices)  # Stable student-specific order, not a common answer position.
        qs.append(dict(id=f"Q{number:02}", topic=topic, prompt=prompt,
                       choices=choices, correct=correct, explanation=why,
                       kind=kind, diagram=diagram))

    # 1. Classification and regression (4)
    year = r.choice([1, 2, 3, 4])
    add("Task Type", f"A university predicts the exact final mark (0–100) of Year {year} students. Which task is this?",
        ["Classification", "Regression", "Association mining", "Clustering"], "Regression",
        "A continuous numerical target is a regression problem.")
    threshold = r.choice([45, 50, 55, 60])
    add("Task Type", f"An app labels a student 'At risk' when predicted marks are below {threshold}; otherwise 'On track'. What is its OUTPUT type?",
        ["Categorical", "Continuous numerical", "Unlabelled cluster", "Distance"], "Categorical",
        "The output is one of two named classes, even though the rule uses a number.")
    add("Task Type", "Which pair correctly identifies X (feature) and y (target) for predicting Pass/Fail from attendance?",
        ["X = attendance, y = Pass/Fail", "X = Pass/Fail, y = attendance", "Both are y", "Both are X"],
        "X = attendance, y = Pass/Fail", "Input attendance is the feature; known Pass/Fail is the training target.")
    logistic_p = r.choice([0.32, 0.43, 0.64, 0.78])
    add("Logistic Regression", f"A logistic classifier predicts P(Pass|X) = {logistic_p:.2f}. With a decision threshold of 0.50, which label is assigned?",
        ["Pass", "Fail", "Continuous mark", "Unlabelled cluster"],
        "Pass" if logistic_p >= .5 else "Fail", "Logistic regression turns a predicted class probability into a label using a decision threshold.")

    # 2. Regression and evaluation (4)
    slope = r.choice([0.5, 0.6, 0.7, 0.8])
    intercept = r.choice([12, 16, 20])
    attendance = r.choice([50, 60, 70, 80])
    predicted = round(intercept + slope * attendance, 1)
    candidates = [predicted, round(predicted + 5, 1), round(predicted - 5, 1), round(predicted + 10, 1)]
    r.shuffle(candidates)
    add("Regression", f"The fitted model is mark = {intercept} + {slope} × attendance. What mark is predicted at {attendance}% attendance?",
        [f"{v:.1f}" for v in candidates], f"{predicted:.1f}", "Substitute the input into the fitted equation.")
    actual = int(predicted + r.choice([-6, -4, 3, 5]))
    residual = round(actual - predicted, 1)
    choices = [residual, -residual, abs(residual) + 2, 0.0]
    choices = list(dict.fromkeys(choices))
    while len(choices) < 4:
        choices.append(float(10 + len(choices)))
    r.shuffle(choices)
    add("Regression", f"Actual mark = {actual}, predicted mark = {predicted:.1f}. What is the SIGNED residual (actual − predicted)?",
        [f"{v:+.1f}" for v in choices], f"{residual:+.1f}", "Residual = observed value minus fitted value.")
    add("Neural Networks", "During MLP training, which process calculates gradients so that the weights can be updated?",
        ["Backpropagation", "Bootstrapping", "K-fold splitting", "Association mining"], "Backpropagation",
        "Backpropagation computes loss gradients through the network for optimisation.")
    records = r.choice([20, 40, 60])
    both = records * 3 // 10
    match = records * 4 // 10
    add("Association Rules", f"Among {records} records, {match} match condition X and {both} match BOTH X and Pass. What is support(X → Pass)?",
        ["30%", "75%", "40%", "100%"], "30%",
        "Support uses all records as the denominator: count(X AND Pass) / total.")

    # 3. Decision Trees (5), including a student-specific visual tree
    n_pass = r.randint(5, 9)
    n_fail = r.randint(1, 4)
    add("Decision Trees", f"A tree node contains {n_pass} Pass and {n_fail} Fail students. Is the node pure?",
        ["Yes, pure", "No, mixed", "It must be a regression leaf", "Purity cannot be checked"], "No, mixed",
        "A pure node contains only one class.")
    gains = sorted(r.sample(range(12, 49), 3), reverse=True)
    features = ["Attendance", "Study Hours", "Previous Grade"]
    r.shuffle(features)
    gain_map = dict(zip(features, gains))
    add("Decision Trees", "Choose the root feature with the highest Information Gain: " +
        "; ".join(f"{k} = {v/100:.2f}" for k, v in gain_map.items()),
        features + ["All three have equal gain"], features[0],
        "ID3 greedily selects the feature with the largest Information Gain.")
    add("Decision Trees", f"After one split, branch A has {r.randint(6,10)} Pass / 0 Fail and branch B has 0 Pass / {r.randint(5,9)} Fail. What happens to weighted child entropy?",
        ["It becomes zero", "It becomes one", "It must increase", "It cannot be calculated"], "It becomes zero",
        "Both child nodes contain a single class, so each has entropy zero.")
    cutoff = r.choice([55, 60, 65, 70, 75])
    hours_cutoff = r.choice([3, 4, 5])
    value = cutoff + r.choice([-8, -4, 4, 8])
    hours = hours_cutoff + r.choice([-1, 1, 2])
    is_pass = value >= cutoff and hours >= hours_cutoff
    add("Decision Trees", f"Follow the displayed tree for a student with attendance {value}% and {hours} study hours/week. What is the predicted class?",
        ["Pass", "Fail", "Regression", "Unknown"], "Pass" if is_pass else "Fail",
        "Follow the attendance branch first; the higher-attendance branch then checks study hours.",
        diagram=(cutoff, hours_cutoff))
    add("Decision Trees", "A very deep decision tree scores 99% on training but much lower on unseen data. What is the likely issue?",
        ["Overfitting", "Underfitting", "Label encoding", "Bagging"], "Overfitting",
        "A model can memorise training patterns and fail to generalise.")

    # 4. Bayesian reasoning (4)
    total = r.choice([40, 50, 60])
    passing = total // 2 + r.choice([5, 10])
    add("Bayesian Classification", f"In {total} labelled records, {passing} students passed. What is the prior P(Pass)?",
        [f"{passing/total:.2f}", f"{(total-passing)/total:.2f}", f"{passing/(total+10):.2f}", "1.00"]
        if passing != total // 2 else [f"{passing/total:.2f}", "0.25", "0.75", "1.00"],
        f"{passing/total:.2f}", "Prior = number in the Pass class divided by all labelled records.")
    observed = r.choice([8, 10, 12])
    passed_high = r.choice([p for p in range(4, min(observed, 9)) if 2*p != observed])
    bayes_options = list(dict.fromkeys([f"{passed_high/observed:.3f}",
        f"{(observed-passed_high)/observed:.3f}", f"{passed_high/(observed+5):.3f}", "1.000"]))
    for extra in ["0.125", "0.250", "0.875", "0.950"]:
        if len(bayes_options) < 4 and extra not in bayes_options:
            bayes_options.append(extra)
    add("Bayesian Classification", f"Among {observed} students with HIGH attendance, {passed_high} passed. What is P(Pass | High attendance)?",
        bayes_options, f"{passed_high/observed:.3f}",
        "This is a posterior probability conditional on the attendance evidence.")
    post_pass = r.choice([0.62, 0.71, 0.84])
    add("Bayesian Classification", f"Naïve Bayes gives P(Pass|X)={post_pass:.2f} and P(Fail|X)={1-post_pass:.2f}. Which class is selected?",
        ["Pass", "Fail", "Both", "Neither"], "Pass", "Select the class with the larger posterior probability.")
    add("Bayesian Classification", "What simplifying assumption is made by Naïve Bayes?",
        ["Features are conditionally independent given the class", "All features have equal values",
         "Classes are unlabelled", "The training data is always balanced"],
        "Features are conditionally independent given the class", "Naïve Bayes models feature likelihoods with a conditional-independence assumption.")

    # 5. KNN (4)
    x = r.randint(3, 7)
    counts = ["Pass", "Pass", "Fail"] if r.random() < .5 else ["Fail", "Fail", "Pass"]
    r.shuffle(counts)
    outcome = "Pass" if counts.count("Pass") > counts.count("Fail") else "Fail"
    add("KNN", f"For k=3, the nearest neighbours to student {x} are {', '.join(counts)}. Which label wins?",
        ["Pass", "Fail", "Regression", "No prediction"], outcome,
        "KNN uses the majority class among the k nearest neighbours.")
    add("KNN", "Attendance is measured 0–100, while Study Hours is 0–8. What should be considered before Euclidean KNN?",
        ["Scale the features", "Delete all study-hour values", "Use a larger test set as training", "Convert classes to continuous targets"],
        "Scale the features", "Unscaled high-range features can dominate Euclidean distance.")
    add("KNN", "Compared with k=1, a much larger k generally produces what type of decision boundary?",
        ["Smoother, with greater risk of underfitting", "More sensitive to a single noisy point",
         "An identical boundary", "No classification capability"],
        "Smoother, with greater risk of underfitting", "Larger neighbourhoods average over more local variation.")
    a, b = r.randint(1, 4), r.randint(2, 5)
    distance = round((a*a+b*b)**0.5, 2)
    choices = [distance, float(a+b), float(abs(a-b)), round(distance+3, 2)]
    if len(set(choices)) < 4:
        choices[2] = round(distance + 3, 2)
    r.shuffle(choices)
    add("KNN", f"Two records differ by {a} units in scaled feature 1 and {b} units in scaled feature 2. Their Euclidean distance is approximately:",
        [f"{v:.2f}" for v in choices], f"{distance:.2f}", "Use √(difference1² + difference2²).")

    # 6. Validation and ensembles (5)
    folds = r.choice([4, 5, 10])
    add("Validation", f"In {folds}-fold cross-validation, how many training/validation fits are performed for one hyperparameter setting?",
        [str(folds), str(folds-1), str(2*folds), "1"], str(folds),
        "Each fold takes one turn as the validation partition.")
    add("Validation", "A data scientist scales all records before splitting into train and test sets. What is the main concern?",
        ["Data leakage", "Too many class labels", "KNN cannot use scaled data", "Entropy becomes negative"],
        "Data leakage", "The scaler can learn information from the held-out test data.")
    add("Validation", "For a dataset with strongly imbalanced Pass/Fail classes, which fold strategy best preserves class proportions?",
        ["Stratified K-fold", "Random values with replacement", "Use only majority-class cases", "Leave the data unsplit"],
        "Stratified K-fold", "Stratification retains similar class proportions across folds.")
    tp, fp = r.choice([(18, 6), (24, 8), (30, 10)])
    add("Evaluation", f"A classifier predicts Pass for {tp + fp} students: {tp} really Pass (TP) and {fp} really Fail (FP). What is precision for Pass?",
        ["75%", "25%", "50%", "100%"], "75%",
        "Precision = TP / (TP + FP), so here it is 0.75.")
    add("Ensembles", "Which technique trains base learners on bootstrap samples, usually independently?",
        ["Bagging", "Boosting", "Backpropagation", "Rule pruning"], "Bagging",
        "Bagging combines learners fit to resampled data and often reduces variance.")
    add("Ensembles", "Which description best matches boosting?",
        ["Sequential learners focus on earlier errors", "All learners are exact duplicates trained on identical weights",
         "There is no combining of outputs", "Only unlabelled data are used"],
        "Sequential learners focus on earlier errors", "Boosting builds an ensemble sequentially to improve upon previous stages.")

    # 7. Fix the wrong model: one word/short phrase (3 = 10%).
    train_acc = r.choice([96, 97, 98, 99])
    test_acc = r.choice([52, 56, 59, 63])
    add("Fix the Model", f"A classification model gets {train_acc}% training accuracy but only {test_acc}% test accuracy. Name the main problem (ONE WORD).",
        [], ("overfitting", "overfit"), "Large train-test gaps often indicate overfitting.", kind="short")
    prep = r.choice(["imputation means", "scaling parameters", "feature-selection scores"])
    add("Fix the Model", f"A student calculated {prep} using the full dataset BEFORE train/test splitting. Name the mistake (ONE OR TWO WORDS).",
        [], ("data leakage", "leakage"), "Fitting preprocessing on held-out data causes leakage.", kind="short")
    raw_range = r.choice([8, 10, 12])
    add("Fix the Model", f"A KNN model compares attendance (0–100) and study hours (0–{raw_range}) without adjusting their scales. What preprocessing step should be applied? (ONE WORD).",
        [], ("scaling", "normalization", "normalisation", "standardization", "standardisation"),
        "Distance-sensitive features generally need comparable scales.", kind="short")

    assert len(qs) == 30 and sum(q["kind"] == "select" for q in qs) == 27
    return qs


def normalize(value):
    return re.sub(r"[^a-z0-9]+", " ", str(value).casefold()).strip()


def is_correct(question, answer):
    if question["kind"] == "short":
        return normalize(answer) in {normalize(x) for x in question["correct"]}
    return answer == question["correct"]


def summary(questions, answers):
    attempted = sum(bool(answers.get(q["id"])) for q in questions)
    score = sum(is_correct(q, answers.get(q["id"], "")) for q in questions if answers.get(q["id"]))
    return attempted, score


def fetch_attempt(student_id):
    data = db.table(TABLE).select("*").eq("student_id", student_id).limit(1).execute().data
    return data[0] if data else None


def persist(answers=None, finished=False):
    """One-row atomic field update. Never overwrite server state after final submission."""
    a = st.session_state.attempt
    payload = {"answers": answers if answers is not None else a["answers"],
               "updated_at": utc_now().isoformat()}
    if finished:
        payload["submitted_at"] = utc_now().isoformat()
        payload["score"] = summary(st.session_state.questions, payload["answers"])[1]
    result = (db.table(TABLE).update(payload).eq("student_id", a["student_id"])
              .is_("submitted_at", "null").execute())
    if not result.data:
        raise RuntimeError("This attempt is no longer editable. Reload the page.")
    a.update(result.data[0])


def save_response(qid, key):
    a = st.session_state.attempt
    if a.get("submitted_at") or (a.get("answers") or {}).get(qid):
        return
    value = st.session_state.get(key, "")
    if value == "Choose an answer":
        value = ""
    updated = dict(a.get("answers") or {})
    updated[qid] = value
    try:
        persist(answers=updated)
        st.session_state.save_error = ""
    except Exception:
        st.session_state.save_error = "Autosave failed. Please retry this answer before leaving the page."


def render_tree(cutoff, hours):
    st.html(f"""
    <div style="padding:8px;background:#f5f9fb;border-radius:12px;max-width:620px">
    <svg viewBox="0 0 600 230" role="img" aria-label="Decision tree showing attendance and study hour thresholds"
         style="width:100%;height:auto" xmlns="http://www.w3.org/2000/svg">
    <g stroke="#9eb9c6" stroke-width="3" fill="none"><path d="M300 52L145 105M300 52L445 105M445 142L365 190M445 142L530 190"/></g>
    <rect x="193" y="8" width="214" height="46" rx="12" fill="#1c6474"/>
    <text x="300" y="36" text-anchor="middle" fill="white" font-size="19">Attendance ≥ {cutoff}%?</text>
    <rect x="85" y="105" width="120" height="45" rx="10" fill="#f4dcd8"/>
    <text x="145" y="132" text-anchor="middle" fill="#7a4237" font-size="18">Fail</text>
    <rect x="349" y="103" width="192" height="46" rx="10" fill="#1c6474"/>
    <text x="445" y="132" text-anchor="middle" fill="white" font-size="18">Hours ≥ {hours}?</text>
    <rect x="311" y="185" width="106" height="38" rx="10" fill="#f4dcd8"/>
    <text x="364" y="210" text-anchor="middle" fill="#7a4237" font-size="18">Fail</text>
    <rect x="483" y="185" width="105" height="38" rx="10" fill="#d9f1e8"/>
    <text x="535" y="210" text-anchor="middle" fill="#176449" font-size="18">Pass</text>
    <text x="196" y="88" fill="#475569" font-size="14">No</text>
    <text x="386" y="87" fill="#475569" font-size="14">Yes</text>
    <text x="350" y="171" fill="#475569" font-size="13">No</text>
    <text x="505" y="171" fill="#475569" font-size="13">Yes</text>
    </svg></div>""")


def make_pdf(attempt, questions):
    """Complete answer/feedback report for submission to myTIMeS."""
    out = BytesIO()
    doc = SimpleDocTemplate(out, pagesize=A4, leftMargin=18*mm, rightMargin=18*mm,
                            topMargin=18*mm, bottomMargin=17*mm)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="W5Title", parent=styles["Title"], fontSize=17, leading=21,
                              textColor=colors.HexColor("#16485A"), spaceAfter=13))
    styles.add(ParagraphStyle(name="W5Body", parent=styles["BodyText"], fontSize=8.7, leading=12.5, spaceAfter=5))
    styles.add(ParagraphStyle(name="W5Question", parent=styles["BodyText"], fontSize=9.2,
                              leading=13.5, spaceAfter=7, spaceBefore=7))

    def p(text, style="W5Body"):
        pdf_safe = str(text)
        for old, new in {"≥": ">=", "≤": "<=", "×": "x", "√": "sqrt", "²": "^2",
                         "–": "-", "—": "-", "−": "-", "→": "->", "’": "'"}.items():
            pdf_safe = pdf_safe.replace(old, new)
        return Paragraph(escape(pdf_safe).replace("\n", "<br/>"), styles[style])

    answers = attempt.get("answers") or {}
    completed, score = summary(questions, answers)
    story = [p("MAC Week 5 — Model Detective Tutorial", "W5Title")]
    identity = [["Student", attempt["student_name"]], ["Student ID", attempt["student_id"]],
                ["Duration", "90 minutes"], ["Submitted (UTC)", str(attempt.get("submitted_at") or "Not submitted")],
                ["Result", f"{score}/30 ({score/30*100:.1f}%) | {completed}/30 answered"]]
    t = Table([[p(a), p(b)] for a,b in identity], colWidths=[38*mm, 133*mm])
    t.setStyle(TableStyle([("BACKGROUND", (0,0),(-1,-1), colors.HexColor("#EDF5F6")),
                          ("VALIGN", (0,0),(-1,-1), "TOP"), ("LEFTPADDING",(0,0),(-1,-1),10),
                          ("TOPPADDING",(0,0),(-1,-1),7), ("BOTTOMPADDING",(0,0),(-1,-1),7)]))
    story.extend([t, Spacer(1, 10), p("Automatically marked report. Review the feedback below.")])
    for q in questions:
        answer = answers.get(q["id"], "")
        correct = is_correct(q, answer) if answer else False
        answer_text = str(answer) if answer else "Not answered"
        expected = q["correct"] if q["kind"] == "select" else ", ".join(q["correct"])
        block = [p(f"{q['id']} — {q['topic']}: {q['prompt']}", "W5Question"),
                 p(f"Your answer: {answer_text}  |  {'Correct' if correct else 'Incorrect / missing'}  |  Mark: {int(correct)}/1"),
                 p(f"Correct answer: {expected}"), p(f"Feedback: {q['explanation']}")]
        story.append(KeepTogether(block))
    story.append(Spacer(1, 12))
    story.append(p("Submission: Upload this PDF to myTIMeS as instructed by your lecturer."))
    doc.build(story)
    return out.getvalue()


st.title("🧠 The Model Detective Challenge")
st.caption("MAC | Week 5 — Classification, Prediction and Evaluation")
st.write("Work through 30 short challenges using the data shown on your screen. Your work is saved as you answer each question.")

if "attempt" not in st.session_state:
    with st.form("identity"):
        sid = st.text_input("Student ID", placeholder="Enter your Student ID", max_chars=40).strip().upper()
        name = st.text_input("Full name", placeholder="As shown in your university records", max_chars=100).strip()
        start = st.form_submit_button("Start / Resume Tutorial", type="primary")
    if start:
        if not re.fullmatch(r"[A-Z0-9_-]{3,40}", sid) or len(name) < 3:
            st.warning("Please enter your full name and a valid Student ID.")
        else:
            try:
                existing = fetch_attempt(sid)
                if existing and normalize(existing["student_name"]) != normalize(name):
                    st.error("The name does not match this Student ID's saved attempt. Contact your lecturer.")
                else:
                    if not existing:
                        payload = dict(student_id=sid, student_name=name, answers={}, started_at=utc_now().isoformat())
                        existing = db.table(TABLE).insert(payload).execute().data[0]
                    st.session_state.attempt = existing
                    st.session_state.questions = questions_for(sid)
                    st.session_state.current = 0
                    st.session_state.save_error = ""
                    st.rerun()
            except Exception:
                st.error("Could not connect to the submission database. Please tell your lecturer.")
    st.stop()

attempt = st.session_state.attempt
qs = st.session_state.questions
elapsed = utc_now() - parsed_time(attempt["started_at"])
remaining = max(timedelta(), timedelta(minutes=LIMIT_MINUTES) - elapsed)
if elapsed >= timedelta(minutes=LIMIT_MINUTES) and not attempt.get("submitted_at"):
    try:
        persist(finished=True)
    except Exception:
        st.error("Time has ended, but the final submission could not be saved. Please refresh and contact your lecturer.")
        st.stop()

answers = attempt.get("answers") or {}
answered, score = summary(qs, answers)
left, mid, right = st.columns(3)
left.metric("Answered", f"{answered}/30")
mid.metric("Current marks", f"{score}/30")
right.metric("Time remaining", "Submitted" if attempt.get("submitted_at") else str(remaining).split(".")[0])
st.progress(answered/30, text=f"Progress: {answered} of 30 answered")

if st.session_state.save_error:
    st.error(st.session_state.save_error)

if attempt.get("submitted_at"):
    st.success("Your tutorial is submitted. Download your feedback report and upload it to myTIMeS.")
    st.metric("Final result", f"{score}/30 ({score/30:.0%})")
    pdf = make_pdf(attempt, qs)
    st.download_button("⬇ Download My Marked PDF", data=pdf,
                       file_name=f"MAC_Week5_{attempt['student_id']}_Feedback.pdf",
                       mime="application/pdf", type="primary", use_container_width=True)
    with st.expander("See my answers and feedback"):
        for q in qs:
            chosen = answers.get(q["id"], "Not answered")
            st.write(f"**{q['id']} ({q['topic']}):** {chosen} — {'✅' if is_correct(q, chosen) else '❌'}")
            st.caption(q["explanation"])
    st.stop()

st.warning("Once you submit or your 90 minutes end, answers cannot be changed. Keep your Student ID and name consistent if you reconnect.")

index = st.session_state.current
q = qs[index]
st.divider()
st.subheader(f"Question {index + 1} of 30")
st.caption(q["topic"] + (" • Fix the wrong model" if q["kind"] == "short" else " • Select the best answer"))
st.caption(f"Assigned student: {attempt['student_name']} | ID: {attempt['student_id']}")
st.write(q["prompt"])
if q["diagram"]:
    render_tree(*q["diagram"])

widget = f"week5_{attempt['student_id']}_{q['id']}"
if widget not in st.session_state:
    st.session_state[widget] = answers.get(q["id"], "") or ("Choose an answer" if q["kind"] == "select" else "")

if q["kind"] == "select":
    st.selectbox("Your answer", ["Choose an answer"] + q["choices"],
                 key=widget, disabled=bool(answers.get(q["id"])),
                 on_change=save_response, args=(q["id"], widget))
else:
    st.text_input("Your answer (one word or short phrase)", key=widget,
                  max_chars=50, disabled=bool(answers.get(q["id"])),
                  on_change=save_response, args=(q["id"], widget))
    st.caption("Press Enter to save. Your first saved answer is final.")

if st.session_state.save_error:
    if st.button("↻ Retry saving this answer"):
        save_response(q["id"], widget)
        st.rerun()

chosen = answers.get(q["id"], "")
if chosen:
    if is_correct(q, chosen):
        st.success("✓ Correct — 1/1 mark saved")
    else:
        st.info("Answer saved — 0/1 mark. Review your lecture notes before continuing.")
    # No correct-answer reveal until final submission.
else:
    st.caption("Your first answer will be saved and marked automatically. You cannot change it afterward.")

st.divider()
back, next_col = st.columns(2)
with back:
    if st.button("← Previous", disabled=index == 0, use_container_width=True):
        st.session_state.current -= 1
        st.rerun()
with next_col:
    if st.button("Next →", disabled=index == 29, use_container_width=True):
        st.session_state.current += 1
        st.rerun()

if answered == 30:
    st.success("All 30 questions are answered. You may now submit your tutorial.")
    if st.button("Submit Final Answers and Generate PDF", type="primary", use_container_width=True):
        try:
            persist(finished=True)
            st.rerun()
        except Exception:
            st.error("Final submission was not saved. Please try again.")
else:
    st.caption(f"Final submission unlocks after all 30 questions are answered ({30-answered} remaining).")
