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
import streamlit.components.v1 as components
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
    """Generate 60 reproducible questions, differing numerically by Student ID."""
    digest = hmac.new(seed_secret.encode(), student_id.encode(), hashlib.sha256).digest()
    r = random.Random(int.from_bytes(digest[:16], "big"))
    qs = []

    def add(topic, prompt, choices, correct, why, kind="select", diagram=None,
            experiment=None, params=None):
        number = len(qs) + 1
        assert (correct in choices or experiment) if kind == "select" else isinstance(correct, (tuple, list))
        if kind == "select":
            assert len(choices) == 4 and len(set(choices)) == 4, f"Duplicate options in Q{number}"
            choices = list(choices)
            r.shuffle(choices)  # Stable student-specific order, not a common answer position.
        qs.append(dict(id=f"Q{number:02}", topic=topic, prompt=prompt,
                       choices=choices, correct=correct, explanation=why,
                       kind=kind, diagram=diagram, experiment=experiment,
                       params=params or {}))

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

    # Part B - model experiments. Sliders alter the visualised model, and the
    # student's answer is checked against the saved slider setting.
    add("Interactive Decision Tree", "EXPERIMENT: Move the attendance split threshold. For the displayed student, which class does the updated tree predict?",
        ["Pass", "Fail", "Unknown", "Regression"], None,
        "If attendance falls below the split, the prediction is Fail; otherwise study hours decide.",
        experiment="tree_cutoff", params={"attendance":r.choice([58,64,72,81]), "hours":r.choice([4,5,6]),"hours_min":4})
    add("Interactive KNN", "EXPERIMENT: Change k (number of neighbours). Which class wins the currently displayed vote?",
        ["Pass", "Fail", "Tie", "No prediction"], None,
        "Count Pass and Fail labels among only the first k neighbours.",
        experiment="knn_k", params={"neighbours":[r.choice(["P","F"]) for _ in range(7)]})
    add("Neural Network", "EXPERIMENT: Move the first weight. What does the ReLU output neuron produce at this setting?",
        ["Positive activation", "Zero activation", "Always a probability", "Cannot calculate"], None,
        "The neuron computes z = 2w1 + 1(1) - 2. ReLU(z) is max(0,z).",
        experiment="neuron_weight")
    add("Dropout", "EXPERIMENT: Increase/decrease the dropout rate. How many hidden neurons remain active in this illustrated pass?",
        ["4", "5", "6", "7"], None,
        "This simplified illustration disables the selected proportion of 8 hidden units. Actual dropout masks are sampled randomly during training.",
        experiment="dropout")
    add("Bagging & Bootstrap", "EXPERIMENT: Switch the bootstrap sample. What is the majority vote of its three trained learners?",
        ["Pass", "Fail", "Tie", "No majority"], None,
        "Each resampled dataset can produce different independent learners; take the majority of their three labels.",
        experiment="bagging_vote", params={"votes":[r.choice([["P","P","F"],["F","P","F"],["P","F","P"]]) for _ in range(3)]})
    add("Boosting", "EXPERIMENT: Adjust the weight of learner A. Which label wins the weighted vote of A=Pass, B=Fail and C=Fail?",
        ["Pass", "Fail", "Tie", "No learner"], None,
        "Compare A's selected weight with B+C's total weight of 4.",
        experiment="boosting_weight")
    add("Logistic Regression", "EXPERIMENT: Move the classification threshold. Which label is assigned to the fixed predicted probability?",
        ["Pass", "Fail", "Clustering", "Numerical mark"], None,
        "The positive label is selected when the predicted probability is at least the threshold.",
        experiment="logistic_threshold", params={"prob":r.choice([0.48,0.58,0.62])})
    add("Interactive Regression", "EXPERIMENT: Change the slope while attendance stays at 60%. In which range is the new predicted mark?",
        ["0–40", "41–60", "61–80", "Above 80"], None,
        "For this model, predicted mark = 15 + slope × 60.",
        experiment="regression_slope")
    add("Cross-Validation", "EXPERIMENT: Change the number of folds k for 60 records. How many validation records are in one equally sized fold?",
        ["10", "12", "15", "20"], None,
        "Each validation fold has 60/k records when the folds are evenly sized.",
        experiment="cv_fold")
    add("Gradient Descent", "EXPERIMENT: Adjust learning rate. Does ONE gradient step increase or decrease the toy loss?",
        ["Loss decreases", "Loss increases", "No change", "Cannot be determined"], None,
        "With L=(w−2)² at w=4, gradient is 4. New w=4−4×learning_rate.",
        experiment="gradient_lr")
    add("Confusion Matrix", "EXPERIMENT: Move the decision threshold and inspect the new TP/FP counts. Which precision range applies?",
        ["Below 70%", "70–79%", "80% or above", "Not defined"], None,
        "Precision = TP/(TP+FP), calculated from the displayed counts.",
        experiment="precision_threshold")
    add("Association Rules", "EXPERIMENT: Move minimum support. A rule has support 25%. Does the rule pass the new threshold?",
        ["Retained", "Filtered out", "Confidence becomes zero", "Lift becomes zero"], None,
        "Keep a rule if its support meets or exceeds the chosen minimum support.",
        experiment="support_threshold")

    # 15 story-based objective questions. Different numbers and contexts make
    # students interpret the case rather than recite a definition.
    add("Dropout Story", "A neural network memorises training samples. During TRAINING, dropout temporarily switches off randomly chosen hidden units. Why might this help?",
        ["Reduces reliance on specific neurons", "Guarantees 100% accuracy", "Increases the number of training labels", "Eliminates need for validation"],
        "Reduces reliance on specific neurons", "Dropout acts as a regulariser and discourages co-adaptation.")
    add("Dropout Story", "At inference time, should standard dropout continue randomly removing hidden neurons?",
        ["No, normally disabled at inference", "Yes, always remove half", "Only if the target is numeric", "Only for decision trees"],
        "No, normally disabled at inference", "Dropout is ordinarily active during training and disabled during standard inference.")
    add("MLP Story", f"A model uses {r.choice([4,6,8])} input features and nonlinear hidden layers to classify images. Which component permits nonlinear decision boundaries?",
        ["Activation functions", "The Student ID", "Test-set labels", "A fixed output threshold alone"],
        "Activation functions", "Nonlinear activations allow compositions beyond a linear map.")
    add("MLP Story", "A gradient points in the direction of increasing loss. To minimise loss, weights usually move:",
        ["Opposite the gradient", "Along the gradient", "Randomly with no objective", "Only to exactly zero"],
        "Opposite the gradient", "Gradient descent updates parameters in the negative-gradient direction.")
    add("Bagging Story", f"From a class of {r.choice([20,30,40])} training records, a bootstrap sample is created WITH replacement. Which outcome is possible?",
        ["Some records repeat and some are omitted", "Every record appears exactly once", "No record can repeat", "Only test records are selected"],
        "Some records repeat and some are omitted", "Bootstrap sampling draws with replacement from the training set.")
    add("Bagging Story", "Three bagged trees predict [Pass, Fail, Pass] for a new record. Which ensemble output is correct?",
        ["Pass", "Fail", "Average score 0.5", "Cannot vote with trees"], "Pass",
        "The class with two of three votes wins.")
    add("Boosting Story", "A classifier repeatedly misclassifies a small group of difficult cases. Which ensemble strategy explicitly emphasises correcting previous stages?",
        ["Boosting", "Bagging", "A single untrained perceptron", "Random hold-out"], "Boosting",
        "Boosting trains stages sequentially to improve earlier predictions or optimise a loss.")
    add("Evaluation Story", f"A medical screening dataset has {r.choice([90,95,98])}% negative cases. Which metric alone can look deceptively high for a model predicting only Negative?",
        ["Accuracy", "Sensitivity", "Recall for Positive", "True-positive count"], "Accuracy",
        "With large class imbalance, majority-class accuracy can hide missed positives.")
    add("Validation Story", "A researcher tunes hyperparameters by repeatedly checking the FINAL test set. What is the better design?",
        ["Use a separate validation set or CV during tuning", "Keep tuning on the test set", "Train on all test labels first", "Remove all validation"],
        "Use a separate validation set or CV during tuning", "A final test set should remain untouched during model selection.")
    add("Validation Story", "Patient visits are ordered in time. Which split avoids training on future records to predict earlier visits?",
        ["Chronological validation", "Random future-to-past shuffle", "Use future records as training only", "Sort by student name"],
        "Chronological validation", "Time-ordered data need chronology-aware evaluation.")
    add("Pruning Story", "An overgrown decision tree fits noisy training details. Which action may improve generalisation?",
        ["Prune weak branches", "Add more unnecessary branches", "Train on the test set", "Remove all target labels"],
        "Prune weak branches", "Pruning reduces complexity and can lower variance.")
    add("Regularisation Story", "Which regularisation term can drive some coefficients exactly to zero and perform feature selection?",
        ["L1", "L2 only", "MAE", "Information gain"], "L1",
        "L1's absolute-value penalty can yield sparse coefficient estimates.")
    add("Association Story", f"A class association rule X → Pass has confidence {r.choice([80,85,90])}%. What does confidence represent?",
        ["Fraction matching X that also Pass", "Fraction of all records matching X and Pass", "Model training accuracy", "Number of neural layers"],
        "Fraction matching X that also Pass", "Confidence is P(Pass|X), not overall support.")
    add("Bayesian Story", "A rare positive class has high likelihood under a symptom, but a very low prior. What must Bayes' theorem combine?",
        ["Prior and likelihood", "Only the likelihood", "Only the training accuracy", "Only the largest feature"],
        "Prior and likelihood", "The posterior depends on both prevalence (prior) and evidence (likelihood).")
    add("Model Selection", "A lecturer needs decisions that students can explain as clear IF–THEN paths. Which first model is suitable?",
        ["Small decision tree", "Very deep MLP only", "Opaque ensemble without explanations", "Unsupervised K-means"],
        "Small decision tree", "A compact tree provides directly interpretable decision paths.")

    # Three further one-phrase diagnostic fixes: 6 short answers out of 60.
    add("Fix the Model", "A deep decision tree memorises noise. Name ONE technique that removes weak branches (ONE WORD).",
        [], ("pruning", "prune"), "Pruning can simplify a tree and reduce overfitting.", kind="short")
    add("Fix the Model", "A model is evaluated with 5-fold CV, but minority-class records are almost absent in some folds. Name the fold strategy to fix this (ONE WORD).",
        [], ("stratification", "stratified", "stratified k fold", "stratified kfold"),
        "Stratification maintains class proportions across folds.", kind="short")
    add("Fix the Model", "A neural network relies too heavily on individual hidden neurons. Name the TRAINING technique that randomly disables some units (ONE WORD).",
        [], ("dropout",), "Dropout can regularise a neural network during training.", kind="short")

    assert len(qs) == 60 and sum(q["kind"] == "select" for q in qs) == 54
    return qs


def normalize(value):
    return re.sub(r"[^a-z0-9]+", " ", str(value).casefold()).strip()


EXPERIMENT_CONTROLS = {
    "tree_cutoff": ("Attendance threshold (%)", 50, 85, 5, 65),
    "knn_k": ("Number of neighbours (k)", 1, 7, 2, 3),
    "neuron_weight": ("Weight w1", -2, 4, 1, 1),
    "dropout": ("Hidden neurons to disable (out of 8)", 1, 4, 1, 2),
    "bagging_vote": ("Bootstrap sample", 1, 3, 1, 1),
    "boosting_weight": ("Weight of learner A", 1, 6, 1, 3),
    "logistic_threshold": ("Classification threshold", 0.3, 0.7, 0.1, 0.5),
    "regression_slope": ("Regression slope", 0.3, 1.0, 0.1, 0.6),
    "cv_fold": ("Number of folds (k)", 3, 6, 1, 5),
    "gradient_lr": ("Learning rate", 0.1, 1.2, 0.1, 0.3),
    "precision_threshold": ("Decision-threshold level", 1, 3, 1, 2),
    "support_threshold": ("Minimum support (%)", 10, 40, 5, 20),
}


def answer_value(answer):
    return answer.get("value", "") if isinstance(answer, dict) else answer


def expected_for(question, control=None):
    exp = question.get("experiment")
    if not exp:
        return question["correct"]
    if control is None:
        control = EXPERIMENT_CONTROLS[exp][4]
    param = question.get("params") or {}
    if exp == "tree_cutoff":
        return "Pass" if param["attendance"] >= control and param["hours"] >= param["hours_min"] else "Fail"
    if exp == "knn_k":
        nearby = param["neighbours"][:int(control)]
        return "Pass" if nearby.count("P") > nearby.count("F") else "Fail"
    if exp == "neuron_weight":
        return "Positive activation" if 2*control + 1 - 2 > 0 else "Zero activation"
    if exp == "dropout":
        return str(8-int(control))
    if exp == "bagging_vote":
        votes = param["votes"][int(control)-1]
        return "Pass" if votes.count("P") > votes.count("F") else "Fail"
    if exp == "boosting_weight":
        return "Pass" if control > 4 else "Fail" if control < 4 else "Tie"
    if exp == "logistic_threshold":
        return "Pass" if param["prob"] >= round(float(control), 2) else "Fail"
    if exp == "regression_slope":
        value = 15 + round(float(control), 2)*60
        return "0–40" if value <= 40 else "41–60" if value <= 60 else "61–80" if value <= 80 else "Above 80"
    if exp == "cv_fold":
        return str(60//int(control))
    if exp == "gradient_lr":
        newer = 4 - 4*float(control)
        return "Loss decreases" if (newer-2)**2 < 4-1e-8 else "Loss increases" if (newer-2)**2 > 4+1e-8 else "No change"
    if exp == "precision_threshold":
        tp, fp = {1:(9,6),2:(7,2),3:(4,1)}[int(control)]
        fraction = tp/(tp+fp)
        return "Below 70%" if fraction < .7 else "70–79%" if fraction < .8 else "80% or above"
    if exp == "support_threshold":
        return "Retained" if int(control) <= 25 else "Filtered out"
    raise ValueError(f"Unknown experiment: {exp}")


def is_correct(question, answer):
    raw = answer_value(answer)
    if question["kind"] == "short":
        return normalize(raw) in {normalize(x) for x in question["correct"]}
    control = answer.get("control") if isinstance(answer, dict) else None
    return raw == expected_for(question, control)


def summary(questions, answers):
    attempted = sum(bool(answers.get(q["id"])) for q in questions)
    score = sum(is_correct(q, answers.get(q["id"], "")) for q in questions if answers.get(q["id"]))
    return attempted, score


def fetch_attempt(student_id):
    data = db.table(TABLE).select("*").eq("student_id", student_id).limit(1).execute().data
    return data[0] if data else None


def persist(answers=None, interactions=None, finished=False):
    """One-row atomic field update. Never overwrite server state after final submission."""
    a = st.session_state.attempt
    payload = {"answers": answers if answers is not None else a["answers"],
               "interaction_states": interactions if interactions is not None else (a.get("interaction_states") or {}),
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
    q = next(q for q in st.session_state.questions if q["id"] == qid)
    control = None
    if q.get("experiment"):
        control_key = f"control_{a['student_id']}_{qid}"
        control = st.session_state.get(control_key, EXPERIMENT_CONTROLS[q["experiment"]][4])
    updated[qid] = {"value": value, "control": control} if q.get("experiment") else value
    try:
        if q.get("experiment"):
            states = dict(a.get("interaction_states") or {})
            states[qid] = control
            persist(answers=updated, interactions=states)
        else:
            persist(answers=updated)
        st.session_state.save_error = ""
    except Exception:
        st.session_state.save_error = "Autosave failed. Please retry this answer before leaving the page."


def save_control(qid, key):
    a = st.session_state.attempt
    if a.get("submitted_at") or (a.get("answers") or {}).get(qid):
        return
    states = dict(a.get("interaction_states") or {})
    states[qid] = st.session_state[key]
    try:
        persist(interactions=states)
        st.session_state.save_error = ""
    except Exception:
        st.session_state.save_error = "Experiment setting was not saved. Please try adjusting the control again."


def render_tree(cutoff, hours):
    components.html(f"""
    <div style="padding:8px;background:#f5f9fb;border-radius:12px;max-width:620px;box-sizing:border-box">
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
    </svg></div>""", height=270, scrolling=False)


def visual_card(svg_body, footer=""):
    """A self-contained SVG; its drawing changes when the Streamlit slider changes."""
    components.html(f"""
    <div style="max-width:750px;padding:14px 18px;border:1px solid #d9e8e8;
                border-radius:16px;background:#f7fbfc;box-sizing:border-box">
      <svg viewBox="0 0 640 205" role="img" aria-label="Interactive experiment result"
           xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto">
      {svg_body}
      </svg>
      <div style="color:#476579;font:13px system-ui;margin-top:4px">{escape(str(footer))}</div>
    </div>""", height=300, scrolling=False)


def visual_text(x, y, value, size=16, color="#174659", anchor="middle"):
    return f'<text x="{x}" y="{y}" text-anchor="{anchor}" fill="{color}" font-size="{size}" font-family="Arial">{escape(str(value))}</text>'


def render_experiment(question, control):
    exp = question["experiment"]
    prm = question.get("params") or {}
    if exp == "tree_cutoff":
        st.caption(f"Student profile: attendance {prm['attendance']}%, study hours {prm['hours']}/week")
        render_tree(int(control), prm["hours_min"])
        return
    if exp == "knn_k":
        body = visual_text(320,25,"Nearest neighbours, ordered by increasing distance",17)
        for i,label in enumerate(prm["neighbours"]):
            x = 58 + i*85
            fill = "#168d80" if label == "P" else "#d87c64"
            opacity = "1" if i < int(control) else ".22"
            body += f'<circle cx="{x}" cy="100" r="28" fill="{fill}" opacity="{opacity}"/>'
            body += visual_text(x,107,label,21,"white" if i < int(control) else "#4a6673")
            body += visual_text(x,157,f"#{i+1}",13)
        visual_card(body,"Green = Pass, orange = Fail. Only the first k highlighted circles vote.")
        return
    if exp == "neuron_weight":
        z = 2*int(control)-1
        output = max(0,z)
        body = '<g stroke="#8cbac0" stroke-width="3"><line x1="130" y1="60" x2="310" y2="100"/><line x1="130" y1="150" x2="310" y2="100"/><line x1="330" y1="100" x2="510" y2="100"/></g>'
        for x,y,c in [(115,60,"#407f9c"),(115,150,"#407f9c"),(320,100,"#116b75"),(525,100,"#159a81" if output>0 else "#a2acb3")]:
            body += f'<circle cx="{x}" cy="{y}" r="30" fill="{c}"/>'
        body += visual_text(115,65,"x1=2",14,"white")+visual_text(115,155,"x2=1",14,"white")
        body += visual_text(320,105,"SUM",14,"white")+visual_text(525,105,f"{output}",19,"white")
        body += visual_text(220,58,f"w1={control}",15)+visual_text(218,150,"w2=1",15)
        body += visual_text(320,170,f"z = {z}, bias = -2",16)
        visual_card(body,"ReLU activation: output = max(0, weighted sum + bias). Change w1 to observe the neuron.")
        return
    if exp == "dropout":
        disabled = int(control)
        body = visual_text(320,26,"8 hidden neurons - training-time dropout",18)
        for i in range(8):
            x=92+(i%4)*150; y=76+(i//4)*72
            fill="#adb7bf" if i<disabled else "#168d80"
            body += f'<circle cx="{x}" cy="{y}" r="24" fill="{fill}"/>'
            body += visual_text(x,y+5,"OFF" if i<disabled else "ON",12,"white")
        visual_card(body,f"Dropout rate = {disabled}/8 = {disabled/8:.0%}; active units = {8-disabled}. Mask is illustrative, not a random draw.")
        return
    if exp == "bagging_vote":
        votes=prm["votes"][int(control)-1]
        draws={1:["A","C","C","B","D"],2:["D","D","A","B","A"],3:["B","C","A","C","B"]}[int(control)]
        body=visual_text(320,26,"Bootstrap draws (WITH replacement): " + ", ".join(draws),17)
        for i,v in enumerate(votes):
            x=135+i*185; color="#159284" if v=="P" else "#d77861"
            body+=f'<rect x="{x-65}" y="66" width="130" height="75" rx="15" fill="{color}"/>'
            body+=visual_text(x,90,f"TREE {i+1}",14,"white")+visual_text(x,118,"Pass" if v=="P" else "Fail",20,"white")
        visual_card(body,"Inspect all three learners and select the ensemble's majority-vote output.")
        return
    if exp == "boosting_weight":
        weights=[int(control),2,2]
        body=visual_text(320,24,"Sequential learner votes - weighted combination",17)
        for i,(w,v) in enumerate(zip(weights,["Pass","Fail","Fail"])):
            x=100+i*200
            body+=f'<rect x="{x-55}" y="{175-w*17}" width="110" height="{w*17}" rx="7" fill="{("#168d80" if v=="Pass" else "#d77861")}"/>'
            body+=visual_text(x,195,f"{v}: {w}",15)
        visual_card(body,"Learner A predicts Pass; B and C predict Fail. Compare total vote weight for each class.")
        return
    if exp == "logistic_threshold":
        p=prm["prob"]; xprob=70+p*500; xt=70+float(control)*500
        body='<rect x="70" y="80" width="500" height="20" rx="10" fill="#d8e5eb"/>'
        body+=f'<rect x="70" y="80" width="{500*p}" height="20" rx="10" fill="#168d80"/>'
        body+=f'<line x1="{xt}" x2="{xt}" y1="55" y2="135" stroke="#d87957" stroke-width="4"/>'
        body+=f'<circle cx="{xprob}" cy="90" r="11" fill="#174659" stroke="white" stroke-width="2"/>'
        body+=visual_text(xprob,45,f"P(Pass)={p:.2f}",17)+visual_text(xt,159,f"Threshold={float(control):.1f}",15)
        visual_card(body,"Blue dot = predicted probability; orange line = your chosen decision threshold.")
        return
    if exp == "regression_slope":
        b=float(control)
        body='<path d="M60 170H580M60 170V20" stroke="#6d8c9a" stroke-width="2" fill="none"/>'
        points=[]
        for attendance in range(0,101,10):
            yval=15+b*attendance
            points.append((60+attendance*5,170-yval*1.45))
        path=" ".join(("M" if i==0 else "L")+f"{x:.1f} {y:.1f}" for i,(x,y) in enumerate(points))
        body+=f'<path d="{path}" fill="none" stroke="#128e86" stroke-width="4"/>'
        py=170-(15+b*60)*1.45
        body+=f'<circle cx="360" cy="{py:.1f}" r="8" fill="#d48a48"/>'
        body+=visual_text(320,20,f"mark = 15 + {b:.1f} x attendance",18)
        body+=visual_text(345,195,"Attendance (%)",14)
        visual_card(body,"The orange point shows the prediction at 60% attendance; adjust slope to rotate the fitted line.")
        return
    if exp == "cv_fold":
        k=int(control); body=visual_text(320,25,f"60 records split into {k} equal folds",17)
        for i in range(k):
            width=540/k-7; x=50+i*(540/k)
            body+=f'<rect x="{x:.1f}" y="74" width="{width:.1f}" height="74" rx="6" fill="{("#d77861" if i==0 else "#2e91a1")}"/>'
            body+=visual_text(x+width/2,102,f"F{i+1}",15,"white")
        visual_card(body,"Orange fold = validation; teal folds = training. Each fold rotates into validation once.")
        return
    if exp == "gradient_lr":
        lr=float(control); wnew=4-4*lr
        body='<path d="M45 175H605M45 175V12" stroke="#718b9a" stroke-width="2" fill="none"/>'
        pts=[]
        for i in range(61):
            w=-1+i*.1
            pts.append((55+(w+1)*84,175-(w-2)**2*13))
        body+='<path d="' + ' '.join(('M' if i==0 else 'L')+f'{x:.1f} {y:.1f}' for i,(x,y) in enumerate(pts)) + '" fill="none" stroke="#168d80" stroke-width="3"/>'
        for w,col,label in [(4,"#3b6c9f","Start"),(wnew,"#d77a5e","After step")]:
            x=55+(w+1)*84; y=175-(w-2)**2*13
            body+=f'<circle cx="{x:.1f}" cy="{y:.1f}" r="9" fill="{col}"/>'
            body+=visual_text(x,max(15,y-15),label,13)
        visual_card(body,f"Initial loss = 4. New loss = {(wnew-2)**2:.2f}. Learning rate = {lr:.1f}.")
        return
    if exp == "precision_threshold":
        tp,fp={1:(9,6),2:(7,2),3:(4,1)}[int(control)]
        body=visual_text(320,27,"Positive predictions at this threshold",19)
        for i,(name,count,color) in enumerate([("True Positive",tp,"#168d80"),("False Positive",fp,"#d77861")]):
            x=60+i*285
            body+=f'<rect x="{x}" y="70" width="250" height="90" rx="14" fill="{color}"/>'
            body+=visual_text(x+125,103,name,16,"white")+visual_text(x+125,141,count,27,"white")
        visual_card(body,"Compute precision from TP/(TP+FP). Higher thresholds may change both counts.")
        return
    if exp == "support_threshold":
        min_sup=int(control)
        body=visual_text(320,30,"Observed rule support: 25%",19)
        body+='<rect x="70" y="85" width="500" height="28" rx="10" fill="#d9e6e8"/>'
        body+='<rect x="70" y="85" width="125" height="28" rx="10" fill="#168d80"/>'
        x=70+min_sup*5
        body+=f'<line x1="{x}" y1="60" x2="{x}" y2="140" stroke="#d77861" stroke-width="4"/>'
        body+=visual_text(x,159,f"Minimum {min_sup}%",15)
        visual_card(body,"Teal bar = observed support. Orange line = minimum support filter.")


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
    total = len(questions)
    story = [p("MAC Week 5 — Model Detective Tutorial", "W5Title")]
    identity = [["Student", attempt["student_name"]], ["Student ID", attempt["student_id"]],
                ["Duration", "90 minutes"], ["Submitted (UTC)", str(attempt.get("submitted_at") or "Not submitted")],
                ["Result", f"{score}/{total} ({score/total*100:.1f}%) | {completed}/{total} answered"]]
    t = Table([[p(a), p(b)] for a,b in identity], colWidths=[38*mm, 133*mm])
    t.setStyle(TableStyle([("BACKGROUND", (0,0),(-1,-1), colors.HexColor("#EDF5F6")),
                          ("VALIGN", (0,0),(-1,-1), "TOP"), ("LEFTPADDING",(0,0),(-1,-1),10),
                          ("TOPPADDING",(0,0),(-1,-1),7), ("BOTTOMPADDING",(0,0),(-1,-1),7)]))
    story.extend([t, Spacer(1, 10), p("Automatically marked report. Review the feedback below.")])
    for q in questions:
        answer = answers.get(q["id"], "")
        correct = is_correct(q, answer) if answer else False
        answer_text = str(answer_value(answer)) if answer else "Not answered"
        control = answer.get("control") if isinstance(answer, dict) else None
        expected = (expected_for(q, control) if q["kind"] == "select"
                    else ", ".join(q["correct"]))
        block = [p(f"{q['id']} — {q['topic']}: {q['prompt']}", "W5Question"),
                 p(f"Your answer: {answer_text}  |  {'Correct' if correct else 'Incorrect / missing'}  |  Mark: {int(correct)}/1"),
                 p(f"Correct answer: {expected}"), p(f"Feedback: {q['explanation']}")]
        if q.get("experiment"):
            label = EXPERIMENT_CONTROLS[q["experiment"]][0]
            block.insert(1, p(f"Experiment setting: {label} = {control if control is not None else 'Not recorded'}"))
        story.append(KeepTogether(block))
    story.append(Spacer(1, 12))
    story.append(p("Submission: Upload this PDF to myTIMeS as instructed by your lecturer."))
    doc.build(story)
    return out.getvalue()


st.title("🧠 The Model Detective Challenge")
st.caption("MAC | Week 5 — Classification, Prediction and Evaluation")
st.write("Explore models, adjust the visual experiments, and answer 60 challenges. Your selections and experiment settings are saved as you go.")

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
                        payload = dict(student_id=sid, student_name=name, answers={},
                                       interaction_states={}, started_at=utc_now().isoformat())
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
total = len(qs)
left, mid, right = st.columns(3)
left.metric("Answered", f"{answered}/{total}")
mid.metric("Current marks", f"{score}/{total}")
right.metric("Time remaining", "Submitted" if attempt.get("submitted_at") else str(remaining).split(".")[0])
st.progress(answered/total, text=f"Progress: {answered} of {total} answered")

if st.session_state.save_error:
    st.error(st.session_state.save_error)

if attempt.get("submitted_at"):
    st.success("Your tutorial is submitted. Download your feedback report and upload it to myTIMeS.")
    st.metric("Final result", f"{score}/{total} ({score/total:.0%})")
    pdf = make_pdf(attempt, qs)
    st.download_button("⬇ Download My Marked PDF", data=pdf,
                       file_name=f"MAC_Week5_{attempt['student_id']}_Feedback.pdf",
                       mime="application/pdf", type="primary", use_container_width=True)
    with st.expander("See my answers and feedback"):
        for q in qs:
            chosen = answers.get(q["id"], "Not answered")
            st.write(f"**{q['id']} ({q['topic']}):** {answer_value(chosen)} — {'✅' if is_correct(q, chosen) else '❌'}")
            st.caption(q["explanation"])
    st.stop()

st.warning("Once you submit or your 90 minutes end, answers cannot be changed. Keep your Student ID and name consistent if you reconnect.")

index = st.session_state.current
q = qs[index]
st.divider()
st.subheader(f"Question {index + 1} of {total}")
st.caption(q["topic"] + (" • Fix the wrong model" if q["kind"] == "short" else " • Select the best answer"))
st.caption(f"Assigned student: {attempt['student_name']} | ID: {attempt['student_id']}")
st.write(q["prompt"])
if q["diagram"]:
    render_tree(*q["diagram"])

if q.get("experiment"):
    spec = EXPERIMENT_CONTROLS[q["experiment"]]
    control_key = f"control_{attempt['student_id']}_{q['id']}"
    saved_answer = answers.get(q["id"])
    if control_key not in st.session_state:
        st.session_state[control_key] = (
            saved_answer.get("control") if isinstance(saved_answer, dict)
            else (attempt.get("interaction_states") or {}).get(q["id"], spec[4]))
    st.slider(spec[0], min_value=spec[1], max_value=spec[2], step=spec[3],
              key=control_key, disabled=bool(saved_answer),
              on_change=save_control, args=(q["id"], control_key))
    render_experiment(q, st.session_state[control_key])
    st.caption("Change the control and inspect the diagram BEFORE choosing an answer. The setting is saved with your response.")

widget = f"week5_{attempt['student_id']}_{q['id']}"
if widget not in st.session_state:
    st.session_state[widget] = answer_value(answers.get(q["id"], "")) or ("Choose an answer" if q["kind"] == "select" else "")

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
    if st.button("Next →", disabled=index == total-1, use_container_width=True):
        st.session_state.current += 1
        st.rerun()

if answered == total:
    st.success(f"All {total} questions are answered. You may now submit your tutorial.")
    if st.button("Submit Final Answers and Generate PDF", type="primary", use_container_width=True):
        try:
            persist(finished=True)
            st.rerun()
        except Exception:
            st.error("Final submission was not saved. Please try again.")
else:
    st.caption(f"Final submission unlocks after all {total} questions are answered ({total-answered} remaining).")
