import math
from collections import Counter
from urllib.parse import urlparse

import joblib
import pandas as pd
import streamlit as st


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Phishing URL Classifier",
    page_icon="🛡️",
    layout="centered"
)


# ============================================================
# MODEL
# ============================================================

MODEL_PATH = "phishing_url_random_forest.pkl"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    model = load_model()
except Exception as e:
    st.error("Unable to load the phishing detection model.")
    st.code(str(e))
    st.stop()


# ============================================================
# FEATURE FUNCTIONS
# ============================================================

def calculate_entropy(text):
    """Calculate Shannon entropy of a string."""

    if not text:
        return 0.0

    counts = Counter(text)
    length = len(text)

    entropy = 0.0

    for count in counts.values():
        probability = count / length
        entropy -= probability * math.log2(probability)

    return entropy


def calculate_suspicious_words(url):
    """
    Approximation of the SUSPICIOUS_WORDS feature.

    IMPORTANT:
    The original SQL/rule used to generate SUSPICIOUS_WORDS
    in PHISHING_CLEAN_DATA was not available.

    Therefore this function uses common phishing-related
    terms as a deployment approximation.

    If the original feature-generation SQL is recovered,
    replace this function with the exact original logic.
    """

    suspicious_terms = [
        "login",
        "log-in",
        "signin",
        "sign-in",
        "verify",
        "verification",
        "secure",
        "security",
        "account",
        "update",
        "confirm",
        "confirmation",
        "password",
        "passwd",
        "credential",
        "credentials",
        "wallet",
        "recover",
        "recovery",
        "support",
        "unlock",
        "authenticate",
        "authentication",
        "authorize",
        "authorization",
        "payment",
        "billing",
        "invoice",
        "bank",
        "banking",
        "paypal",
        "microsoft",
        "office365",
        "outlook",
        "apple",
        "icloud",
        "google",
        "gmail",
        "facebook",
        "instagram",
        "linkedin",
        "amazon",
        "netflix",
        "roblox",
        "ledger",
        "metamask",
        "kucoin",
        "chase",
        "visa",
        "mastercard",
        "3dsecure",
    ]

    url_lower = url.lower()

    for term in suspicious_terms:
        if term in url_lower:
            return 1

    return 0


def extract_features(url):
    """
    Extract the 11 raw features expected by the trained pipeline.
    """

    parsed_url = urlparse(url)

    hostname = parsed_url.netloc
    path = parsed_url.path
    query = parsed_url.query

    # Remove port if present
    hostname_only = hostname.split(":")[0]

    # --------------------------------------------------------
    # TLD
    # --------------------------------------------------------

    if "." in hostname_only:
        tld = hostname_only.split(".")[-1].lower()
    else:
        tld = "unknown"

    # --------------------------------------------------------
    # Number of subdirectories
    # --------------------------------------------------------

    path_parts = [
        part
        for part in path.split("/")
        if part
    ]

    num_subdirs = max(len(path_parts) - 1, 0)

    # --------------------------------------------------------
    # Basic URL features
    # --------------------------------------------------------

    num_dots = url.count(".")

    url_length = len(url)

    entropy = calculate_entropy(url)

    digits_count = sum(
        char.isdigit()
        for char in url
    )

    # --------------------------------------------------------
    # IP address detection
    # --------------------------------------------------------

    hostname_parts = hostname_only.split(".")

    has_ip = int(
        len(hostname_parts) == 4
        and all(
            part.isdigit()
            for part in hostname_parts
        )
    )

    # --------------------------------------------------------
    # HTTPS
    # --------------------------------------------------------

    has_https = int(
        parsed_url.scheme.lower() == "https"
    )

    # --------------------------------------------------------
    # Special characters
    # --------------------------------------------------------

    special_char_count = sum(
        not char.isalnum()
        for char in url
    )

    # --------------------------------------------------------
    # URL parameters
    # --------------------------------------------------------

    num_params = (
        len(query.split("&"))
        if query
        else 0
    )

    # --------------------------------------------------------
    # Suspicious words
    # --------------------------------------------------------

    suspicious_words = calculate_suspicious_words(url)

    # --------------------------------------------------------
    # EXACT FEATURE NAMES EXPECTED BY MODEL
    # --------------------------------------------------------

    features = pd.DataFrame([{
        "URL_LENGTH": url_length,
        "NUM_DOTS": num_dots,
        "HAS_HTTPS": has_https,
        "HAS_IP": has_ip,
        "NUM_SUBDIRS": num_subdirs,
        "NUM_PARAMS": num_params,
        "SUSPICIOUS_WORDS": suspicious_words,
        "TLD": tld,
        "SPECIAL_CHAR_COUNT": special_char_count,
        "DIGITS_COUNT": digits_count,
        "ENTROPY": entropy
    }])

    return features


# ============================================================
# URL VALIDATION
# ============================================================

def is_valid_url(url):
    """Basic URL validation."""

    try:
        parsed = urlparse(url)

        return (
            parsed.scheme.lower() in ["http", "https"]
            and bool(parsed.netloc)
        )

    except Exception:
        return False


# ============================================================
# HEADER
# ============================================================

st.title("🛡️ Phishing URL Classifier")

st.write(
    "Enter a URL below to classify it as "
    "legitimate or potentially phishing."
)

st.info(
    "This classifier is a machine-learning model trained "
    "on the project's phishing URL dataset. Results should "
    "be treated as model predictions, not as a guarantee of safety."
)


# ============================================================
# INPUT
# ============================================================

url = st.text_input(
    "Enter URL",
    placeholder="https://example.com"
)


# ============================================================
# BUTTON
# ============================================================

check_button = st.button(
    "🔍 Check URL",
    use_container_width=True,
    type="primary"
)


# ============================================================
# PREDICTION
# ============================================================

if check_button:

    if not url.strip():

        st.warning(
            "Please enter a URL."
        )

        st.stop()

    clean_url = url.strip()

    if not is_valid_url(clean_url):

        st.error(
            "Please enter a valid HTTP or HTTPS URL."
        )

        st.stop()

    try:

        # ----------------------------------------------------
        # Extract raw features
        # ----------------------------------------------------

        features = extract_features(clean_url)

        # ----------------------------------------------------
        # Model prediction
        # ----------------------------------------------------

        prediction = model.predict(features)[0]

        probabilities = model.predict_proba(features)[0]

        legitimate_probability = probabilities[0]

        phishing_probability = probabilities[1]

        # ----------------------------------------------------
        # Result
        # ----------------------------------------------------

        st.subheader("Prediction")

        if prediction == 1:

            st.error(
                "🚨 PHISHING URL"
            )

            st.metric(
                "Phishing Probability",
                f"{phishing_probability:.2%}"
            )

        else:

            st.success(
                "✅ LEGITIMATE URL"
            )

            st.metric(
                "Legitimate Probability",
                f"{legitimate_probability:.2%}"
            )

        # ----------------------------------------------------
        # Probability breakdown
        # ----------------------------------------------------

        st.subheader("Prediction Probabilities")

        probability_df = pd.DataFrame({
            "Classification": [
                "Legitimate",
                "Phishing"
            ],
            "Probability": [
                legitimate_probability,
                phishing_probability
            ]
        })

        probability_df["Probability"] = (
            probability_df["Probability"] * 100
        ).round(2)

        st.dataframe(
            probability_df,
            use_container_width=True,
            hide_index=True
        )

        # ----------------------------------------------------
        # Extracted features
        # ----------------------------------------------------

        st.subheader("Extracted Features")

        st.dataframe(
            features,
            use_container_width=True,
            hide_index=True
        )

        # ----------------------------------------------------
        # Feature interpretation
        # ----------------------------------------------------

        st.subheader("URL Analysis")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.write("**URL Length**")
            st.write(features["URL_LENGTH"].iloc[0])

        with col2:
            st.write("**Number of Dots**")
            st.write(features["NUM_DOTS"].iloc[0])

        with col3:
            st.write("**Subdirectories**")
            st.write(features["NUM_SUBDIRS"].iloc[0])

        col4, col5, col6 = st.columns(3)

        with col4:
            st.write("**HTTPS**")
            st.write(
                "Yes"
                if features["HAS_HTTPS"].iloc[0]
                else "No"
            )

        with col5:
            st.write("**IP Address**")
            st.write(
                "Yes"
                if features["HAS_IP"].iloc[0]
                else "No"
            )

        with col6:
            st.write("**Suspicious Words**")
            st.write(
                "Detected"
                if features["SUSPICIOUS_WORDS"].iloc[0]
                else "Not detected"
            )

    except Exception as e:

        st.error(
            "Unable to process this URL."
        )

        st.exception(e)


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Phishing URL Classification • "
    "Random Forest model"
)