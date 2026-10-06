"""Project settings: file paths, constants and the word lists used by the
feature extractor. Keeping them here means no paths are hard-coded elsewhere.
COS30049 - Session 4 Group 5 (PhishGuard)."""

from pathlib import Path

# Paths are built relative to the project root so it runs on any machine.
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

RAW_DATASET = RAW_DIR / "malicious_phish.csv"
PHIUSIIL_DATASET = RAW_DIR / "phiusiil_urls.csv.gz"   # extra dataset (PhiUSIIL)
PROCESSED_DATASET = PROCESSED_DIR / "final_dataset.csv"

FINAL_MODEL_FILE = MODELS_DIR / "final_classifier.pkl"
FEATURE_PIPELINE_FILE = MODELS_DIR / "feature_pipeline.pkl"

RANDOM_SEED = 42
TEST_SIZE = 0.20

# Binary target: phishing = 1, everything else = 0.
# Note: 0 means "not phishing", not "safe" (it still includes malware/defacement).
POSITIVE_CLASS = "phishing"
TARGET_COLUMN = "label"
RAW_LABEL_COLUMN = "type"
CLASS_NAMES = {0: "not phishing", 1: "phishing"}

# Words that show up a lot in phishing URLs (from Sahingoz et al. 2019 and
# Mohammad et al. 2014).
SUSPICIOUS_KEYWORDS = [
    "login", "signin", "sign-in", "verify", "verification", "account",
    "secure", "security", "update", "confirm", "password", "banking", "bank",
    "ebayisapi", "webscr", "paypal", "wallet", "recover", "unlock", "suspend",
    "billing", "invoice", "payment", "credential", "authenticate", "support",
    "free", "bonus", "gift", "prize", "winner", "alert", "limited", "urgent",
]

# URL shorteners hide the real destination.
SHORTENER_SERVICES = [
    "bit.ly", "goo.gl", "tinyurl.com", "t.co", "ow.ly", "is.gd", "buff.ly",
    "adf.ly", "bit.do", "cutt.ly", "shorte.st", "rebrand.ly", "rb.gy",
    "tiny.cc", "soo.gd", "clck.ru", "t.ly", "v.gd", "x.co", "lnkd.in",
]

# TLDs that are cheap/free and commonly abused.
SUSPICIOUS_TLDS = [
    "zip", "review", "country", "kim", "cricket", "science", "work", "party",
    "gq", "link", "tk", "ml", "ga", "cf", "xyz", "top", "club", "online",
    "support", "click", "loan", "download", "racing", "win", "bid", "stream",
    "men", "date", "faith", "accountant", "realtor", "mom", "rest",
]

# Risky file endings (downloads / fake documents).
SUSPICIOUS_EXTENSIONS = [
    ".exe", ".zip", ".rar", ".scr", ".js", ".jar", ".msi", ".bat", ".cmd",
    ".apk", ".dll", ".vbs", ".ps1", ".7z", ".gz", ".iso",
]

# Two-part domain endings, used to count subdomains correctly without internet.
MULTI_LABEL_SUFFIXES = {
    "co.uk", "org.uk", "gov.uk", "ac.uk", "me.uk", "ltd.uk", "plc.uk",
    "com.au", "net.au", "org.au", "gov.au", "edu.au", "id.au",
    "co.nz", "net.nz", "org.nz", "govt.nz",
    "com.br", "net.br", "org.br", "gov.br",
    "co.in", "net.in", "org.in", "gen.in", "firm.in", "ind.in",
    "com.cn", "net.cn", "org.cn", "gov.cn",
    "co.jp", "or.jp", "ne.jp", "ac.jp", "go.jp",
    "com.sg", "edu.sg", "gov.sg", "org.sg",
    "co.za", "org.za", "gov.za",
    "com.mx", "com.tr", "com.ar", "com.hk", "com.tw", "com.my", "com.ua",
    "co.id", "co.kr", "or.kr",
}
