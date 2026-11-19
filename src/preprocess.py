"""Area B - classical NLP preprocessing.

* regex text normalization (Turkish-aware lowercasing, URLs, e-mails, phone numbers,
  repeated letters / punctuation)
* regex extraction of product model codes (e.g. "DW-6534", "rf 4410" -> "RF-4410")
* minimum-edit-distance (Levenshtein) spelling correction of product names against a
  TR + EN product lexicon
* a simple regex tokenizer shared by the classical models
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# ----------------------------------------------------------------------------- regexes
URL_RE = re.compile(r"(https?://\S+|www\.\S+)")
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")
PHONE_RE = re.compile(r"(\+?\d[\d\s]{8,}\d)")
# model codes: 2-4 letters, optional '-' or space, 3-5 digits, optional letter suffix
# (uppercase letters required when separated by a space, to avoid matching "is 2024")
MODEL_CODE_RE = re.compile(
    r"\b(?:(?P<a>[A-Za-z]{2,4})-(?P<b>\d{3,5}[A-Za-z]?)"
    r"|(?P<c>[A-Z]{2,4}) ?(?P<d>\d{3,5}[A-Z]?)"
    r"|(?P<e>[a-z]{2,3})(?P<f>\d{3,5}))\b"
)
REPEAT_CHAR_RE = re.compile(r"([^\W\d_])\1{2,}")     # "çalışmıyorrrr" -> "çalışmıyor"
REPEAT_PUNCT_RE = re.compile(r"([!?.,])\1+")    # "!!!" -> "!"
TOKEN_RE = re.compile(r"[\w']+|[^\w\s]", re.UNICODE)
WORD_RE = re.compile(r"\w+", re.UNICODE)

MODEL_PLACEHOLDER = "modelcode"


def turkish_lower(text: str) -> str:
    """Lowercase that keeps Turkish dotted capital İ -> i (str.lower gives 'i̇')."""
    return text.replace("İ", "i").lower()


def extract_model_codes(text: str) -> list[str]:
    codes = []
    for m in MODEL_CODE_RE.finditer(text):
        letters = m.group("a") or m.group("c") or m.group("e")
        digits = m.group("b") or m.group("d") or m.group("f")
        codes.append(f"{letters.upper()}-{digits.upper()}")
    return codes


def tokenize(text: str) -> list[str]:
    """Word tokens only (punctuation dropped) - used by tf-idf / word2vec."""
    return WORD_RE.findall(text)


# ------------------------------------------------------------------- edit distance
def edit_distance(a: str, b: str) -> int:
    """Minimum edit distance (Levenshtein; insert = delete = substitute = 1), DP table."""
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1,                 # deletion
                         cur[j - 1] + 1,              # insertion
                         prev[j - 1] + (ca != cb))    # substitution
        prev = cur
    return prev[-1]


TR_FOLD = str.maketrans("şığüöçâî", "siguocai")

PRODUCT_LEXICON = sorted(set("""
dishwasher dishwashers refrigerator fridge freezer washer washing machine dryer tumble oven
microwave cooktop hob hood vacuum cleaner kettle toaster boiler conditioner heater compressor
filter thermostat appliance appliances
bulaşık makine makinesi makinem makinemin makinesinin makinesine buzdolabı buzdolabım buzdolabımın
buzdolabının çamaşır kurutma fırın fırınım fırının mikrodalga ocak ocağım aspiratör davlumbaz
süpürge süpürgem klima klimam kombi kombim dondurucu ankastre ütü su ısıtıcısı garanti servis
teknisyen fatura iade yedek parça
""".split()))


@dataclass
class SpellCorrector:
    """Corrects out-of-vocabulary tokens to the closest product-lexicon word."""
    lexicon: list[str] = field(default_factory=lambda: list(PRODUCT_LEXICON))
    vocab: set[str] = field(default_factory=set)   # known words (from training data)

    def max_dist(self, word: str) -> int:
        return 1 if len(word) < 9 else 2

    @staticmethod
    def fold(word: str) -> str:
        """Turkish letters -> ASCII, so 'bulasik' and 'bulaşık' are distance 0 after folding."""
        return word.translate(TR_FOLD)

    def correct_token(self, tok: str) -> str:
        if len(tok) < 4 or not tok.isalpha() or tok in self.vocab or tok in self.lexicon:
            return tok
        best, best_d = tok, self.max_dist(tok) + 1
        for w in self.lexicon:
            if abs(len(w) - len(tok)) > 2:
                continue
            d = edit_distance(self.fold(tok), self.fold(w))
            if d < best_d:
                best, best_d = w, d
        return best if best_d <= self.max_dist(tok) else tok

    def correct(self, text: str) -> tuple[str, list[tuple[str, str]]]:
        changes = []

        def repl(m):
            tok = m.group(0)
            new = self.correct_token(tok)
            if new != tok:
                changes.append((tok, new))
            return new

        return WORD_RE.sub(repl, text), changes


@dataclass
class Normalized:
    original: str
    text: str                       # what the models see
    model_codes: list[str]
    corrections: list[tuple[str, str]]


class Normalizer:
    def __init__(self, corrector: SpellCorrector | None = None):
        self.corrector = corrector or SpellCorrector()

    def __call__(self, raw: str) -> Normalized:
        codes = extract_model_codes(raw)
        t = MODEL_CODE_RE.sub(f" {MODEL_PLACEHOLDER} ", raw)
        t = URL_RE.sub(" url ", t)
        t = EMAIL_RE.sub(" email ", t)
        t = PHONE_RE.sub(" phone ", t)
        t = turkish_lower(t)
        t = REPEAT_CHAR_RE.sub(r"\1", t)
        t = REPEAT_PUNCT_RE.sub(r"\1", t)
        t = " ".join(TOKEN_RE.findall(t))
        t, changes = self.corrector.correct(t)
        return Normalized(raw, t, codes, changes)

    def texts(self, raws):
        return [self(r).text for r in raws]
