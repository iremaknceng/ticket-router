"""Area B - ELIZA-style rule-based replies for greetings and small talk (TR + EN).

Each rule is a regex with a list of reply templates. Like Weizenbaum's ELIZA, captured
groups are "reflected" (my -> your, benim -> sizin) and inserted into the reply.
A message counts as *pure* small talk only if almost nothing is left after removing
the matched phrase; "merhaba, bulaşık makinem su boşaltmıyor" is routed, not chatted.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass

REFLECTIONS = {
    "i": "you", "me": "you", "my": "your", "am": "are", "i'm": "you're", "mine": "yours",
    "benim": "sizin", "ben": "siz", "bana": "size",
}


def reflect(fragment: str) -> str:
    return " ".join(REFLECTIONS.get(w, w) for w in fragment.lower().split())


@dataclass
class Rule:
    name: str
    lang: str
    pattern: re.Pattern
    replies: list[str]


def _r(name, lang, pattern, replies):
    return Rule(name, lang, re.compile(pattern, re.IGNORECASE | re.UNICODE), replies)


RULES = [
    # --- Turkish
    _r("greeting", "tr", r"\b(merhaba|selam(lar)?|günaydın|iyi (günler|akşamlar)|hey)\b",
       ["Merhaba! Size nasıl yardımcı olabilirim? Lütfen ürününüzü ve sorununuzu yazın.",
        "Selam! Hangi cihazınızla ilgili destek istiyorsunuz?"]),
    _r("how_are_you", "tr", r"\b(nasılsın(ız)?|naber|ne haber)\b",
       ["Teşekkürler, iyiyim! Cihazınızla ilgili nasıl yardımcı olabilirim?"]),
    _r("thanks", "tr", r"\b(teşekkür(ler| ederim)?|sağ ?ol(un)?|eyvallah)\b",
       ["Rica ederim! Başka bir sorunuz olursa yazabilirsiniz."]),
    _r("goodbye", "tr", r"\b(görüşürüz|hoşça ?kal(ın)?|iyi geceler|bay bay)\b",
       ["Hoşça kalın, iyi günler dileriz!"]),
    _r("bot", "tr", r"\b(sen kimsin|bot musun|robot musun|adın ne)\b",
       ["Ben otomatik destek asistanıyım. Mesajınızı doğru ekibe yönlendiriyorum."]),
    _r("feeling", "tr", r"\b(çok )?(sinirliyim|kızgınım|üzgünüm|bıktım)\b",
       ["Bunu duyduğuma üzüldüm. Sorunu biraz anlatırsanız hemen ilgili ekibe iletirim."]),
    # --- English
    _r("greeting", "en", r"\b(hi|hello|hey|good (morning|afternoon|evening))\b",
       ["Hello! How can I help you today? Please tell me your product and the problem.",
        "Hi there! Which appliance do you need help with?"]),
    _r("how_are_you", "en", r"\b(how are you|how's it going|what's up)\b",
       ["I'm fine, thanks! How can I help with your appliance?"]),
    _r("thanks", "en", r"\b(thanks|thank you|thx|cheers)\b",
       ["You're welcome! Let me know if there is anything else."]),
    _r("goodbye", "en", r"\b(bye|goodbye|see you|good night)\b",
       ["Goodbye, have a nice day!"]),
    _r("bot", "en", r"\b(who are you|are you a (bot|robot|human)|what is your name)\b",
       ["I'm the automatic support assistant - I route your message to the right team."]),
    _r("feeling", "en", r"\bi(?: am|'m) (?:so |very |really )?(angry|upset|frustrated|annoyed|sad)\b",
       ["I'm sorry you are {0}. Tell me what happened and I'll pass it to the right team."]),
    _r("i_need", "en", r"^i need (?:a |an |some )?(hug|friend|someone to talk to)\b",
       ["Why do you need {0}?I'm a support bot, but I'm happy to help with your appliances."]),
]

FILLER = re.compile(r"\b(please|lütfen|bir|a|the|there|you|too|very|much|so|çok|da|de|ya|ve|and|again)\b|[^\w\s]",
                    re.IGNORECASE)


@dataclass
class ElizaReply:
    rule: str
    lang: str
    reply: str


class Eliza:
    def __init__(self, rules=RULES, seed: int = 0):
        self.rules = rules
        self.rng = random.Random(seed)

    def respond(self, text: str) -> ElizaReply | None:
        """Return a reply if the message is *only* small talk, else None."""
        remainder = text
        first = None
        for rule in self.rules:
            m = rule.pattern.search(remainder)
            if m:
                if first is None:
                    first = (rule, m)
                remainder = rule.pattern.sub(" ", remainder)
        if first is None:
            return None
        leftover = FILLER.sub(" ", remainder).split()
        if len(leftover) > 1:          # real content left -> let the classifier route it
            return None
        rule, m = first
        groups = [reflect(g) for g in m.groups() if g] or [""]
        template = self.rng.choice(rule.replies)
        try:
            reply = template.format(*groups)
        except IndexError:
            reply = template
        return ElizaReply(rule.name, rule.lang, reply)
