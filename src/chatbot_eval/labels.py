from __future__ import annotations

from .models import Outcome

OUTCOME_HEBREW = {
    Outcome.CORRECT_ANSWER: "תשובה נכונה",
    Outcome.PARTIAL_TOO_LITTLE: "תשובה חלקית — חסר מידע",
    Outcome.PARTIAL_TOO_MUCH: "תשובה חלקית — עודף מידע",
    Outcome.UNRELATED_ANSWER: "תשובה שגויה ולא קשורה",
    Outcome.MISLEADING_HALLUCINATION: "הזיה מטעה",
    Outcome.INCORRECT_ABSTENTION: "נמנע ממענה למרות שהמידע קיים",
    Outcome.CORRECT_ABSTENTION: "נמנע ממענה בצדק",
    Outcome.SHOULD_HAVE_ABSTAINED: "ענה למרות שהיה צריך להימנע",
    Outcome.CHATBOT_ERROR: "שגיאת מערכת בצ׳אטבוט",
    Outcome.JUDGE_ERROR: "שגיאה בתהליך הבדיקה",
}

OUTCOME_COLORS = {
    Outcome.CORRECT_ANSWER: "#15803d",
    Outcome.PARTIAL_TOO_LITTLE: "#ca8a04",
    Outcome.PARTIAL_TOO_MUCH: "#d97706",
    Outcome.UNRELATED_ANSWER: "#64748b",
    Outcome.MISLEADING_HALLUCINATION: "#dc2626",
    Outcome.INCORRECT_ABSTENTION: "#ea580c",
    Outcome.CORRECT_ABSTENTION: "#0f766e",
    Outcome.SHOULD_HAVE_ABSTAINED: "#b91c1c",
    Outcome.CHATBOT_ERROR: "#475569",
    Outcome.JUDGE_ERROR: "#7c3aed",
}

ANSWER_SCOPE_HEBREW = {
    "exact": "היקף מתאים",
    "too_little": "חסר מידע",
    "too_much": "עודף מידע",
}

INCORRECT_TYPE_HEBREW = {
    "not_applicable": "לא רלוונטי",
    "unrelated": "תשובה לא קשורה",
    "hallucination": "הזיה מטעה",
}
