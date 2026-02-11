from __future__ import annotations

from typing import List, Tuple
import random
import textwrap
import pandas as pd


OPENERS = [
    "A pediatric neurodevelopmental evaluation is requested based on caregiver and school concerns.",
    "A child is referred for evaluation due to concerns about functioning in everyday contexts.",
    "This case describes a child presenting with developmental and behavioral concerns.",
]

CONTEXT_SENT = {
    "preschool": [
        "The child is currently attending preschool.",
        "The child is in a preschool setting.",
    ],
    "primary school": [
        "The child is currently attending primary school.",
        "The child is in a primary school setting.",
    ],
    "home+school": [
        "Concerns are reported across both home and school contexts.",
        "Difficulties are described in multiple settings, including home and school.",
    ],
}

SEVERITY_SENT = {
    "mild": [
        "Overall severity is described as mild, though specific situations can be challenging.",
        "Difficulties are mild overall, with noticeable impact in select contexts.",
    ],
    "moderate": [
        "Overall severity is described as moderate, with meaningful impact on daily routines.",
        "Difficulties are of moderate severity and interfere with daily functioning.",
    ],
    "severe": [
        "Overall severity is described as severe, with substantial functional impact.",
        "Difficulties are severe and significantly disrupt daily functioning.",
    ],
}

MISSING_TO_QUESTIONS = {
    "onset timeline": [
        "When did the difficulties first emerge, and was the onset sudden or gradual?",
        "Were there any early developmental concerns (language, play, social engagement)?",
    ],
    "functional impairment details": [
        "How do the difficulties affect daily functioning at home, school, and with peers?",
        "Which situations lead to the most impairment (transitions, homework, social demands)?",
    ],
    "cross-setting symptoms (home vs school)": [
        "Are the symptoms present across settings (home, school, community), or situation-specific?",
        "What differences do caregivers and teachers report?",
    ],
    "teacher report": [
        "Could we obtain a teacher report describing classroom behavior and learning progress?",
        "Are there standardized school observations or rating scales available?",
    ],
    "developmental history": [
        "Can we review developmental milestones (language, motor, adaptive skills) and early social communication?",
        "Any relevant perinatal/medical history?",
    ],
    "language/pragmatics assessment": [
        "Is a language and pragmatic communication assessment available (including social use of language)?",
        "Any concerns about speech production, comprehension, or narrative skills?",
    ],
    "learning assessment (reading/writing/math)": [
        "Have reading, writing, and math skills been formally assessed (psychoeducational testing)?",
        "What was the response to targeted school interventions?",
    ],
}


def _parse_symptoms(symptom_str: str) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    if not isinstance(symptom_str, str) or symptom_str.strip() == "":
        return out
    for part in symptom_str.split(";"):
        part = part.strip()
        if ":" in part:
            dom, sym = part.split(":", 1)
            out.append((dom.strip(), sym.strip()))
        else:
            out.append(("general", part))
    return out


def _pretty_list(xs: List[str], max_items: int = 6) -> str:
    xs = [x for x in xs if x]
    xs = xs[:max_items]
    if not xs:
        return ""
    if len(xs) == 1:
        return xs[0]
    if len(xs) == 2:
        return f"{xs[0]} and {xs[1]}"
    return ", ".join(xs[:-1]) + f", and {xs[-1]}"


def _band(x: float) -> str:
    if x <= 4:
        return "markedly low"
    if x <= 7:
        return "below average"
    if x <= 12:
        return "average"
    return "above average"


def _cognitive_sentence(row: pd.Series) -> str:
    vl = row["cog_verbal_language"]
    vs = row["cog_visuospatial"]
    wm = row["cog_working_memory"]
    ps = row["cog_processing_speed"]
    att = row["cog_attention"]
    mot = row["cog_motor"]
    pattern = row["cognitive_pattern"]

    if pattern == "homogeneous_low":
        return (
            "Cognitive screening suggests a globally reduced profile across domains "
            f"(verbal {_band(vl)}, visuospatial {_band(vs)}, working memory {_band(wm)}, "
            f"processing speed {_band(ps)}, attention {_band(att)}, motor {_band(mot)})."
        )

    if pattern == "homogeneous_average":
        return (
            "Cognitive screening is broadly even across domains "
            f"(verbal {_band(vl)}, visuospatial {_band(vs)}, working memory {_band(wm)}, "
            f"processing speed {_band(ps)}, attention {_band(att)}, motor {_band(mot)})."
        )

    scores = {
        "verbal/language": vl,
        "visuospatial": vs,
        "working memory": wm,
        "processing speed": ps,
        "attention": att,
        "motor": mot,
    }
    strongest = max(scores, key=scores.get)
    weakest = min(scores, key=scores.get)
    mean_score = (vl + vs + wm + ps + att + mot) / 6.0
    return (
        "Cognitive screening indicates a heterogeneous profile, "
        f"with relative strength in {strongest} and relative weakness in {weakest} "
        f"(other domains fall in the {_band(mean_score)} range)."
    )


def build_vignette(row: pd.Series, rng: random.Random) -> str:
    age = int(row["age"])
    sex = row["sex"]
    context = row["context"]
    duration = row["duration"]
    severity = row["severity"]

    sym_items = _parse_symptoms(row["symptoms"])
    rng.shuffle(sym_items)
    n_sym = min(len(sym_items), rng.choice([4, 5, 6]))
    sym_phrases = [s for _, s in sym_items[:n_sym]]
    symptoms_text = _pretty_list(sym_phrases, max_items=6)

    red_text = ""
    if isinstance(row.get("red_flags", ""), str) and row["red_flags"].strip():
        red_text = " A safety screen notes a potential red-flag item that warrants clinical attention."

    cog_text = _cognitive_sentence(row)

    missing = row.get("missing_info", "")
    missing_list = [m.strip() for m in str(missing).split(",") if m.strip()]
    if missing_list:
        missing_note = (
            f" Key information is currently missing (e.g., {_pretty_list(missing_list, max_items=3)}), "
            "which increases diagnostic uncertainty."
        )
    else:
        missing_note = " Available information is reasonably complete, though further clarification may still be helpful."

    opener = rng.choice(OPENERS)
    ctx_sent = rng.choice(CONTEXT_SENT.get(context, ["The child is currently in school."]))
    sev_sent = rng.choice(SEVERITY_SENT.get(severity, ["Severity is variable."]))

    txt = (
        f"{opener} The patient is a {age}-year-old {('boy' if sex=='M' else 'girl')}. "
        f"{ctx_sent} Reported difficulties have been present for {duration}. {sev_sent} "
        f"Core features include {symptoms_text}.{red_text}\n\n"
        f"{cog_text}\n\n"
        f"{missing_note}"
    )

    return textwrap.fill(txt, width=110)


def build_questions(row: pd.Series, rng: random.Random, max_q: int = 5) -> List[str]:
    missing = row.get("missing_info", "")
    missing_list = [m.strip() for m in str(missing).split(",") if m.strip()]

    qs: List[str] = []
    for m in missing_list:
        qs.extend(MISSING_TO_QUESTIONS.get(m, []))

    if len(qs) < 2:
        qs.extend(
            [
                "What are the child's strengths and which situations are most successful?",
                "Are there any prior assessments or interventions, and what was the response?",
            ]
        )

    # dedup + shuffle
    seen = set()
    uniq = []
    for q in qs:
        if q not in seen:
            seen.add(q)
            uniq.append(q)
    rng.shuffle(uniq)
    return uniq[:max_q]


def build_rationale(row: pd.Series) -> str:
    if int(row.get("risk_high", 0)) == 1:
        return "Defer: a red-flag item requires clinician-led risk assessment."
    missing = str(row.get("missing_info", "")).strip()
    if int(row.get("should_defer", 0)) == 1:
        if missing:
            return f"Defer: missing key information ({missing}) increases diagnostic uncertainty."
        return "Defer: uncertainty is high; additional clinical information is needed before decisions."
    return "No defer: available information supports a tentative working hypothesis, with routine follow-up questions."


def add_text_fields(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    rng = random.Random(seed)
    out = df.copy()
    out["vignette_en"] = out.apply(lambda r: build_vignette(r, rng), axis=1)
    out["questions_to_ask_en"] = out.apply(lambda r: build_questions(r, rng, max_q=5), axis=1)
    out["should_defer_rationale_en"] = out.apply(build_rationale, axis=1)
    return out