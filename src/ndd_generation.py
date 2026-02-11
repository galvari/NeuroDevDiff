from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import random
import pandas as pd


@dataclass
class NDDConfig:
    version: str = "1"
    n_cases: int = 2000
    seed: int = 42
    noise_level: float = 1.0


RED_FLAGS = [
    "self-harm thoughts",
    "acute aggression risk",
    "psychotic-like symptoms",
]


ARCHETYPES: Dict[str, Dict] = {
    "ASD": {
        "symptoms": {
            "social": ["reduced reciprocity", "poor peer interaction", "limited social initiation"],
            "language": ["pragmatic difficulties", "literal interpretation", "atypical prosody"],
            "rrb": ["rigid routines", "restricted interests", "repetitive behaviors"],
            "sensory": ["sensory sensitivity", "sensory seeking"],
        },
        "base_comorb": {"ADHD": 0.25, "ANXIETY": 0.20, "SLD": 0.10},
        "weight": 0.20,
    },
    "ADHD": {
        "symptoms": {
            "attention": ["inattention", "disorganization", "forgetfulness"],
            "behavior": ["impulsivity", "restlessness", "interrupting others"],
            "school": ["homework incomplete", "poor sustained effort"],
        },
        "base_comorb": {"ANXIETY": 0.15, "ASD": 0.15, "SLD": 0.20},
        "weight": 0.18,
    },
    "OCD": {
        "symptoms": {
            "anxiety": ["intrusive thoughts", "high distress with uncertainty"],
            "compulsions": ["checking", "washing", "counting", "reassurance seeking"],
            "avoidance": ["avoid triggers", "ritual-dependent routines"],
        },
        "base_comorb": {"ANXIETY": 0.25, "ASD": 0.10},
        "weight": 0.10,
    },
    "ANXIETY": {
        "symptoms": {
            "anxiety": ["worry", "avoidance", "somatic complaints", "sleep difficulties"],
            "school": ["school refusal", "performance fear"],
            "social": ["shy/withdrawn in novel contexts"],
        },
        "base_comorb": {"ADHD": 0.10, "ASD": 0.10, "SELECTIVE_MUTISM": 0.12},
        "weight": 0.16,
    },
    "SELECTIVE_MUTISM": {
        "symptoms": {
            "communication": ["speaks at home but not at school", "freezes in social settings"],
            "anxiety": ["high social inhibition", "avoidance of speaking"],
            "school": ["teacher reports silence"],
        },
        "base_comorb": {"ANXIETY": 0.35, "ASD": 0.10},
        "weight": 0.08,
    },
    "SLD": {
        "symptoms": {
            "learning": ["reading difficulties", "spelling errors", "math difficulties"],
            "school": ["slow progress despite effort", "avoidance of homework"],
            "emotional": ["frustration", "low self-esteem"],
        },
        "base_comorb": {"ADHD": 0.25, "ANXIETY": 0.15},
        "weight": 0.14,
    },
    "GDD_ID": {
        "symptoms": {
            "development": ["delayed milestones", "adaptive difficulties", "global learning delays"],
            "language": ["language delay", "limited vocabulary for age"],
            "motor": ["poor coordination"],
        },
        "base_comorb": {"ASD": 0.15, "ANXIETY": 0.05},
        "weight": 0.07,
    },
    "NDD_UNSPEC": {
        "symptoms": {
            "mixed": ["mixed difficulties across domains", "unclear onset", "inconsistent reports"],
            "attention": ["variable attention"],
            "social": ["social withdrawal in some contexts"],
        },
        "base_comorb": {"ASD": 0.10, "ADHD": 0.10, "ANXIETY": 0.10},
        "weight": 0.07,
    },
}


# Scores are scaled 1–19 (proxy scale).
COGNITIVE_PROFILES: Dict[str, Dict[str, Tuple[float, float]]] = {
    "ASD": {
        "verbal_language": (8, 2),
        "visuospatial": (12, 2),
        "working_memory": (7, 2),
        "processing_speed": (6, 2),
        "attention": (8, 3),
        "motor": (9, 3),
    },
    "ADHD": {
        "verbal_language": (10, 2),
        "visuospatial": (10, 2),
        "working_memory": (7, 2),
        "processing_speed": (7, 2),
        "attention": (5, 2),
        "motor": (10, 3),
    },
    "OCD": {
        "verbal_language": (12, 2),
        "visuospatial": (10, 2),
        "working_memory": (10, 2),
        "processing_speed": (8, 2),
        "attention": (10, 2),
        "motor": (10, 2),
    },
    "ANXIETY": {
        "verbal_language": (10, 2),
        "visuospatial": (10, 2),
        "working_memory": (9, 2),
        "processing_speed": (8, 2),
        "attention": (9, 2),
        "motor": (10, 2),
    },
    "SELECTIVE_MUTISM": {
        "verbal_language": (9, 2),
        "visuospatial": (10, 2),
        "working_memory": (9, 2),
        "processing_speed": (8, 2),
        "attention": (9, 2),
        "motor": (10, 2),
    },
    "SLD": {
        "verbal_language": (10, 1.5),
        "visuospatial": (10, 1.5),
        "working_memory": (10, 1.5),
        "processing_speed": (10, 1.5),
        "attention": (10, 1.5),
        "motor": (10, 1.5),
    },
    "GDD_ID": {
        "verbal_language": (4, 1),
        "visuospatial": (4, 1),
        "working_memory": (4, 1),
        "processing_speed": (4, 1),
        "attention": (4, 1),
        "motor": (4, 1),
    },
    "NDD_UNSPEC": {
        "verbal_language": (8, 3),
        "visuospatial": (8, 3),
        "working_memory": (8, 3),
        "processing_speed": (8, 3),
        "attention": (8, 3),
        "motor": (8, 3),
    },
}


NON_SPECIFIC = [
    ("sleep", "sleep difficulties"),
    ("emotional", "irritability"),
    ("attention", "concentration problems"),
    ("emotional", "low frustration tolerance"),
]

NEIGHBORS = {
    "ASD": ["ADHD", "ANXIETY", "OCD"],
    "ADHD": ["ASD", "ANXIETY", "SLD"],
    "ANXIETY": ["OCD", "SELECTIVE_MUTISM", "ADHD"],
    "OCD": ["ANXIETY", "ASD"],
    "SELECTIVE_MUTISM": ["ANXIETY", "ASD"],
    "SLD": ["ADHD", "ANXIETY"],
    "GDD_ID": ["ASD", "NDD_UNSPEC"],
    "NDD_UNSPEC": ["ASD", "ADHD", "ANXIETY"],
}


def _score(rng: random.Random, mean: float, spread: float, low: int = 1, high: int = 19) -> int:
    v = int(round(rng.gauss(mean, spread)))
    return max(low, min(high, v))


def _cognitive_pattern(scores: Dict[str, int]) -> str:
    mx, mn = max(scores.values()), min(scores.values())
    var = mx - mn
    if var < 3 and mx <= 6:
        return "homogeneous_low"
    if var < 3:
        return "homogeneous_average"
    return "dishomogeneous"


def _pick_symptoms(rng: random.Random, symptom_dict: Dict[str, List[str]], k: int = 4) -> List[Tuple[str, str]]:
    all_items: List[Tuple[str, str]] = []
    for dom, items in symptom_dict.items():
        all_items.extend([(dom, s) for s in items])
    if not all_items:
        return []
    k = min(k, len(all_items))
    return rng.sample(all_items, k=k)


def _sample_comorbidity(rng: random.Random, base_comorb: Dict[str, float]) -> List[str]:
    out: List[str] = []
    for dx, p in base_comorb.items():
        if rng.random() < p:
            out.append(dx)
    return sorted(set(out))


def make_case(case_id: int, cfg: NDDConfig, rng: random.Random) -> Dict:
    labels = list(ARCHETYPES.keys())
    weights = [ARCHETYPES[k]["weight"] for k in labels]
    true_dx = rng.choices(labels, weights=weights, k=1)[0]
    base = ARCHETYPES[true_dx]

    age = rng.choice([5, 6, 7, 8, 9, 10, 11, 12])
    sex = rng.choice(["M", "F"])
    context = rng.choice(["preschool", "primary school", "home+school"])
    duration = rng.choice(["3 months", "6 months", "1 year", "since early childhood"])
    severity = rng.choices(["mild", "moderate", "severe"], weights=[0.35, 0.45, 0.20], k=1)[0]

    core_sym = _pick_symptoms(rng, base["symptoms"], k=4)
    comorbid = _sample_comorbidity(rng, base["base_comorb"])

    # Nonspecific symptoms
    if rng.random() < 0.65 * cfg.noise_level:
        core_sym.append(rng.choice(NON_SPECIFIC))

    # Neighbor contamination
    if rng.random() < 0.25 * cfg.noise_level:
        neigh = rng.choice(NEIGHBORS[true_dx])
        core_sym.extend(_pick_symptoms(rng, ARCHETYPES[neigh]["symptoms"], k=1))

    # Sometimes comorbidity dominates
    if comorbid and rng.random() < 0.15 * cfg.noise_level:
        dominant = rng.choice(comorbid)
        core_sym.extend(_pick_symptoms(rng, ARCHETYPES[dominant]["symptoms"], k=1))

    # Missingness
    onset_described = rng.random() < 0.80
    cross_setting = rng.random() < (0.82 if true_dx in ["ASD", "ADHD", "GDD_ID"] else 0.68)
    functioning_described = rng.random() < (0.75 if severity != "mild" else 0.60)

    has_teacher_report = rng.random() < 0.85
    has_dev_history = rng.random() < 0.70
    has_language_assessment = rng.random() < 0.50
    has_learning_assessment = rng.random() < 0.45

    risk_high = rng.random() < 0.035
    red_flags = [rng.choice(RED_FLAGS)] if risk_high else []

    # Differential shortlist
    diff = set()
    if true_dx == "ASD":
        diff.update(["ADHD", "ANXIETY"])
    if true_dx == "ADHD":
        diff.update(["SLD", "ANXIETY", "ASD"])
    if true_dx == "SLD":
        diff.update(["ADHD", "ANXIETY"])
    if true_dx == "OCD":
        diff.update(["ANXIETY", "ASD"])
    if true_dx == "SELECTIVE_MUTISM":
        diff.update(["ANXIETY", "ASD"])
    if true_dx == "GDD_ID":
        diff.update(["ASD", "NDD_UNSPEC"])
    if true_dx == "NDD_UNSPEC":
        diff.update(["ASD", "ADHD", "ANXIETY"])

    diff.update(comorbid)
    diff.discard(true_dx)
    plausible_alternatives = sorted(list(diff))[:3]

    # Cognitive profile
    prof = COGNITIVE_PROFILES[true_dx]
    scores = {k: _score(rng, mean=v[0], spread=v[1]) for k, v in prof.items()}

    # Variant noise (avoid barcode learning)
    if true_dx == "ASD" and rng.random() < 0.20 * cfg.noise_level:
        for k in scores:
            scores[k] = _score(rng, 9, 2)
    if true_dx == "ADHD" and rng.random() < 0.15 * cfg.noise_level:
        for k in scores:
            scores[k] = _score(rng, 10, 1.5)
    if true_dx == "SLD" and rng.random() < 0.10 * cfg.noise_level:
        scores["working_memory"] = _score(rng, 8, 2)
        scores["processing_speed"] = _score(rng, 8, 2)

    cognitive_pattern = _cognitive_pattern(scores)

    missing: List[str] = []
    if not onset_described:
        missing.append("onset timeline")
    if not functioning_described:
        missing.append("functional impairment details")
    if not cross_setting:
        missing.append("cross-setting symptoms (home vs school)")
    if not has_teacher_report:
        missing.append("teacher report")
    if not has_dev_history:
        missing.append("developmental history")
    if (true_dx in ["ASD", "SELECTIVE_MUTISM"] or "ASD" in plausible_alternatives) and not has_language_assessment:
        missing.append("language/pragmatics assessment")
    if (true_dx == "SLD" or "SLD" in plausible_alternatives) and not has_learning_assessment:
        missing.append("learning assessment (reading/writing/math)")

    # Defer policy
    critical_missing = (not onset_described) or (not functioning_described)
    high_ambiguity = (len(plausible_alternatives) >= 3) and (severity != "severe") and (len(missing) >= 2)

    if risk_high or critical_missing:
        should_defer = 1
    elif high_ambiguity:
        should_defer = 1 if rng.random() < 0.60 else 0
    else:
        should_defer = 1 if rng.random() < 0.15 else 0

    return {
        "case_id": case_id,
        "age": age,
        "sex": sex,
        "context": context,
        "duration": duration,
        "severity": severity,
        "true_profile": true_dx,
        "comorbidity": comorbid,
        "plausible_alternatives": plausible_alternatives,
        "symptoms": core_sym,  # list of (domain, symptom)
        "red_flags": red_flags,
        "missing_info_list": missing,
        "should_defer": should_defer,
        "risk_high": int(risk_high),
        "cognitive_profile": scores,
        "cognitive_pattern": cognitive_pattern,
        "info": {
            "onset_described": onset_described,
            "cross_setting": cross_setting,
            "functioning_described": functioning_described,
            "teacher_report": has_teacher_report,
            "developmental_history": has_dev_history,
            "language_assessment": has_language_assessment,
            "learning_assessment": has_learning_assessment,
        },
    }


def generate_neurodevdiff(cfg: NDDConfig) -> pd.DataFrame:
    rng = random.Random(cfg.seed)
    cases = [make_case(i, cfg, rng) for i in range(1, cfg.n_cases + 1)]

    rows = []
    for c in cases:
        cp = c["cognitive_profile"]
        rows.append(
            {
                "case_id": c["case_id"],
                "age": c["age"],
                "sex": c["sex"],
                "context": c["context"],
                "duration": c["duration"],
                "severity": c["severity"],
                "true_profile": c["true_profile"],
                "comorbidity": ", ".join(c["comorbidity"]) if c["comorbidity"] else "",
                "plausible_alternatives": ", ".join(c["plausible_alternatives"]) if c["plausible_alternatives"] else "",
                "symptoms": "; ".join([f"{d}:{s}" for d, s in c["symptoms"]]),
                "red_flags": ", ".join(c["red_flags"]) if c["red_flags"] else "",
                "missing_info": ", ".join(c["missing_info_list"]) if c["missing_info_list"] else "",
                "should_defer": c["should_defer"],
                "risk_high": c["risk_high"],
                "cog_verbal_language": cp["verbal_language"],
                "cog_visuospatial": cp["visuospatial"],
                "cog_working_memory": cp["working_memory"],
                "cog_processing_speed": cp["processing_speed"],
                "cog_attention": cp["attention"],
                "cog_motor": cp["motor"],
                "cognitive_pattern": c["cognitive_pattern"],
            }
        )

    return pd.DataFrame(rows)


def print_summary(df: pd.DataFrame) -> None:
    print("Dataset shape:", df.shape)

    print("\nClass balance (true_profile):")
    print(df["true_profile"].value_counts(normalize=True).round(3))

    print(
        "\nDefer rate:",
        round(df["should_defer"].mean(), 3),
        " | High-risk rate:",
        round(df["risk_high"].mean(), 3),
    )

    print("\nDefer rate:", df["should_defer"].mean().round(3))
    print(
        df.groupby("true_profile")["should_defer"]
          .mean()
          .round(3)
          .sort_values(ascending=False)
    )

    print("\nMean cognitive scores by profile:")
    print(
        df.groupby("true_profile")[[
            "cog_verbal_language","cog_visuospatial","cog_working_memory",
            "cog_processing_speed","cog_attention","cog_motor"
        ]].mean().round(2)
    )