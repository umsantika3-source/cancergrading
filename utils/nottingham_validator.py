"""
nottingham_validator.py — Nottingham Histologic Grading validation layer.

Implements the clinical grading logic from the Nottingham (Elston-Ellis) modification
of the Scarff-Bloom-Richardson grading system for breast cancer.

This module maps model predictions (Grade I/II/III) to the underlying Nottingham
component scores and provides clinical interpretation context for the diagnostic report.

Clinical reference:
    Tubule Formation:    1 (>75% tubules), 2 (10-75%), 3 (<10%)
    Mitotic Count:       1 (0-9/10 HPF), 2 (10-19/10 HPF), 3 (>=20/10 HPF)
    Nuclear Pleomorphism: 1 (small regular), 2 (moderate variation), 3 (marked variation)
    Total: 3-9
    Grade I:  3-5  (Low Grade / well-differentiated)
    Grade II: 6-7  (Intermediate Grade / moderately-differentiated)
    Grade III: 8-9 (High Grade / poorly-differentiated)
"""

from typing import Dict, List, Tuple

# ── Nottingham Grade Definitions ────────────────────────────────────────────

# (min_total, max_total) for each grade
GRADE_RANGES = {
    0: (3, 5),   # Grade I
    1: (6, 7),   # Grade II
    2: (8, 9),   # Grade III
}

GRADE_LABELS = ["Grade I", "Grade II", "Grade III"]
GRADE_NAMES_CLINICAL = [
    "Low Grade (I) — well-differentiated",
    "Intermediate Grade (II) — moderately-differentiated",
    "High Grade (III) — poorly-differentiated",
]

# Typical component score patterns for each grade (most common combinations)
TYPICAL_COMPONENT_PATTERNS: Dict[int, List[Tuple[int, int, int]]] = {
    0: [(2, 1, 1), (1, 2, 1), (1, 1, 2), (1, 1, 1), (2, 2, 1), (2, 1, 2), (1, 2, 2)],
    1: [(2, 2, 2), (3, 2, 2), (2, 3, 2), (2, 2, 3), (3, 1, 3), (3, 3, 1), (1, 3, 3)],
    2: [(3, 3, 2), (3, 2, 3), (2, 3, 3), (3, 3, 3)],
}

# Clinical descriptors for each component score
COMPONENT_DESCRIPTIONS = {
    "tubule": {
        1: ">75% of tumor area forming tubules (well-differentiated)",
        2: "10-75% of tumor area forming tubules (moderately-differentiated)",
        3: "<10% of tumor area forming tubules (poorly-differentiated)",
    },
    "mitotic": {
        1: "0-9 mitoses per 10 high-power fields",
        2: "10-19 mitoses per 10 high-power fields",
        3: "≥20 mitoses per 10 high-power fields",
    },
    "pleomorphism": {
        1: "Small, regular uniform nuclei",
        2: "Moderate variation in nuclear size and shape",
        3: "Marked variation, large irregular nuclei",
    },
}


# ── Core Grading Logic (from gradingplan.md) ─────────────────────────────────

def calculate_grade(tubule_score: int, mitotic_score: int, pleomorphism_score: int) -> Tuple[int, str]:
    """
    Compute total score and final grade from the three component scores.
    This is the exact implementation from gradingplan.md.

    Args:
        tubule_score: 1, 2, or 3
        mitotic_score: 1, 2, or 3
        pleomorphism_score: 1, 2, or 3

    Returns:
        (total_score, grade_string)
    """
    total = tubule_score + mitotic_score + pleomorphism_score
    if total <= 5:
        grade = "Low Grade (I)"
    elif total <= 7:
        grade = "Intermediate Grade (II)"
    else:
        grade = "High Grade (III)"
    return total, grade


# ── Clinical Interpretations ────────────────────────────────────────────────

def grade_to_nottingham_range(grade_idx: int) -> Dict:
    """
    Map a predicted grade index (0, 1, 2) to the corresponding Nottingham
    total score range and clinical description.

    Args:
        grade_idx: 0 = Grade I, 1 = Grade II, 2 = Grade III

    Returns:
        Dictionary with range, description, and typical component patterns.
    """
    if grade_idx not in GRADE_RANGES:
        return {
            "grade": "Unknown",
            "total_range": "N/A",
            "clinical_name": "Unknown",
            "typical_component_patterns": [],
        }

    low, high = GRADE_RANGES[grade_idx]
    return {
        "grade": GRADE_LABELS[grade_idx],
        "grade_index": grade_idx,
        "total_range": f"{low}–{high}",
        "total_min": low,
        "total_max": high,
        "clinical_name": GRADE_NAMES_CLINICAL[grade_idx],
        "typical_component_patterns": TYPICAL_COMPONENT_PATTERNS.get(grade_idx, []),
        "num_patterns": len(TYPICAL_COMPONENT_PATTERNS.get(grade_idx, [])),
    }


def infer_possible_components(grade_idx: int) -> List[Dict]:
    """
    Given a predicted grade, list the possible Nottingham component score
    combinations that could produce that grade.

    Args:
        grade_idx: 0 = Grade I, 1 = Grade II, 2 = Grade III

    Returns:
        List of dicts with tubule, mitotic, pleomorphism scores and total.
    """
    patterns = TYPICAL_COMPONENT_PATTERNS.get(grade_idx, [])
    results = []
    for t, m, p in patterns:
        total = t + m + p
        results.append({
            "tubule_score": t,
            "mitotic_score": m,
            "pleomorphism_score": p,
            "total_score": total,
            "tubule_description": COMPONENT_DESCRIPTIONS["tubule"][t],
            "mitotic_description": COMPONENT_DESCRIPTIONS["mitotic"][m],
            "pleomorphism_description": COMPONENT_DESCRIPTIONS["pleomorphism"][p],
        })
    return results


def get_clinical_summary(predictions: List[Dict], class_names: List[str]) -> Dict:
    """
    Generate a clinical summary from a list of prediction records.

    Args:
        predictions: List of dicts with 'label' (true) and 'pred' (predicted) keys.
        class_names: List of class name strings.

    Returns:
        Dictionary with Nottingham clinical context for the predictions.
    """
    from collections import Counter

    if not predictions:
        return {"error": "No predictions available"}

    pred_counts = Counter(p["pred"] for p in predictions)
    true_counts = Counter(p["label"] for p in predictions)
    total = len(predictions)

    grade_summaries = []
    for idx in range(min(3, len(class_names))):
        pred_count = pred_counts.get(idx, 0)
        true_count = true_counts.get(idx, 0)
        nott_range = grade_to_nottingham_range(idx)

        grade_summaries.append({
            "grade": class_names[idx] if idx < len(class_names) else f"Class {idx}",
            "grade_index": idx,
            "predicted_count": pred_count,
            "predicted_percentage": round(pred_count / total * 100, 1) if total > 0 else 0,
            "true_count": true_count,
            "true_percentage": round(true_count / total * 100, 1) if total > 0 else 0,
            "nottingham_range": nott_range["total_range"],
            "clinical_name": nott_range["clinical_name"],
            "typical_patterns_count": nott_range["num_patterns"],
        })

    return {
        "total_samples": total,
        "grade_summaries": grade_summaries,
        "nottingham_system": {
            "name": "Nottingham Histologic Grade (Elston-Ellis modification)",
            "components": ["Tubule Formation (1-3)", "Mitotic Count (1-3)", "Nuclear Pleomorphism (1-3)"],
            "total_range": "3–9",
            "grade_mapping": "3–5 = Grade I, 6–7 = Grade II, 8–9 = Grade III",
            "reference": "gradingplan.md",
        },
        "clinical_interpretation": generate_clinical_conclusion(grade_summaries),
    }


def generate_clinical_conclusion(grade_summaries: List[Dict]) -> str:
    """
    Generate a human-readable clinical conclusion from grade summaries.
    """
    if not grade_summaries:
        return "No data available for clinical interpretation."

    # Find the dominant grade by prediction count
    dominant = max(grade_summaries, key=lambda g: g["predicted_count"])

    lines = [
        f"Nottingham Histologic Grade Assessment",
        f"---------------------------------------",
        f"Model predictions map to the following Nottingham grade distribution:",
    ]
    for g in grade_summaries:
        lines.append(
            f"  {g['grade']}: {g['predicted_count']}/{g['predicted_count'] + sum(s['predicted_count'] for s in grade_summaries if s['grade_index'] != g['grade_index'])} "
            f"({g['predicted_percentage']}%) — Nottingham total {g['nottingham_range']} — {g['clinical_name']}"
        )
        lines.append(
            f"    Compatible with {g['typical_patterns_count']} component score combination(s)"
        )

    lines.append("")
    lines.append(
        f"The most common prediction is {dominant['grade']} "
        f"(Nottingham total {dominant['nottingham_range']})."
    )
    lines.append("")
    lines.append(
        "This provides clinical validation that the model's outputs align with "
        "the established Nottingham histologic grading standard used by pathologists."
    )

    return "\n".join(lines)


# ── Misclassification Risk Assessment ───────────────────────────────────────

RISK_LEVELS = {
    "critical": {
        "level": "critical",
        "label": "Critical Risk",
        "description": "Grade III misclassified as Grade I — most dangerous error, "
                       "would lead to under-treatment of aggressive cancer.",
        "color": "B71C1C",
    },
    "high": {
        "level": "high",
        "label": "High Risk",
        "description": "Grade III misclassified as Grade II or Grade II misclassified "
                       "as Grade I — significant under-grading risk.",
        "color": "E65100",
    },
    "moderate": {
        "level": "moderate",
        "label": "Moderate Risk",
        "description": "Grade I misclassified as Grade II — over-grading, may lead "
                       "to unnecessary treatment.",
        "color": "F57F17",
    },
    "low": {
        "level": "low",
        "label": "Low Risk",
        "description": "Grade II misclassified as Grade III — over-grading with "
                       "moderate clinical impact.",
        "color": "2E7D32",
    },
}


def classify_misclassification_risk(true_idx: int, pred_idx: int) -> Dict:
    """
    Classify the clinical risk of a misclassification.

    Args:
        true_idx: True grade index (0, 1, 2)
        pred_idx: Predicted grade index (0, 1, 2)

    Returns:
        Risk level dictionary with label, description, and color.
    """
    if true_idx == pred_idx:
        return {"level": "none", "label": "Correct", "description": "", "color": "1B5E20"}

    # Grade III (2) → Grade I (0): Critical
    if true_idx == 2 and pred_idx == 0:
        return dict(RISK_LEVELS["critical"])
    # Grade III (2) → Grade II (1) or Grade II (1) → Grade I (0): High
    if (true_idx == 2 and pred_idx == 1) or (true_idx == 1 and pred_idx == 0):
        return dict(RISK_LEVELS["high"])
    # Grade I (0) → Grade II (1): Moderate
    if true_idx == 0 and pred_idx == 1:
        return dict(RISK_LEVELS["moderate"])
    # Grade I (0) → Grade III (2) or Grade II (1) → Grade III (2): Low
    return dict(RISK_LEVELS["low"])


# ── Comprehensive Validation Report ─────────────────────────────────────────

def build_nottingham_validation_block(eval_dict: Dict, pred_records: List[Dict],
                                       class_names: List[str]) -> Dict:
    """
    Build a complete Nottingham validation block for embedding into the diagnostic report.

    Args:
        eval_dict: Evaluation dictionary from trainer.evaluate()
        pred_records: List of prediction records with 'label' and 'pred'
        class_names: List of class names (e.g., ['Grade I', 'Grade II', 'Grade III'])

    Returns:
        Dictionary with Nottingham validation data ready for the report.
    """
    clinical_summary = get_clinical_summary(pred_records, class_names)

    # Count misclassifications by clinical risk
    risk_counts = {"critical": 0, "high": 0, "moderate": 0, "low": 0, "none": 0}
    for rec in pred_records:
        risk = classify_misclassification_risk(rec["label"], rec["pred"])
        risk_counts[risk["level"]] = risk_counts.get(risk["level"], 0) + 1

    total = len(pred_records)
    risk_percentages = {
        k: round(v / total * 100, 1) if total > 0 else 0
        for k, v in risk_counts.items()
    }

    return {
        "clinical_summary": clinical_summary,
        "risk_analysis": {
            "critical_misclassifications": risk_counts.get("critical", 0),
            "critical_pct": risk_percentages.get("critical", 0),
            "high_risk_misclassifications": risk_counts.get("high", 0),
            "high_risk_pct": risk_percentages.get("high", 0),
            "moderate_risk_misclassifications": risk_counts.get("moderate", 0),
            "moderate_risk_pct": risk_percentages.get("moderate", 0),
            "low_risk_misclassifications": risk_counts.get("low", 0),
            "low_risk_pct": risk_percentages.get("low", 0),
            "correct_predictions": risk_counts.get("none", 0),
            "correct_pct": risk_percentages.get("none", 0),
        },
        "grade_ranges": [
            grade_to_nottingham_range(i) for i in range(len(class_names))
        ],
    }
