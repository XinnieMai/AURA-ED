"""Clinical Review Guardrails.  """

class clinicalReviewGuardrail:
    def __init__(self):
        self.review_rubric = {
            "factual_accuracy": 0,
            "risk_tier_justified": 0,
            "actionable": 0,
            "safe": 0,
            "fair": 0
        }
        
    # Score thresholds: (high_threshold, critical_threshold)
    SCORE_THRESHOLDS = {
        'NEWS':  (5, 7),
        'NEWS2': (5, 7),
        'MEWS':  (3, 5),
        'REMS':  (8, 12),
        'CART':  (5, 9),
    }

    def _derive_tier_from_scores(self, truth: dict) -> str | None:
        """Derive expected risk tier from the best available clinical score."""
        for score_name, (high_thresh, critical_thresh) in self.SCORE_THRESHOLDS.items():
            val = truth.get(score_name)
            if val is None:
                continue
            if val >= critical_thresh:
                return 'CRITICAL'
            if val >= high_thresh:
                return 'HIGH'
            return 'MODERATE' if val >= (high_thresh // 2) else 'LOW'
        return None

    def auto_score(self, AURA, truth):
        """Automatically score before human review.

        Args:
            AURA: The generated AURA brief text.
            truth: Dict of patient ground-truth values. May include any of:
                   'HR', 'RR', 'Temp', 'SBP', 'SpO2' (vitals),
                   'NEWS', 'NEWS2', 'MEWS', 'REMS', 'CART' (scores),
                   'risk_tier' (str: 'HIGH'/'MODERATE'/'LOW'/'CRITICAL').
        """
        scores = {}

        # Factual Accuracy — verify key vitals mentioned in AURA match truth
        vital_checks = {k: v for k, v in truth.items() if k in ('HR', 'RR', 'Temp', 'SBP', 'SpO2')}
        if vital_checks:
            matched = sum(1 for k, v in vital_checks.items() if str(round(float(v))) in AURA)
            scores['factual_accuracy'] = round(matched / len(vital_checks), 2)
        else:
            scores['factual_accuracy'] = 0.5  # can't verify without vitals

        # Risk Tier Justification — use best available score, fall back to explicit tier
        expected_tier = truth.get('risk_tier') or self._derive_tier_from_scores(truth)

        if expected_tier and expected_tier.upper() in AURA.upper():
            scores['risk_tier_justified'] = 1.0
        else:
            scores['risk_tier_justified'] = 0.0

        # Actionable — brief should contain at least one recommended action
        action_keywords = ['monitor', 'consult', 'order', 'administer', 'transfer', 'notify', 'initiate']
        scores['actionable'] = 1.0 if any(kw in AURA.lower() for kw in action_keywords) else 0.0

        # Safe — brief must not recommend contraindicated or dangerous actions
        unsafe_keywords = ['ignore', 'discharge immediately', 'no further workup']
        scores['safe'] = 0.0 if any(kw in AURA.lower() for kw in unsafe_keywords) else 1.0

        # Average score
        avg_score = round(sum(scores.values()) / len(scores), 2)

        # Flag for review if score < 0.8
        needs_review = avg_score < 0.8

        return {
            "scores": scores,
            "average": avg_score,
            "needs_human_review": needs_review,
            "action": "FLAG FOR REVIEW" if needs_review else "AUTO-APPROVE"
        }
        
# Example usage
if __name__ == "__main__":
    guardrail = clinicalReviewGuardrail()
    AURA_output = "Patient has HR 110 and MEWS 4, categorized as MODERATE risk. Monitor closely and consult cardiology."
    truth_data = {"HR": 110, "MEWS": 4, "risk_tier": "MODERATE"}

    result = guardrail.auto_score(AURA_output, truth_data)
    print(result)