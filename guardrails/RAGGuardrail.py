"""Knowledge Groudning using a Retrieval-Augmented Generation (RAG) approach.
This module provides a framework for integrating external knowledge sources into the generation process of language models.
Tasks: 
- Define a base class for RAG models.
- Check that all claims are sourced from the knowledge base."""

class RAGGuardrail:
    def __init__(self):
        self.approved_sources = {
            'MEWS':  'MEWS >= 3 = elevated risk, >= 5 = critical',
            'NEWS':  'NEWS >= 5 = escalate care, >= 7 = urgent',
            'NEWS2': 'NEWS2 >= 5 = escalate care, >= 7 = urgent',
            'REMS':  'REMS >= 8 = elevated risk, >= 12 = critical',
            'CART':  'CART >= 5 = elevated risk',
            'CCI':   'CCI >= 2 = significant comorbidity burden',
            'qSOFA': 'qSOFA >= 2 = sepsis risk',
            'Sepsis_bundle': 'Lactate > 2 = sepsis concern',
        }

        # Columns to cross-check against the generated brief
        self._numeric_checks = {
            'score_MEWS':        ('mews',  'MEWS'),
            'score_NEWS':        ('news',  'NEWS'),
            'score_NEWS2':       ('news2', 'NEWS2'),
            'triage_heartrate':  ('heart rate', 'HR'),
            'triage_temperature':('temp',  'temperature'),
            'triage_sbp':        ('sbp',   'systolic'),
        }

    def validate_grounding(self, AURA, patient_row=None):
        """Check that numeric values cited in the brief match the patient's actual data.

        If patient_row is provided, verifies that any score/vital mentioned in the
        brief carries the correct value from the dataset.  Discrepancies are flagged
        as potential hallucinations.
        """
        unsourced_claims = []

        if patient_row is not None:
            import math, pandas as pd
            for col, (keyword, label) in self._numeric_checks.items():
                raw = patient_row.get(col)
                if raw is None:
                    continue
                try:
                    fval = float(raw)
                except (TypeError, ValueError):
                    continue
                if math.isnan(fval):
                    continue

                expected_str = str(round(fval))
                aura_lower = AURA.lower()

                # Only flag if the brief mentions the concept but cites a wrong number
                if keyword in aura_lower:
                    # Accept the rounded value OR the original one-decimal representation
                    accepted = {expected_str, f"{fval:.1f}"}
                    if not any(tok in AURA for tok in accepted):
                        unsourced_claims.append(
                            f"{label} value mismatch: brief does not cite {expected_str}"
                        )

        if unsourced_claims:
            return {
                "grounded": False,
                "unsourced_claims": unsourced_claims,
                "action": "REGENERATE AURA",
            }
        return {
            "grounded": True,
            "action": "APPROVE",
        }
            
# Example usage
if __name__ == "__main__":
    guardrail = RAGGuardrail()
    AURA_example = "The patient has a hallucinated stat of 5, which is not sourced from MEWS."
    result = guardrail.validate_grounding(AURA_example)
    print(result)
    