# AURA-ED: AI-powered Utilization & Risk Analysis in the Emergency Department

**Course: DS 5003: Healthcare Data Science**

AURA-ED is a clinical decision-support tool designed to convert fragmented Emergency Department (ED) data into structured, actionable "Early Risk Profiles". By leveraging Large Language Models (LLMs) and the Stanford MC-MED dataset, this tool reduces clinicians' cognitive load by synthesizing vital signs, laboratory results, and medical histories into concise risk narratives.

# Project Purpose: 
Emergency physicians spend significant portions of their shifts on indirect care tasks, leading to professional burnout and potential delays in risk identification. AURA-ED aims to:
- Automate Clinical Synthesis: Transform raw data (vitals, labs, history) into a human-readable brief.
- Improve Early Detection: Reliably flag patients at risk for critical outcomes like sepsis, stroke, or ICU transfer.
- Enhance Efficiency: Potential to reclaim clinical hours by providing an immediate, high-fidelity understanding of a patient's risk profile.

# Dataset Access: 
- This project utilizes the Stanford MultiModal Clinical Monitoring in the Emergency Department (MC-MED) dataset, which includes 118,385 adult ED visits.
Request Access: The dataset is hosted on PhysioNet. You must complete the required CITI training to gain access.
- MC-MED database is not provided with this repository and is **required** for this workflow. 
## Generating the Master Dataset:
To generate the `master_dataset.csv` required by `brief_generator_llama.py`, follow these steps based on the MC-MED processing pipeline:
The structure of this repository is detailed as follows:

- `benchmark_scripts/...` contains the scripts for benchmark dataset generation (master_data.csv).

**Master Dataset Workflow**
Before proceeding, download and set up the MC-MED repository locally. 

### 1. Benchmark Data Generation
~~~
python extract_master_dataset.py 
~~~

**Arguements**:
- `VISITS_PATH` : Path to the directory containing the patient's ED visit data.
- `MEDS_PATH ` : Path to the directory containing the patient's home medication records.
- `PMH_PATH` : Path to the directory containing the patient's past medical history.
- `LABS_PATH` : Path to the directory containing the patient's laboratory tests and results.
- `output_path` : Path to output directory

**Output**:
`master_dataset.csv` output to `output_path`

# Dataset
- Source: [https://physionet.org/content/mc-med/1.0.0/](https://physionet.org/content/mc-med/1.0.0/) (Publicly available)
- Composition: A total of **216** variables are included in `master_dataset.csv`
- Protocol: 80/20 train-test split.

# Inference Pipeline
To generate the Clinical Risk Briefs, the system utilizes a Retrieval-Augmented Generation (RAG) architecture.
- Employs a zero-shot chain-of-thought prompting strategy to force the model to weigh acute physiological instability against chronic disease burden.
- The engine integrates triage vitals, high-priority labs (Lactate, Troponin), and validated risk scores (NEWS2, CCI, ECI) into a constrained output schema.

~~~
python brief_generator_llama.py --input master_dataset.csv --model llama3.1:8b
~~~

# Model Performance
The models were evaluated across eight clinical outcomes using AUROC to account for extreme class imbalance in ED data.

# Limitations
Pilot Sample Size: While the initial plan involved the full 80/20 split, computational constraints (8-hour run times for 300 patients) limited the pilot evaluation to a representative sample of 300 ED encounters.

HIPAA Compliance: To protect patient privacy, all names are omitted; users must search via SubjectID.

Model Blind Spots: No AI should operate without physician oversight. For example, llama3.1:8b showed sub-chance performance for Acute Kidney Injury (AKI), indicating where human adjudication is irreplaceable.

# Future Work
Infrastructure: Migrating from local inference to a Virtual Private Server (VPS) or Cloud environment (e.g., AWS Bedrock) to achieve sub-second inference speeds.

Integration: Transitioning from a Streamlit prototype to a secure, FHIR-compliant hospital plugin.

Equity: Implementing real-time auditing for Algorithmic Equity to ensure consistent risk narratives across diverse demographics.

# References

