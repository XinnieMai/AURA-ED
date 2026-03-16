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

# References

