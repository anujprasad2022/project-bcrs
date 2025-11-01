# **Bladder Cancer Recurrence Dataset**

## **About**

This dataset provides detailed information on **recurrences of bladder cancer** and has been widely used to demonstrate methodologies for **recurrent event modelling** in survival analysis.
It originates from a **clinical trial** comparing different treatments and includes patient-level data tracking **initial tumour characteristics** and **subsequent recurrences** over time.

The structure supports multiple analytical frameworks, including:

- **Competing risks models**
- **Wei–Lin–Weissfeld (WLW) format**
- **Anderson–Gill (AG) time-to-event models**

---

## **Columns**

| Column        | Description                                              | Format / Codes                                                               |
| :------------ | :------------------------------------------------------- | :--------------------------------------------------------------------------- |
| **id**        | Unique identifier for each patient                       | Numeric                                                                      |
| **treatment** | Treatment received (placebo, pyridoxine B6, or thiotepa) | Text or coded (1 = placebo, 2 = thiotepa)                                    |
| **number**    | Initial count of tumours                                 | 8 = eight or more                                                            |
| **size**      | Size (cm) of largest initial tumour                      | Numeric                                                                      |
| **recur**     | Total number of recurrences observed                     | Integer                                                                      |
| **start**     | Start time of observation interval                       | Time                                                                         |
| **stop**      | End time of interval (recurrence or censoring)           | Time                                                                         |
| **status**    | Event indicator at end of interval                       | 0 = censored; 1 = recurrence; 2 = death (bladder); 3 = death (other/unknown) |
| **rtumor**    | Number of tumours at recurrence                          | Integer                                                                      |
| **rsize**     | Size (cm) of largest tumour at recurrence                | Numeric                                                                      |
| **enum**      | Sequential number of event or observation                | Integer                                                                      |
| **rx**        | Numeric code for treatment                               | 1 = placebo; 2 = thiotepa                                                    |
| **event**     | Binary recurrence indicator                              | 1 = recurrence; 0 = censored                                                 |

---

## **Dataset Variants**

### **1. Bladder**

- Contains 85 subjects (thiotepa or placebo arms only).
- Up to **four recurrences per patient**.
- **Status** = 1 for recurrence, 0 for all other outcomes.
- Structured in **Wei–Lin–Weissfeld (WLW)** competing risks format.

### **2. Bladder1**

- Full dataset from the original study.
- Includes **all three treatments** (placebo, pyridoxine B6, thiotepa).
- Covers **118 subjects** with up to **nine recurrences**.
- Represents the **complete follow-up** data.

### **3. Bladder2**

- Same subset of subjects as _Bladder_.
- Reformatted in **(start, stop]** intervals for **Anderson–Gill (AG)** analysis.
- Corrected version avoids the common error that causes **extra follow-up time** beyond the 4th recurrence (seen in some published analyses).

---

## **Distribution**

- Format: **CSV**
- Typical size: ~7 KB
- Common version: 340 records, 8 columns (85 patients, ≤ 4 recurrences each)
- Extended version: 118 patients (full clinical data)

---

## **Usage**

Ideal for:

- **Survival analysis** and **recurrent event modelling**
- Demonstrating **competing risks** and **multi-event** approaches
- Comparing statistical frameworks such as:

  - **Wei, Lin, and Weissfeld (WLW)**
  - **Anderson–Gill (AG)**

Applications include:

- Evaluating **treatment efficacy** for bladder cancer
- Exploring **recurrence patterns** and **tumour progression**
- Teaching datasets for **biostatistics**, **epidemiology**, and **medical data analysis**

---

## **Coverage**

- **Source:** Clinical trial on bladder cancer treatment
- **Treatments:** Placebo, Pyridoxine (B6), Thiotepa
- **Scope:** Tracks multiple recurrences across follow-up periods

---

## **License**

**CC0: Public Domain**
Free to use for any purpose, including research, teaching, and publication.

---

## **Who Can Use It**

- **Statisticians & Data Scientists:** To develop and test recurrent event models
- **Medical Researchers:** To assess treatment efficacy and recurrence risks
- **Students & Academics:** For courses on survival analysis or biostatistics
- **Healthcare Analysts:** For predictive oncology and outcomes research

---

## **Suggested Dataset Names**

- _Bladder Cancer Recurrence Study_
- _Clinical Trial Data for Bladder Cancer_
- _Recurrent Event Modelling: Bladder Cancer_
- _Survival Analysis Dataset for Cancer Recurrence_

---

## **Attributes**

- **Original Source:** Bladder Cancer Recurrence Study
- **Public Dataset:** [Kaggle – Bladder Cancer Recurrences](https://www.kaggle.com/datasets/utkarshx27/bladder-cancer-recurrences)
- **Usability:** 10.0
