# Bladder Cancer Recurrence Dataset

## Overview

This repository contains the **Bladder Cancer Recurrence Dataset**, which provides detailed information on recurrences of bladder cancer. The dataset originates from a clinical trial comparing different treatments and is widely used for demonstrating methodologies in **recurrent event modeling** and **survival analysis**.

## Dataset Details

### Columns

| Column        | Description                                              |
|---------------|----------------------------------------------------------|
| **id**        | Unique identifier for each patient                       |
| **treatment** | Treatment received (placebo, pyridoxine B6, or thiotepa) |
| **number**    | Initial count of tumors                                  |
| **size**      | Size (cm) of largest initial tumor                       |
| **recur**     | Total number of recurrences observed                     |
| **start**     | Start time of observation interval                       |
| **stop**      | End time of interval (recurrence or censoring)           |
| **status**    | Event indicator at end of interval                       |
| **rtumor**    | Number of tumors at recurrence                           |
| **rsize**     | Size (cm) of largest tumor at recurrence                 |
| **enum**      | Sequential number of event or observation                |
| **rx**        | Numeric code for treatment                               |
| **event**     | Binary recurrence indicator                              |

### Dataset Variants

1. **Bladder**: Subset with 85 subjects (thiotepa or placebo arms only), up to 4 recurrences per patient.
2. **Bladder1**: Full dataset with 118 subjects and up to 9 recurrences, including all three treatments.
3. **Bladder2**: Reformatted version of Bladder for Anderson–Gill (AG) analysis.

## Applications

This dataset is ideal for:

- **Survival analysis** and **recurrent event modeling**
- Demonstrating **competing risks** and **multi-event** approaches
- Comparing statistical frameworks such as:
  - Wei–Lin–Weissfeld (WLW)
  - Anderson–Gill (AG)

## Usage

The dataset can be used for:

- Evaluating treatment efficacy for bladder cancer
- Exploring recurrence patterns and tumor progression
- Teaching datasets for biostatistics, epidemiology, and medical data analysis

## License

This dataset is released under **CC0: Public Domain** and is free to use for any purpose, including research, teaching, and publication.

## Next Steps

- Explore the dataset to understand its structure and characteristics.
- Plan analyses or visualizations to extract meaningful insights.
- Develop workflows for reproducible research.
