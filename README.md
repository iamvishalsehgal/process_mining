# Process Mining Project (Group 9)

This repository contains the assignment of **Predictive Process Monitoring (PPM)**, made by group 9. The project uses process mining techniques to predict based on a production event log dataset.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)


## Overview

The notebook performs the following tasks:
- Preprocessing the event log data to prepare it for analysis.
- Transforming the dataset into a format compatible with PM4Py.
- Extracting useful features such as prefixes and time-based metrics.
- Implementing predictive process monitoring to forecast completion times and potential bottlenecks.

## Dataset

The analysis is based on a production dataset located in the Dataset folder.
The dataset contains information about production processes.

### Key Attributes:
- **Case ID**: Identifier for production cases.
- **Start and Completion Timestamps**: Temporal data for tracking event durations.
- **Activities and Resources**: Event-specific details (e.g., machines or operators).
- **Quantities Completed/Rejected**: Performance metrics for quality analysis.

## Installation

To run the notebook, ensure you have the following dependencies installed. You can install them using `pip`:
pip install pandas pm4py scikit-learn matplotlib numpy tqdm python-Levenshtein joblib

## Usage
Run the notebook inside an environment where all dependencies are installed. Then run all cells at once,
or one by one. 

