# MAMR
An End-to-End Multi-Agent Syatem for Autonomous Mendelian Randomization Analysis

# Overview

This repository provides the reference implementation of the multi-agent system for automated Mendelian randomization (MR) analysis described in the main paper.

The system implements an end-to-end MR workflow by coordinating multiple large language model (LLM)-driven agents under a centralized workflow orchestrator. 
It automates research planning, GWAS data preparation, MR analysis, result interpretation, and manuscript drafting, while preserving transparency and reproducibility.

# System Design

The system follows a stage-based workflow corresponding to the six phases described in the paper:

Research plan formulation
GWAS data acquisition and preprocessing
MR analysis execution
Results interpretation
Scientific manuscript generation
Paper review

A central control module acts as a lightweight workflow orchestrator, maintaining global execution state, triggering agent roles at predefined stages, 
and passing structured intermediate artefacts between agents. All domain reasoning and code generation are performed by the respective LLM-driven agents.

# MR Knowledge Base (MRKB)

MR methods and computational resources are organized in a structured MR Knowledge Base (MRKB) using YAML specifications.
Each entry encodes:
Required inputs and outputs
Key assumptions and parameters
Associated R packages and functions
During execution, the system retrieves relevant MRKB entries in response to runtime errors to support error-driven code diagnosis and repair.

# Data Sources
For Experiment 2 (TraitMatcher evaluation), we use a curated benchmark dataset stored as trait_queries.xlsx in the data/ directory of the repository. 
This dataset contains 168 queries constructed from 24 target traits, with seven query variants per trait, covering standard expressions, synonyms, misspellings, 
measurement-related expressions, and Chinese expressions. No additional external datasets are required for this experiment.

# Intended Use
This software is designed as a research assistant framework.
Automatically generated results and manuscripts should be reviewed and validated by human researchers before publication.

