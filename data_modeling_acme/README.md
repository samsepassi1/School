# Data Modeling for ACME

Udacity Data Engineering with AWS project solution for ACME Corporation.

## Contents

- `ACME Data Modeling Project Starter.ipynb` — completed project notebook
- `data/` — provided CSV datasets used by the PostgreSQL section
- `utils/graphrag_chatbot.py` — optional GraphRAG helper provided by the workspace

## Project Work

The notebook includes:

1. PostgreSQL OLTP schema in 3NF for customers, products, purchases, and user ratings.
2. MongoDB document model for ACME 3D Printing using embedded recent purchases, embedded industry products, flexible product documents, and purchase references.
3. Neo4j graph model with customers, products, industries, categories, PURCHASED and ALSO_BOUGHT relationships, plus optional standout recommendation queries.

## Running

This project is intended to run in the Udacity Workspace because PostgreSQL, MongoDB, and Neo4j containers are preconfigured there. In the workspace:

1. Open the notebook.
2. Select Kernel → Restart & Run All.
3. Confirm all cells complete and outputs are saved.
4. Submit through Udacity.

## Verification

Static validation in this repository checks that the notebook has no remaining required TODO placeholders and includes the expected PostgreSQL, MongoDB, and Neo4j deliverables.
