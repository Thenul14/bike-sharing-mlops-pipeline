# Portfolio preparation changes

- Replaced hardcoded database credentials, absolute Linux paths and personal contact details with environment configuration.
- Removed generated MLflow model binaries, run metadata, databases, derived feature CSV, notebook outputs and caches from the distributable archive. Retain the original ZIP privately for historical evidence; rotate old credentials if still in use.
- Preserved the original ingestion, cleaning, validation, star schema, feature engineering and model classes. Centralised feature order and unit conversion, including for Ridge.
- Replaced full-dataset model selection with training-fold cross-validation comparison. Held-out metrics remain separate. Model registration uses the `champion` alias rather than the deprecated stage API. Historical results are kept separately and do not describe a newly executed pipeline.
- Added local MariaDB/Redis Compose configuration, environment template, dependency files, API validation and a feature contract test. MariaDB is a portable substitute for the original university ColumnStore environment, not a claim that the original ColumnStore deployment is reproduced.
- XGBoost remains an optional training script, not part of the original three-model DAG.
- The original was a team project. Confirm teammate consent, academic rules and code ownership before publishing. No new code license is asserted.
