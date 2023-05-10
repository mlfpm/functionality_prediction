# WHODAS 2.0 functionality prediction from digital biomarkers

Functional limitations are associated with poor clinical outcomes, higher mortality, and disability rates, especially in the elderly. Continuous assessment of patients’ functionality is important for clinical practice; however, traditional questionnaire-based assessment methods are very time-consuming and infrequently used. Mobile sensing offers a great range of sources that can assess function and disability daily.

This repository contains the baseline and follow-up approaches for predicting patient functionality using passively sensed digital biomarkers and socio-demographic information of psychiatric outpatients. 

## Baseline approach
This work aimed to prove the feasibility of an interpretable machine learning pipeline for predicting WHODAS 2.0 outcomes utilising solely passively collected digital biomarkers.
One-month long time-series data were summarised using statistical measures (minimum, maximum, mean, median, standard deviation, IQR), creating 64 features. A sequential feature selection method was then applied on each WHODAS 2.0 domain (cognition, mobility, self-care, getting along, life activities, participation). Finally, the WHODAS 2.0 functional domain scores were predicted using linear regression on the best feature subsets. 

* Work submitted for publication. Preprint available at: https://preprints.jmir.org/preprint/38231. Currently under review at JMIR Formative Research.

## DL-based approach
In our second approach, we defined a pipeline that performs feature encoding for the daily information by applying Time2Vec, followed by an LSTM encoder with attention for the 48-long half-hourly daily sequences, then another LSTM encoder with attention for the 30-day embedded input sequence. A feed-forward layer on top of the second LSTM's outputs concatenated with socio-demographic data is then used to get the predictions. Moreover, since the temporal data is regularly sampled but frequently missing, probabilistic generative models are used to perform data imputation. 

* Work submitted for publication. Currently under review at Internet Interventions.