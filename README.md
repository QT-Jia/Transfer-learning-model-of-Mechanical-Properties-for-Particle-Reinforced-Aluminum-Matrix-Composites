# Transfer-learning-model-of-Mechanical-Properties-for-Particle-Reinforced-Aluminum-Matrix-Composites
### By Qingtao Jia

## Update Log
- **2025-11-10**: Initial commits.
  - 1.Core code for the PAMCs-MP model (Pre-traIn,Fine tuning).
  - 2.Machine learning and grid-search code, including implementations for SVR, Random Forest (RF), Gradient Boosting (GB), and BPNN.
  - 3.Code for calculating Orowan strengthening and thermal mismatch strengthening, provided in the `Cal_CTE` file.

## Dependencies
Ensure you have the following installed:
- **Python**: 3.8.9
- **Libraries**:
  - pandas==2.3.3
  - numpy==20.3.4
  - scikit-learn==1.7.2
  - torch==2.10.0
  - openxyl==3.1.5
