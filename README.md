Beyond Sensors: Interpretable Audio-Based Machine Learning for Real-Time Vehicle Fault and Emergency Sound Classification
Abstract
Unrecognized mechanical faults and emergency sounds in vehicles can compromise safety, particularly for individuals with hearing impairments and in sound-insulated or autonomous driving environments.
As intelligent transportation systems (ITS) evolve, there is a growing need for inclusive, non-intrusive, and real-time diagnostic solutions that enhance situational awareness and accessibility.
This study introduces an interpretable, sound-based machine-learning framework to detect vehicle faults and emergency sound events using acoustic signals as a scalable diagnostic source.
Three purpose-built datasets were developed: one for vehicular fault detection, another for emergency and environmental sounds, and a third integrating both to reflect real-world ITS acoustic scenarios. 
Audio data were preprocessed through normalization, resampling, and segmentation and transformed into numerical vectors using Mel-Frequency Cepstral Coefficients (MFCCs), Mel spectrograms, and Chroma features.
To ensure performance and interpretability, feature selection was conducted using SHAP (explainability), Boruta (relevance), and ANOVA (statistical significance). 
A two-phase experimental workflow was implemented: Phase 1 evaluated 15 classical models, identifying ensemble classifiers and multi-layer perceptrons (MLPs) as top performers; Phase 2 applied advanced feature selection to refine model accuracy and transparency.
Ensemble models such as Extra Trees, LightGBM, and XGBoost achieved over 91% accuracy and AUC scores exceeding 0.99. SHAP provided model transparency without performance loss, while ANOVA achieved high accuracy with fewer features.
The proposed framework supports accessible and real-time diagnostics for ITS, enabling auditory-to-visual translation of critical events and enhancing safety for people with disabilities. 
The system contributes to equitable mobility, proactive vehicle monitoring, and resilient urban transportation infrastructures by aligning with inclusive smart city development principles.
Keywords: Emergency Sound Recognition; Feature Selection; Intelligent Transportation Systems (ITS); Machine Learning; Sound Classification; Vehicle Fault Detection.
