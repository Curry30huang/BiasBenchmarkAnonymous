from typing import Union, Dict, Any
import enum

class StudyDesign(str, enum.Enum):
    RANDOMIZED_CONTROLLED_TRIAL = 'Randomized Controlled Trial'
    Observational_Study = 'Observational Study'
    Pre_specified_Subgroup_Analysis = 'Pre-specified Subgroup Analysis'
    Post_Hoc_Subgroup_Analysis = 'Post Hoc Subgroup Analysis'
    Other_Study_Design = 'Other Study Design'

class RCTRisk(str, enum.Enum):
    LOW_RISK = 'Low risk of bias'
    MODERATE_RISK = 'Some concerns'
    HIGH_RISK = 'High risk of bias'
