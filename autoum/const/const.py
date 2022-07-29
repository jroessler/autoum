"""
File containing constants
"""

############################## APPROACH NAMES ##############################

BCF_TITLE = "BCF"
CVT_TITLE = "CVT"
GRF_TITLE = "GRF"
LAIS_TITLE = "Lais"
RLEARNER_TITLE = "R_Learner"
SLEARNER_TITLE = "S_Learner"
TRADITIONAL_TITLE = "Traditional"
TREATMENT_DUMMY_TITLE = "Treatment_Dummy"
TWO_MODEL_TITLE = "Two_Model"
URF_TITLE = "URF"
URF_ED_TITLE = "URF_ED"
URF_KL_TITLE = "URF_KL"
URF_CHI_TITLE = "URF_CHI"
URF_DDP_TITLE = "URF_DDP"
URF_CTS_TITLE = "URF_CTS"
URF_IT_TITLE = "URF_IT"
URF_CIT_TITLE = "URF_CIT"
XLEARNER_TITLE = "X_Learner"

############################## COLOR SCHEMA ##############################
COLOR_SCHEMA = {
    BCF_TITLE: "#4daf4a",
    CVT_TITLE: "#984ea3",
    GRF_TITLE: "#00E307",
    LAIS_TITLE: "#6A0213",
    RLEARNER_TITLE: "#9400E6",
    SLEARNER_TITLE: "#003C86",
    TRADITIONAL_TITLE: "#e41a1c",
    TREATMENT_DUMMY_TITLE: "#FFDC3D",
    TWO_MODEL_TITLE: "#008607",
    URF_KL_TITLE: "#008169",
    URF_ED_TITLE: "#EF0096",
    URF_CHI_TITLE: "#00DCB5",
    URF_CTS_TITLE: "#FFCFE2",
    URF_DDP_TITLE: "#7CFFFA",
    URF_IT_TITLE: "#009FFA",
    URF_CIT_TITLE: "#68023F",
    XLEARNER_TITLE: "#F60239"
}

############################## GRAY COLOR SCHEMA ##############################
COLOR_SCHEMA_GRAY = {
    BCF_TITLE: "gray",
    CVT_TITLE: "silver",
    GRF_TITLE: "gainsboro",
    LAIS_TITLE: "whitesmoke",
    RLEARNER_TITLE: "darkgrey",
    SLEARNER_TITLE: "lightgray",
    TRADITIONAL_TITLE: "dimgray",
    TREATMENT_DUMMY_TITLE: "gray",
    TWO_MODEL_TITLE: "darkgray",
    URF_KL_TITLE: "#008169",
    URF_ED_TITLE: "#EF0096",
    URF_CHI_TITLE: "#00DCB5",
    URF_CTS_TITLE: "#FFCFE2",
    URF_DDP_TITLE: "#7CFFFA",
    URF_IT_TITLE: "#009FFA",
    URF_CIT_TITLE: "#68023F",
    XLEARNER_TITLE: "slategray"
}

############################## LINESTYLES ##############################
LINESTYLES = {
    BCF_TITLE: (0, (1, 1)),
    CVT_TITLE: (0, (1, 1)),
    GRF_TITLE: (0, (1, 1)),
    LAIS_TITLE: (0, (1, 1)),
    RLEARNER_TITLE: (0, (1, 1)),
    SLEARNER_TITLE: (0, (1, 10)),
    TRADITIONAL_TITLE: (0, (1, 1)),
    TREATMENT_DUMMY_TITLE: (0, (3, 1, 1, 1)),
    TWO_MODEL_TITLE: (0, (1, 1)),
    URF_KL_TITLE: (0, (1, 1)),
    URF_ED_TITLE: (0, (5, 10)),
    URF_CHI_TITLE: (0, (5, 1)),
    URF_CTS_TITLE: (0, (3, 10, 1, 10)),
    URF_DDP_TITLE: (0, (3, 5, 1, 5)),
    URF_IT_TITLE: (0, (1, 1)),
    URF_CIT_TITLE: (0, (1, 1)),
    XLEARNER_TITLE: (0, (1, 1))
}

############################## ABBREVIATIONS ##############################
NAMING_SCHEMA = {
    BCF_TITLE: "BCF",
    CVT_TITLE: "CVT",
    GRF_TITLE: "GRF",
    LAIS_TITLE: "LG",
    RLEARNER_TITLE: "RL",
    SLEARNER_TITLE: "SL",
    TRADITIONAL_TITLE: TRADITIONAL_TITLE,
    TREATMENT_DUMMY_TITLE: "TDA",
    TWO_MODEL_TITLE: "TM",
    URF_KL_TITLE: "U-KL",
    URF_ED_TITLE: "U-ED",
    URF_CHI_TITLE: "U-Chi",
    URF_CTS_TITLE: "CTS",
    URF_DDP_TITLE: "DDP",
    URF_IT_TITLE: "IT",
    URF_CIT_TITLE: "CIT",
    XLEARNER_TITLE: "XL"
}

############################## FOLDERS ##############################
RESULS = "/results/"
METRICS = RESULS + "metrics/"
FIGURES = RESULS + "figures/"
