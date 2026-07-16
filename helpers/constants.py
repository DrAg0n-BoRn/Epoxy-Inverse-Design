TARGET = "Tensile Strength(MPa)"

CURING = "Curing" # categorical
FILLER = "Filler" # categorical 
EPOXY_CURING_RATIO = "Epoxy/Curing Ratio" # continuous
FILLER_PROPORTION = "Filler Proportion(%)" # continuous 
TEMPERATURE = "Temperature(K)" # continuous




# For template
ALLOWED_RANGES = {
    EPOXY_CURING_RATIO: (0,10),
    FILLER_PROPORTION: (0,30),
    TEMPERATURE: (295,450),
}


TARGET_RANGE = (70, 90)
TARGET_UNIT = "MPa"