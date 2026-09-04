from enum import Enum


class RadiologyModalityValue(str, Enum):
    """Imaging modalities including external like CT and MRI and internal like ERCP."""

    Radiograph = "Radiograph"
    CT = "CT"
    Ultrasound = "Ultrasound"
    MRI = "MRI"
    Mammogram = "Mammogram"
    CTU = "CTU"
    Fluoroscopy = "Fluoroscopy"
    Carotidultrasound = "Carotid ultrasound"
    Paracentesis = "Paracentesis"
    MRCP = "MRCP"
    UpperGISeries = "Upper GI Series"
    Drainage = "Drainage"
    MRE = "MRE"
    MRA = "MRA"
    ERCP = "ERCP"
    PTC = "PTC"
