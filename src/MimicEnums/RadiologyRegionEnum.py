from enum import Enum


class RadiologyRegionValue(str, Enum):
    """Venous = Lower extremity veins"""

    Chest = "Chest"
    Abdomen = "Abdomen"
    Head = "Head"
    Spine = "Spine"
    Venous = "Venous"
    Knee = "Knee"
    Neck = "Neck"
    Foot = "Foot"
    Shoulder = "Shoulder"
    Ankle = "Ankle"
    Wrist = "Wrist"
    Hand = "Hand"
    Hip = "Hip"
    Finger = "Finger"
    Femur = "Femur"
    Bone = "Bone"
    Scrotum = "Scrotum"
    Heel = "Heel"
    Thigh = "Thigh"
