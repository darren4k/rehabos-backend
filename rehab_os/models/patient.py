"""Patient context and EMR data models."""

from datetime import date
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class CareSetting(str, Enum):
    """Healthcare setting for the patient encounter."""

    INPATIENT = "inpatient"
    OUTPATIENT = "outpatient"
    HOME_HEALTH = "home_health"
    SNF = "snf"  # Skilled Nursing Facility
    IRF = "irf"  # Inpatient Rehab Facility
    ACUTE = "acute"
    TELEHEALTH = "telehealth"


class Discipline(str, Enum):
    """Rehabilitation discipline."""

    PT = "PT"  # Physical Therapy
    OT = "OT"  # Occupational Therapy
    SLP = "SLP"  # Speech-Language Pathology


class Vitals(BaseModel):
    """Patient vital signs."""

    heart_rate: Optional[int] = Field(None, ge=0, le=300, description="BPM")
    blood_pressure_systolic: Optional[int] = Field(None, ge=0, le=300)
    blood_pressure_diastolic: Optional[int] = Field(None, ge=0, le=200)
    respiratory_rate: Optional[int] = Field(None, ge=0, le=60)
    oxygen_saturation: Optional[float] = Field(None, ge=0, le=100, description="SpO2 %")
    temperature: Optional[float] = Field(None, description="Fahrenheit")
    pain_level: Optional[int] = Field(None, ge=0, le=10, description="0-10 scale")

    @property
    def blood_pressure(self) -> Optional[str]:
        """Format blood pressure as string."""
        if self.blood_pressure_systolic and self.blood_pressure_diastolic:
            return f"{self.blood_pressure_systolic}/{self.blood_pressure_diastolic}"
        return None


class FunctionalStatus(BaseModel):
    """Functional assessment scores."""

    # General mobility
    berg_balance: Optional[float] = Field(None, ge=0, le=56)
    timed_up_and_go: Optional[float] = Field(None, description="seconds")
    six_minute_walk: Optional[float] = Field(None, description="meters")
    gait_speed: Optional[float] = Field(None, description="m/s")

    # ADL/IADL
    fim_score: Optional[float] = Field(None, ge=18, le=126, description="FIM total")
    barthel_index: Optional[float] = Field(None, ge=0, le=100)
    katz_adl: Optional[int] = Field(None, ge=0, le=6)

    # Upper extremity
    grip_strength_left: Optional[float] = Field(None, description="kg")
    grip_strength_right: Optional[float] = Field(None, description="kg")
    nine_hole_peg_left: Optional[float] = Field(None, description="seconds")
    nine_hole_peg_right: Optional[float] = Field(None, description="seconds")

    # Cognition/Communication
    moca: Optional[float] = Field(None, ge=0, le=30)
    mmse: Optional[float] = Field(None, ge=0, le=30)
    nihss: Optional[float] = Field(None, ge=0, le=42)

    # Pain/Quality of Life
    oswestry: Optional[float] = Field(None, ge=0, le=100, description="ODI %")
    neck_disability_index: Optional[float] = Field(None, ge=0, le=100)
    dash_score: Optional[float] = Field(None, ge=0, le=100)

    # Custom scores
    custom_scores: dict[str, float] = Field(default_factory=dict)


class PatientContext(BaseModel):
    """Complete patient context for clinical reasoning."""

    # Demographics
    age: int = Field(..., ge=0, le=150)
    sex: str = Field(..., pattern="^(male|female|other)$")
    height_cm: Optional[float] = None
    weight_kg: Optional[float] = None

    # Clinical information
    chief_complaint: str = Field(..., description="Primary reason for referral")
    diagnosis: list[str] = Field(default_factory=list, description="Current diagnoses")
    icd_codes: list[str] = Field(default_factory=list)
    comorbidities: list[str] = Field(default_factory=list)
    medications: list[str] = Field(default_factory=list)
    allergies: list[str] = Field(default_factory=list)
    surgical_history: list[str] = Field(default_factory=list)
    precautions: list[str] = Field(default_factory=list, description="Weight bearing, activity")

    # Vitals and functional status
    vitals: Optional[Vitals] = None
    functional_status: Optional[FunctionalStatus] = None

    # Context
    setting: CareSetting = CareSetting.OUTPATIENT
    discipline: Discipline = Discipline.PT
    referral_date: Optional[date] = None
    days_since_onset: Optional[int] = None
    prior_level_of_function: Optional[str] = None

    # Additional notes
    subjective_notes: Optional[str] = None
    objective_findings: Optional[str] = None
    physician_orders: Optional[str] = None

    @property
    def bmi(self) -> Optional[float]:
        """Calculate BMI if height and weight available."""
        if self.height_cm and self.weight_kg:
            height_m = self.height_cm / 100
            return round(self.weight_kg / (height_m**2), 1)
        return None

    def summary(self) -> str:
        """Generate brief patient summary."""
        parts = [f"{self.age}yo {self.sex}"]
        if self.diagnosis:
            parts.append(f"with {', '.join(self.diagnosis[:3])}")
        if self.setting:
            parts.append(f"in {self.setting.value}")
        return " ".join(parts)
