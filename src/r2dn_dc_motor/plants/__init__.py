"""Physical plant backends."""

from r2dn_dc_motor.plants.electrothermal import (
    ElectrothermalDCMotor,
    IntegrationSettings,
    MotorLimits,
    MotorParameters,
    MotorState,
    ResetRanges,
    Rollout,
    StepResult,
    TerminationReason,
    ThermalPowerBalance,
)

__all__ = [
    "ElectrothermalDCMotor",
    "IntegrationSettings",
    "MotorLimits",
    "MotorParameters",
    "MotorState",
    "ResetRanges",
    "Rollout",
    "StepResult",
    "TerminationReason",
    "ThermalPowerBalance",
]
