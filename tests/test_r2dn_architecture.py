import pytest

from r2dn_dc_motor.models.r2dn_adapter import R2DNArchitecture


def test_phase1_pilot_architecture_is_valid():
    architecture = R2DNArchitecture()

    architecture.validate()

    assert architecture.input_size == 3
    assert architecture.output_size == 2
    assert architecture.state_size == 4
    assert architecture.do_polar_param is True


def test_architecture_without_additional_latent_capacity_is_rejected():
    architecture = R2DNArchitecture(state_size=2)

    with pytest.raises(ValueError, match="larger than"):
        architecture.validate()
