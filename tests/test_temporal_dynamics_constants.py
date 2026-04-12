import unittest

from core import constants_eq
from core import hamiltonian_eq
from core import pulse_eq
from core import utils


class TemporalDynamicsConstantsTests(unittest.TestCase):
    def setUp(self) -> None:
        constants_eq.clear_cache()
        utils.store("env/temperature_k", 300.0)
        utils.store("env/velocity_fraction_c", 0.12)
        utils.store("env/flux_factor", 1.4)
        utils.store("env/strain_factor", 0.8)
        utils.store("env/field_strength", 2.5)
        utils.store("env/phase_radians", 0.0)
        utils.store("env/frequency_hz", 12.0)
        utils.store("env/wavelength_m", 0.75)
        utils.store("env/amplitude", 0.6)
        utils.store("env/zero_point_offset", 0.0)
        utils.store("env/vector_field", [1.0, 0.0, 0.0])
        utils.store("env/spin_vector", [1.0, 0.0, 0.0])
        utils.store("env/orientation_vector", [1.0, 0.0, 0.0])

    def test_effective_constants_include_temporal_dynamics_fields(self) -> None:
        eff = constants_eq.get_effective()
        for key in (
            "h_eff",
            "k_b_eff",
            "c_eff",
            "e_charge_eff",
            "mu_0_eff",
            "epsilon_0_eff",
            "temporal_relativity",
            "phase_alignment",
            "zero_point_overlap",
            "wave_path_speed",
            "entanglement_weight",
        ):
            self.assertIn(key, eff)
        self.assertGreater(eff["temporal_relativity"], 0.0)
        self.assertGreater(eff["wave_path_speed"], 0.0)

    def test_phase_alignment_changes_derived_constants(self) -> None:
        aligned = constants_eq.get_effective()
        constants_eq.clear_cache()
        utils.store("env/phase_radians", 3.1415926535)
        misaligned = constants_eq.get_effective()

        self.assertGreater(aligned["phase_alignment"], misaligned["phase_alignment"])
        self.assertGreater(aligned["entanglement_weight"], misaligned["entanglement_weight"])
        self.assertNotEqual(aligned["h_eff"], misaligned["h_eff"])

    def test_alignment_changes_temporal_weighting_and_core_runtime_values(self) -> None:
        aligned = constants_eq.get_effective()
        constants_eq.clear_cache()
        utils.store("env/orientation_vector", [0.0, 1.0, 0.0])
        orthogonal = constants_eq.get_effective()

        self.assertGreater(aligned["vector_alignment"], orthogonal["vector_alignment"])
        self.assertGreater(aligned["entanglement_weight"], orthogonal["entanglement_weight"])

        constants_eq.clear_cache()
        h_norm = hamiltonian_eq.hamiltonian_norm()
        group_velocity = pulse_eq.pulse_group_velocity()
        self.assertGreater(h_norm, 0.0)
        self.assertGreater(group_velocity, 0.0)


if __name__ == "__main__":
    unittest.main()
