################################################################################
# © Copyright 2021-2022 Zapata Computing Inc.
################################################################################
import numpy as np
import pytest
from orquestra.quantum.api.circuit_runner_contracts import (
    CIRCUIT_RUNNER_CONTRACTS,
    STRICT_CIRCUIT_RUNNER_CONTRACTS,
)
from orquestra.quantum.api.wavefunction_simulator_contracts import (
    simulator_contracts_for_tolerance,
    simulator_contracts_with_nontrivial_initial_state,
    simulator_gate_compatibility_contracts,
)
from orquestra.quantum.circuits import Circuit, H, MultiPhaseOperation, X

from orquestra.integrations.qulacs.simulator import QulacsSimulator


@pytest.fixture
def wf_simulator():
    return QulacsSimulator()


class TestQulacs:
    @pytest.mark.parametrize(
        "circuit, target_wavefunction",
        [
            (
                Circuit(
                    [
                        H(0),
                        H(1),
                        MultiPhaseOperation((-0.1, 0.3, -0.5, 0.7)),
                        X(0),
                        X(0),
                    ]
                ),
                np.exp(1j * np.array([-0.1, 0.3, -0.5, 0.7])) / 2,
            ),
            (
                Circuit(
                    [
                        H(0),
                        H(1),
                        MultiPhaseOperation((-0.1, 0.3, -0.5, 0.7)),
                        MultiPhaseOperation((-0.2, 0.1, -0.2, -0.3)),
                        X(0),
                        X(0),
                    ]
                ),
                np.exp(1j * np.array([-0.3, 0.4, -0.7, 0.4])) / 2,
            ),
            (
                Circuit(
                    [
                        MultiPhaseOperation((-0.1, 0.3, -0.5, 0.7)),
                    ]
                ),
                np.array([np.exp(-0.1j), 0, 0, 0]),
            ),
            (
                Circuit(
                    [
                        H(0),
                        MultiPhaseOperation((-0.1, 0.3, -0.5, 0.7)),
                    ]
                ),
                np.array([np.exp(-0.1j), 0, np.exp(-0.5j), 0]) / np.sqrt(2),
            ),
        ],
    )
    def test_get_wavefunction_works_with_multiphase_operator(
        self, wf_simulator, circuit, target_wavefunction
    ):
        wavefunction = wf_simulator.get_wavefunction(circuit)

        np.testing.assert_almost_equal(wavefunction.amplitudes, target_wavefunction)

    def test_run_circuit_and_measure_works_with_multiphase_operator(self, wf_simulator):
        params = [-0.1, 0.3, -0.5, 0.7]
        circuit = Circuit([H(0), X(1), MultiPhaseOperation(params)])

        measurements = wf_simulator.run_and_measure(circuit, n_samples=1000)

        assert len(measurements.bitstrings) == 1000
        assert all(
            bitstring in [(0, 1), (1, 1)] for bitstring in measurements.bitstrings
        )


@pytest.mark.parametrize("contract", CIRCUIT_RUNNER_CONTRACTS)
def test_qulacs_runner_fulfills_circuit_runner_contracts(wf_simulator, contract):
    assert contract(wf_simulator)


@pytest.mark.parametrize(
    "contract",
    simulator_contracts_for_tolerance()
    + simulator_contracts_with_nontrivial_initial_state(),
)
def test_qulacs_wf_simulator_fulfills_wf_simulator_contracts(wf_simulator, contract):
    assert contract(wf_simulator)


@pytest.mark.parametrize("contract", simulator_gate_compatibility_contracts())
def test_qulacs_simulator_uses_correct_gate_definitionscontract(wf_simulator, contract):
    assert contract(wf_simulator)
