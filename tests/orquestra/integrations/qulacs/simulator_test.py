################################################################################
# © Copyright 2021-2022 Zapata Computing Inc.
################################################################################
import numpy as np
import pytest
from orquestra.quantum.api.backend_test import (
    QuantumSimulatorGatesTest,
    QuantumSimulatorTests,
)
from orquestra.quantum.circuits import Circuit, H, MultiPhaseOperation, X

from orquestra.integrations.qulacs.simulator import QulacsSimulator


@pytest.fixture
def backend():
    return QulacsSimulator()


@pytest.fixture
def wf_simulator():
    return QulacsSimulator()


class TestQulacs(QuantumSimulatorTests):
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
        self, backend, circuit, target_wavefunction
    ):
        wavefunction = backend.get_wavefunction(circuit)

        np.testing.assert_almost_equal(wavefunction.amplitudes, target_wavefunction)

    def test_run_circuit_and_measure_works_with_multiphase_operator(self, backend):
        params = [-0.1, 0.3, -0.5, 0.7]
        circuit = Circuit([H(0), X(1), MultiPhaseOperation(params)])

        measurements = backend.run_circuit_and_measure(circuit, n_samples=1000)

        assert len(measurements.bitstrings) == 1000
        assert all(
            bitstring in [(0, 1), (1, 1)] for bitstring in measurements.bitstrings
        )


class TestQulacsGates(QuantumSimulatorGatesTest):
    pass
