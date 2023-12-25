################################################################################
# © Copyright 2021-2022 Zapata Computing Inc.
################################################################################
import numpy as np
import pytest
import sympy
from orquestra.quantum.api.circuit_runner_contracts import (
    CIRCUIT_RUNNER_CONTRACTS,
    STRICT_CIRCUIT_RUNNER_CONTRACTS,
)
from orquestra.quantum.api.wavefunction_simulator_contracts import (
    simulator_contracts_for_tolerance,
    simulator_contracts_with_nontrivial_initial_state,
    simulator_gate_compatibility_contracts,
)
from orquestra.quantum.circuits import (
    Circuit,
    CustomGateDefinition,
    H,
    MultiPhaseOperation,
    X,
)
from orquestra.quantum.circuits._builtin_gates import make_parametric_gate_prototype
from orquestra.quantum.circuits._matrices import (
    u3_matrix,
    xx_matrix,
    yy_matrix,
    zz_matrix,
)

from orquestra.integrations.qulacs.simulator import QulacsSimulator


def _su4(angles: np.ndarray) -> np.ndarray:
    """
    A simplified implementation of SU4 gates based on that from
    ```orquestra.qml.models.qcbm.gate_factories.su4_factories```
    for testing purposes, with jax.numpy replaced with numpy
    """
    angle_scaling = np.array([2, 1, 1, 2, 1, 1, 2, 2, 2, 2, 1, 1, 2, 1, 1])

    angles = angles * angle_scaling
    return (
        np.kron(u3_matrix(*angles[9:12]), u3_matrix(*angles[12:15]))
        .dot(
            np.dot(
                np.dot(zz_matrix(angles[8]), yy_matrix(angles[7])),
                xx_matrix(angles[6]),
            )
        )
        .dot(np.kron(u3_matrix(*angles[0:3]), u3_matrix(*angles[3:6])))
    )


SU4 = make_parametric_gate_prototype("SU4", _su4, 2)


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
            # Single-gate U4 Circuit Test (Checks numerical correctness)
            (
                Circuit(
                    [
                        CustomGateDefinition(
                            "U(4)",
                            sympy.Matrix(
                                [
                                    [
                                        -0.975009125641972,
                                        0.219974039490932,
                                        -0.0310830290906098,
                                        0.00157231277796265,
                                    ],
                                    [
                                        -0.0311767839440925,
                                        0.00321671111665351,
                                        0.997554769484509,
                                        -0.0624671336880612,
                                    ],
                                    [
                                        -0.00295223846610674,
                                        -0.0147889480463319,
                                        -0.0625351353074372,
                                        -0.997928819182913,
                                    ],
                                    [
                                        -0.2199465783908,
                                        -0.975388313306714,
                                        -0.00277200889962023,
                                        0.0152792959760415,
                                    ],
                                ]
                            ),
                            params_ordering=(),
                        )()(0, 1),
                    ]
                ),
                np.array(
                    [
                        -0.975009125641972,
                        -0.0311767839440925,
                        -0.00295223846610674,
                        -0.2199465783908,
                    ]
                ),
            ),
            # Two-gate U4 circuit test (verifies permutation correctness)
            (
                Circuit(
                    [
                        CustomGateDefinition(
                            "U(4)",
                            sympy.Matrix(
                                [
                                    [
                                        -0.975009125641972,
                                        0.219974039490932,
                                        -0.0310830290906098,
                                        0.00157231277796265,
                                    ],
                                    [
                                        -0.0311767839440925,
                                        0.00321671111665351,
                                        0.997554769484509,
                                        -0.0624671336880612,
                                    ],
                                    [
                                        -0.00295223846610674,
                                        -0.0147889480463319,
                                        -0.0625351353074372,
                                        -0.997928819182913,
                                    ],
                                    [
                                        -0.2199465783908,
                                        -0.975388313306714,
                                        -0.00277200889962023,
                                        0.0152792959760415,
                                    ],
                                ]
                            ),
                            params_ordering=(),
                        )()(1, 2),
                        CustomGateDefinition(
                            "U(4)",
                            sympy.Matrix(
                                [
                                    [
                                        -0.0459090904246442,
                                        0.0985391938322458,
                                        0.682694504945565,
                                        0.722572207888182,
                                    ],
                                    [
                                        -0.9529980327183,
                                        -0.272150538114841,
                                        0.0842896395204549,
                                        -0.103073230803954,
                                    ],
                                    [
                                        -0.288932367318481,
                                        0.705668218741528,
                                        -0.5235481638074,
                                        0.380062853154314,
                                    ],
                                    [
                                        0.0787730421347993,
                                        -0.646727513672376,
                                        -0.502713427074055,
                                        0.568170345170737,
                                    ],
                                ]
                            ),
                            params_ordering=(),
                        )()(0, 1),
                    ]
                ),
                np.array(
                    [
                        0.04447087,
                        -0.02024206,
                        0.92998523,
                        0.08956999,
                        0.27962839,
                        -0.14620133,
                        -0.07489514,
                        0.13978961,
                    ]
                ),
            ),
            (
                Circuit(
                    [
                        SU4(
                            np.array(
                                [
                                    0.46470114,
                                    0.02254167,
                                    0.90526399,
                                    0.83328487,
                                    0.81589856,
                                    0.17742145,
                                    0.09612701,
                                    0.69926428,
                                    0.86622599,
                                    0.02417199,
                                    0.3877851,
                                    0.95556799,
                                    0.41237115,
                                    0.33757767,
                                    0.14386267,
                                ]
                            )
                        )(0, 1)
                    ]
                ),
                np.array(
                    [
                        0.26800149 - 0.3167006j,
                        0.10799243 + 0.2658063j,
                        0.00769411 + 0.35803059j,
                        -0.42462006 + 0.6610698j,
                    ]
                ),
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
def test_qulacs_simulator_uses_correct_gate_definitions_contract(
    wf_simulator, contract
):
    assert contract(wf_simulator)
