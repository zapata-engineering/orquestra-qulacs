################################################################################
# © Copyright 2020-2022 Zapata Computing Inc.
################################################################################
from functools import lru_cache
from typing import Any

import numpy as np
import qulacs
from orquestra.quantum.api.wavefunction_simulator import BaseWavefunctionSimulator
from orquestra.quantum.circuits import Circuit, GateOperation
from orquestra.quantum.measurements.expectation_values import ExpectationValues
from orquestra.quantum.operators import PauliRepresentation
from orquestra.quantum.typing import StateVector
from orquestra.quantum.wavefunction import flip_amplitudes
from qulacs.observable import create_observable_from_openfermion_text

from ..conversions import convert_to_qulacs


@lru_cache()
def get_qulacs_terms_from_orquestra_operator(qubit_operator: PauliRepresentation):
    """
    Convert an orquestra operator to a qulacs observable
    """
    qulacs_terms = []
    for term in qubit_operator.terms:
        # openfermion text is in the style of '2.0 [Z0 Z1]' if the operator is
        # QubitOperator("Z0 Z1", 2) aka PauliTerm("2*Z0*Z1")
        openfermion_terms_str = " ".join(
            [f"{pauli_str}{qubit_idx}" for qubit_idx, pauli_str in term.operations]
        )
        openfermion_str = f"{term.coefficient} [{openfermion_terms_str}]"
        qulacs_observable = create_observable_from_openfermion_text(openfermion_str)

        for term_id in range(qulacs_observable.get_term_count()):
            qulacs_terms.append(qulacs_observable.get_term(term_id))
    return qulacs_terms


class QulacsSimulator(BaseWavefunctionSimulator):
    """Wavefunction simulator using Qulacs."""

    supports_batching = False

    def __init__(self):
        super().__init__()

    def _get_qulacs_state(
        self, circuit: Circuit, initial_state=None
    ) -> qulacs.QuantumState:
        if initial_state is None:
            initial_state = np.array(
                [1] + (2**circuit.n_qubits - 1) * [0], dtype=np.int8
            )
        qulacs_state = qulacs.QuantumState(circuit.n_qubits)
        qulacs_state.load(flip_amplitudes(initial_state))  # type: ignore
        qulacs_circuit = convert_to_qulacs(circuit)
        qulacs_circuit.update_quantum_state(qulacs_state)
        return qulacs_state

    def _get_wavefunction_from_native_circuit(
        self, circuit: Circuit, initial_state: StateVector
    ) -> StateVector:
        return flip_amplitudes(
            self._get_qulacs_state(circuit, initial_state).get_vector()
        )

    def can_be_executed_natively(self, operation: Any) -> bool:
        return isinstance(operation, GateOperation)

    def get_exact_expectation_values(
        self, circuit: Circuit, qubit_operator: PauliRepresentation
    ) -> float:
        self._n_circuits_executed += 1
        self._n_jobs_executed += 1

        qulacs_state = self._get_qulacs_state(circuit)
        expectation_values = []
        for qulacs_term in get_qulacs_terms_from_orquestra_operator(qubit_operator):
            expectation_values.append(
                np.real(qulacs_term.get_expectation_value(qulacs_state))
            )
        return ExpectationValues(np.array(expectation_values)).values.sum()
