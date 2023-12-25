################################################################################
# © Copyright 2021-2022 Zapata Computing Inc.
################################################################################
from typing import Callable

import numpy as np
import qulacs
from orquestra.quantum.circuits import Circuit, GateOperation
from qulacs import gate as qulacs_gate


def _identity(x):
    return x


def _negate(x):
    return -x


def _no_params(x):
    raise RuntimeError(
        "This gate isn't parametric, you shouldn't need to map its params"
    )


def _gate_factory_from_pauli_rotation(axes):
    def _factory(*args):
        qubit_indices = args[: len(axes)]
        params = args[len(axes) :]
        return qulacs_gate.PauliRotation(qubit_indices, axes, *params)

    return _factory


QULACS_GATE_FACTORY = Callable[..., qulacs.QuantumGateBase]

ORQUESTRA_TO_QULACS_GATES: dict[str, tuple[QULACS_GATE_FACTORY, Callable]] = {
    # 1-qubit, non-parametric
    "I": (qulacs_gate.Identity, _no_params),
    **{
        gate_name: (getattr(qulacs_gate, gate_name), _no_params)
        for gate_name in ["X", "Y", "Z", "H", "S", "T"]
    },
    # 1-qubit, parametric
    **{
        gate_name: (getattr(qulacs_gate, gate_name), _negate)
        for gate_name in ["RX", "RY", "RZ"]
    },
    "PHASE": (qulacs_gate.U1, _identity),
    # 2-qubit, non-parametric
    **{
        gate_name: (getattr(qulacs_gate, gate_name), _no_params)
        for gate_name in ["CNOT", "SWAP"]
    },
    # 2-qubit, parametric
    **{
        gate_name: (_gate_factory_from_pauli_rotation([ax, ax]), _negate)
        for ax, gate_name in enumerate(["XX", "YY", "ZZ"], start=1)
    },
}


def _make_cphase_gate(operation: GateOperation):
    matrix = np.diag([1.0, np.exp(1.0j * operation.gate.params[0])])  # type: ignore
    gate_to_add = qulacs_gate.DenseMatrix(
        operation.qubit_indices[1], matrix  # type: ignore
    )
    gate_to_add.add_control_qubit(operation.qubit_indices[0], 1)
    return gate_to_add


def _make_su4_gate(operation: GateOperation):
    q0, q1 = operation.qubit_indices

    # This scaling is copied verbatim from
    # orquestra.qml.models.qcbm.gate_factories.su4_factories
    angle_scaling = np.array([2, 1, 1, 2, 1, 1, 2, 2, 2, 2, 1, 1, 2, 1, 1])
    params = operation.params[0] * angle_scaling

    u3_factory, u3_param_transform = ORQUESTRA_TO_QULACS_GATES["U3"]
    xx_factory, xx_param_transform = ORQUESTRA_TO_QULACS_GATES["XX"]
    yy_factory, yy_param_transform = ORQUESTRA_TO_QULACS_GATES["YY"]
    zz_factory, zz_param_transform = ORQUESTRA_TO_QULACS_GATES["ZZ"]

    u3_factory(q0, *map(u3_param_transform, params[:3]))
    gates_to_add = [
        u3_factory(q0, *map(u3_param_transform, params[:3])),
        u3_factory(q1, *map(u3_param_transform, params[3:6])),
        ##
        xx_factory(q0, q1, *map(xx_param_transform, params[6])),
        yy_factory(q0, q1, *map(yy_param_transform, params[7])),
        zz_factory(q0, q1, *map(zz_param_transform, params[8])),
        ##
        u3_factory(q0, *map(u3_param_transform, params[9:12])),
        u3_factory(q1, *map(u3_param_transform, params[12:])),
    ]

    return qulacs_gate.merge(gate_list=gates_to_add)


GATE_SPECIAL_CASES = {"CPHASE": _make_cphase_gate, "SU4": _make_su4_gate}


def _qulacs_gate(operation: GateOperation):
    try:
        factory = GATE_SPECIAL_CASES[operation.gate.name]
        return factory(operation)
    except KeyError:
        pass

    try:
        qulacs_gate_factory, param_transform = ORQUESTRA_TO_QULACS_GATES[
            operation.gate.name
        ]
        return qulacs_gate_factory(
            *operation.qubit_indices, *map(param_transform, operation.params)
        )
    except KeyError:
        pass

    return _custom_qulacs_gate(operation)


def _custom_qulacs_gate(operation: GateOperation):
    matrix = operation.gate.matrix
    dense_matrix = np.array(matrix, dtype=complex)

    # Need to sandwich matrix between two SWAP gates
    if dense_matrix.shape == (4, 4):
        dense_matrix[:, [1, 2]] = dense_matrix[:, [2, 1]]
        dense_matrix[[1, 2], :] = dense_matrix[[2, 1], :]

    return qulacs_gate.DenseMatrix(
        list(operation.qubit_indices), dense_matrix  # type: ignore
    )


def convert_to_qulacs(circuit: Circuit) -> qulacs.QuantumCircuit:
    qulacs_circuit = qulacs.QuantumCircuit(circuit.n_qubits)
    for operation in circuit.operations:
        qulacs_circuit.add_gate(_qulacs_gate(operation))
    return qulacs_circuit
