"""
A simplified implementation of SU4 gates based on that from
```orquestra.qml.models.qcbm.gate_factories.su4_factories```
for testing purposes, with jax.numpy replaced with numpy
"""

import numpy as np

from orquestra.quantum.circuits._builtin_gates import make_parametric_gate_prototype
from orquestra.quantum.circuits._matrices import (
    u3_matrix,
    xx_matrix,
    yy_matrix,
    zz_matrix,
)


angle_scaling = np.array([2, 1, 1, 2, 1, 1, 2, 2, 2, 2, 1, 1, 2, 1, 1])


def su4(angles: np.ndarray) -> np.ndarray:
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


SU4 = make_parametric_gate_prototype("SU4", su4, 2)
