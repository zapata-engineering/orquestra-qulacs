# orquestra-qulacs

## What is it?

`orquestra-qulacs` is a [Zapata](https://www.zapatacomputing.com) library holding modules for integrating qulacs with [Orquestra](https://www.zapatacomputing.com/orquestra/).

## Installation

Even though it's intended to be used with Orquestra, `orquestra-qulacs` can be also used as a Python module.
Just run `pip install .` from the main directory.

## Usage

`orquestra-qulacs` is a Python module that exposes Qulacs's simulators as an [`QuantumSimulator`](https://github.com/zapatacomputing/orquestra-quantum/blob/main/src/orquestra/quantum/api/backend.py) compatible with [Orquestra Core framework](https://github.com/zapatacomputing/orquestra-core). It can be imported with:

```
from orquestra.integrations.qulacs.simulator import QulacsSimulator
```

In addition, it also provides converters that allow switching between `qulacs` circuits and those of `orquestra`.

The module can be used directly in Python or in an [Orquestra](https://www.orquestra.io) workflow.
For more details, see the [Orquestra Qulacs integration docs](http://docs.orquestra.io/other-resources/framework-integrations/qulacs/).

For more information regarding Orquestra and resources, please refer to the [Orquestra documentation](https://www.orquestra.io/docs).

## Development and contribution

You can find the development guidelines in the [`orquestra-quantum` repository](https://github.com/zapatacomputing/orquestra-quantum).
