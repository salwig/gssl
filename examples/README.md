# Examples

We here provide an example of how to apply the GSSL algorithms to the [Letter Recognition](https://archive.ics.uci.edu/dataset/59/letter+recognition) dataset.

## Requirements

Before running the examples, ensure that you have successfully installed [**GSSL**](../README.md#installation).

To verify that **GSSL** is installed correctly, you can run the following command:

```bash
python3 -c "import GSSL; print('GSSL is installed')"
```

## Run the Demo

To run the examples, execute the [`demo.py`](./demo.py) script:

```bash
python3 demo.py
```

This script applies all ten graph-based semi-supervised learning algorithms and reports the classification error rates of these algorithms.

### Dataset Information

The [Letter Recognition](https://archive.ics.uci.edu/dataset/59/letter+recognition) dataset is provided in the `data/` folder.

Integrating other datasets should be straightforward by following the structure provided in the [`demo.py`](./demo.py) example.
