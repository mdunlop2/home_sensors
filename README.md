# Home Sensors

Build a simple model that can predict whether a household is single or multiple occupancy using data from motion sensors installed in the home.

## Usage

```
git clone git@github.com:mdunlop2/home_sensors.git
cd home_sensors
docker run -p 8888:8888 -v $PWD:/home/jovyan/work docker.io/mdunlop/home_sensors:latest jupyter lab --NotebookApp.token='' --NotebookApp.notebook_dir=/home/jovyan/work
```

You can connect to the Jupyter server at `http://127.0.0.1:8888/lab`

Notebooks contained in this project:

| Notebook             | Description                                                                                    |
|----------------------|------------------------------------------------------------------------------------------------|
| explore.ipynb        | First look with some data cleaning.                                                            |
| features.ipynb       | Transform to format suitable for prediction and add features.                                  |
| fit.ipynb            | Compare multiple classification models using feature selection and performance on unseen data. |
| onnx_inference.ipynb | Convert best model to ONNX format and perform predictions against the final unseen test set.   |

## Testing

### Unit Tests

Unit testing suite can be run with the following command:

```bash
make py-test
```

This testing suite covers data cleaning, transformation and feature engineering.

### Integration Tests

Covers the download of the raw sensor data and addition of new tables required for feature engineering.

```bash
make py-integration-test
```

## Code Style and Documentation

I used mypy, black, isort and pylint to enforce better standards. The suite for validating can be run with:

```bash
make format
```