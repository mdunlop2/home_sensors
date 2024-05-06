import numpy as np
from onnxruntime import InferenceSession
from skl2onnx import to_onnx
from sklearn.datasets import load_diabetes
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

_FP_R_TOL = 1e6


def test_sklearn_to_onnx() -> None:
    """
    Example ported from
    https://onnx.ai/sklearn-onnx/auto_tutorial/plot_abegin_convert_pipeline.html

    Ensure that the ONNX converted model is close to the original SKLearn model.
    """
    X, y = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, _ = train_test_split(X, y)

    reg1 = GradientBoostingRegressor(random_state=1, n_estimators=5)
    reg2 = RandomForestRegressor(random_state=1, n_estimators=5)
    reg3 = LinearRegression()

    ereg = Pipeline(
        steps=[
            ("voting", VotingRegressor([("gb", reg1), ("rf", reg2), ("lr", reg3)])),
        ]
    )
    ereg.fit(X_train, y_train)

    onx = to_onnx(ereg, X_train[:1].astype(np.float32), target_opset=12)

    sess = InferenceSession(onx.SerializeToString(), providers=["CPUExecutionProvider"])
    pred_ort = sess.run(None, {"X": X_test.astype(np.float32)})[0]
    pred_skl = ereg.predict(X_test.astype(np.float32))

    assert np.all(np.isclose(pred_ort, pred_skl, rtol=_FP_R_TOL))
