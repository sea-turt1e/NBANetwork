# test_inference.py

import json
from pathlib import Path

from src.inference import input_fn, model_fn, output_fn, predict_fn


def test_inference():
    # load Model
    model_dir = Path("src/")
    model = model_fn(model_dir)

    # sample input
    sample_input = {"player1": "Luka Doncic_2018_3", "player2": "LeBron James_2003_1"}
    request_body = json.dumps(sample_input)
    request_content_type = "application/json"

    # example input_fn
    input_data = input_fn(request_body, request_content_type)

    # example predict_fn
    prediction = predict_fn(input_data, model)

    # response of output
    response = output_fn(prediction, "application/json")
    print("Prediction Response:", response)


if __name__ == "__main__":
    test_inference()
