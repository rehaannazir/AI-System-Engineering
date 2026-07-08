import json


def validate_output(output: str, required_fields: list):

    if isinstance(output, str):
        data = json.loads(output)

    elif isinstance(output, dict):
        data = output

    else:
        raise ValueError("Output must be JSON string or dictionary")

    for field in required_fields:

        if field not in data:
            raise ValueError(f"Missing field: {field}")

    return True
