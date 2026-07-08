from validators.validator import validate_output


def test_output():

    output = {
        "summary": "AI is the ultimate source of Everything",
        "keywords": ["AI", "ML", "DL"],
    }

    assert validate_output(output, ["summary", "keywords"])


def test_missing_field():

    output = {"summary": "Ai is not good"}

    try:

        validate_output(output, ["summary", "keywords"])

        assert False

    except ValueError:

        assert True


def test_invalid_field():

    output = {"what": "How are you broo really glad to connect"}

    try:
        validate_output(output, ["summary", "keywords"])

        assert False

    except ValueError:

        assert True
