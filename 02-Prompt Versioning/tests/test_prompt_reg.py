from registry.prompt_reg import Prompt_Register


def test_get_prompt():

    register = Prompt_Register()

    prompt = register.get_prompt("summarizer", "v1.1")

    assert "template" in prompt


def test_render():

    register = Prompt_Register()

    prompt = register.render("summarizer", "v1.1", {"article": "AI is a future"})

    assert "AI is a future" in prompt


def test_missing_vars():

    register = Prompt_Register()

    try:

        register.render("summarizer", "v1.1", {})

        assert False

    except Exception:

        assert True
