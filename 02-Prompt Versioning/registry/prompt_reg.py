from prompts import PROMPTS


class Prompt_Register:

    def get_prompt(self, name: str, version: str):
        try:
            return PROMPTS[name][version]

        except KeyError:
            raise ValueError(f"No such prompt: {name} {version}")

    def render(self, name: str, version: str, variables: dict):

        data = self.get_prompt(name, version)

        prompt = data["template"]
        vars = data["variables"]

        for v in vars:

            if v not in variables:

                raise Exception("All variables are not present")

        return prompt.format(**variables)
