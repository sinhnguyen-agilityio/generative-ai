def build_logic_messages(text):
    system_prompt = f"""
        You are a logical sequence solver.
        Analyze the sequence and identify the most likely rule.

        Rules:
        - Keep your answer under 40 tokens.
        - Do not explain your reasoning.
        - If uncertain, choose the most likely pattern.
        - Return only the following JSON.

        Output:
            The rule is (explaining content). The next is (value)
        """

    user_prompt = f"""
        Sequence:
        {text}
        """

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
