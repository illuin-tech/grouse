import litellm


def register_models():
    litellm.register_model(
        {
            "fireworks_ai/accounts/fireworks/models/llama-v3p1-8b-instruct": {
                "max_tokens": 16_384,
                "input_cost_per_token": 0.0000002,
                "output_cost_per_token": 0.0000002,
            },
            "fireworks_ai/accounts/fireworks/models/llama-v3p1-70b-instruct": {
                "max_tokens": 16_384,
                "input_cost_per_token": 0.0000009,
                "output_cost_per_token": 0.0000009,
            },
        }
    )
