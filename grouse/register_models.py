import litellm


def register_models():
    litellm.register_model(
        {
            "vertex_ai/claude-3-5-sonnet-v2@20241022": {
                "max_tokens": 200_000,
                "input_cost_per_token": 0.000003,
                "output_cost_per_token": 0.000015,
            },
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
