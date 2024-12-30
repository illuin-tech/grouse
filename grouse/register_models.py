import litellm


def register_models():
    litellm.register_model(
        {
            "vertex_ai/claude-3-5-sonnet-v2@20241022": {
                "max_tokens": 200_000,
                "input_cost_per_token": 0.000003,
                "output_cost_per_token": 0.000015,
            },
            "vertex_ai/gemini-1.0-pro": {
                "max_tokens": 128_000,
                "input_cost_per_token": 0.0000005,
                "output_cost_per_token": 0.0000015,
            },  # approximation given 1 token = 4 chars
            "fireworks_ai/llama-v3p1-8b-instruct": {
                "max_tokens": 16_384,
                "input_cost_per_token": 0.0000002,
                "output_cost_per_token": 0.0000002,
            },
            "fireworks_ai/llama-v3p1-70b-instruct": {
                "max_tokens": 16_384,
                "input_cost_per_token": 0.0000009,
                "output_cost_per_token": 0.0000009,
            },
            "fireworks_ai/llama-v3p1-405b-instruct": {
                "max_tokens": 16_384,
                "input_cost_per_token": 0.000003,
                "output_cost_per_token": 0.000003,
            },
            "fireworks_ai/mixtral-8x7b-instruct": {
                "max_tokens": 32_000,
                "input_cost_per_token": 0.0000005,
                "output_cost_per_token": 0.0000005,
            },
            "fireworks_ai/mixtral-8x22b-instruct": {
                "max_tokens": 64_000,
                "input_cost_per_token": 0.0000009,
                "output_cost_per_token": 0.0000009,
            },
            "fireworks_ai/llama-v3p3-70b-instruct": {
                "max_tokens": 32_000,
                "input_cost_per_token": 0.0000009,
                "output_cost_per_token": 0.0000009,
            },
        }
    )
