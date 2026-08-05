"""smart-router-proxy — OpenAI-compatible proxy with task-aware model routing.

Fully standalone: the routing core (deterministic classifier, task enums,
route table, alias mappings) is vendored into this package so the proxy has
no runtime dependency on the hermes-smart-router plugin package.
"""

__version__ = "0.2.0"
