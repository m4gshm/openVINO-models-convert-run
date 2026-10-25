"""FIM (fill-in-the-middle) benchmark for OpenVINO agent servers.

Compares code-completion models by running a fixed set of FIM tasks through
the OpenAI-compatible ``/v1/completions`` endpoint and scoring the generated
middle part against expected key fragments.
"""
