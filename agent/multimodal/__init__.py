"""Multimodal input support for chat requests: images and audio alongside text.

The package is intentionally free of pipeline dependencies: it turns OpenAI style
message content parts into OpenVINO GenAI generation inputs (text prompt with media
tags plus decoded ``ov.Tensor`` media), so any controller can reuse it.
"""
