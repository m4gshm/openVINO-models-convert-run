import logging
from typing import Callable

from fastapi.routing import APIRoute
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse


class LoggingRoute(APIRoute):

    def get_route_handler(self) -> Callable:
        log_http = logging.getLogger("http")

        original_route_handler = super().get_route_handler()

        async def custom_route_handler(request: Request) -> Response:
            body_bytes = await request.body()
            request_body_str = body_bytes.decode("utf-8") if body_bytes else None
            log_http.debug(f"--> inbound {request.method} {request.url.path} body {request_body_str}")

            response: Response = await original_route_handler(request)

            if isinstance(response, StreamingResponse):
                log_http.debug(f"<-- outbound {response.status_code}, media-type {response.media_type}")
                old_iterator = response.body_iterator

                async def re_iterator():
                    async for chunk in old_iterator:
                        if isinstance(chunk, str):
                            chunk_bytes = chunk.encode("utf-8")
                        else:
                            chunk_bytes = chunk

                        res_body_str = chunk_bytes.decode("utf-8", errors="ignore")

                        log_http.debug(f"<-- outbound body chunk {res_body_str}")
                        yield chunk

                response.body_iterator = re_iterator()
            else:
                res_body_str = response.body.decode("utf-8", errors="ignore") if response.body else None
                log_http.debug(
                    f"<-- outbound {response.status_code}, media-type {response.media_type}, body {res_body_str}")

            return response

        return custom_route_handler
