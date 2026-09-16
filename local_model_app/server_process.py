from __future__ import annotations

import argparse

import uvicorn

from local_model_app.server import app


def main() -> None:
    parser = argparse.ArgumentParser(description="Private Local Model application service")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8765, type=int)
    arguments = parser.parse_args()

    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host=arguments.host,
            port=arguments.port,
            log_level="info",
        )
    )
    app.state.request_process_exit = lambda: setattr(server, "should_exit", True)
    server.run()


if __name__ == "__main__":
    main()
