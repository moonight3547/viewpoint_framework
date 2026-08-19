#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generic framework utilities.

Currently contains:
    - output path utilities
    - lightweight local HTTP server
"""

import functools
import threading
import time
import webbrowser

from http.server import (
    SimpleHTTPRequestHandler,
    ThreadingHTTPServer,
)
from pathlib import Path
from urllib.parse import quote


# ==============================================================================
# Path utilities
# ==============================================================================

def prepare_html_output_path(
    output_path: str,
    default_filename: str = "index.html",
) -> Path:
    """
    Normalize an HTML output path.

    Examples
    --------

    Input:

        output_path = "vis/index.html"

    Output:

        .../vis/index.html


    Input:

        output_path = "vis"

    Output:

        .../vis/index.html

    Parent directories are automatically created.
    """

    output_path = Path(
        output_path
    ).expanduser().resolve()

    if output_path.suffix.lower() == ".html":

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    else:

        output_path.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path = (
            output_path
            / default_filename
        )

    return output_path


# ==============================================================================
# Local HTTP server
# ==============================================================================

class LocalHTTPServer(
    ThreadingHTTPServer
):
    """
    Lightweight local HTTP server.

    allow_reuse_address:
        Makes repeated debugging/restarting more convenient.

    daemon_threads:
        Active HTTP requests do not prevent Ctrl-C shutdown.
    """

    allow_reuse_address = True
    daemon_threads = True


class QuietHTTPRequestHandler(
    SimpleHTTPRequestHandler
):
    """
    SimpleHTTPRequestHandler without per-request console logs.
    """

    def log_message(
        self,
        format,
        *args,
    ):
        pass


def serve_html(
    html_path: str,
    host: str = "127.0.0.1",
    port: int = 8000,
    open_browser: bool = False,
) -> None:
    """
    Serve one generated HTML file using a lightweight HTTP server.

    This function blocks until Ctrl-C.

    Args:
        html_path:
            HTML file to serve.

        host:
            Bind address.

            Recommended default:
                127.0.0.1

        port:
            HTTP port.

            port=0:
                let the OS choose a free port.

        open_browser:
            Automatically open the default local browser.

            Usually keep False on remote GPU servers.
    """

    html_path = Path(
        html_path
    ).expanduser().resolve()

    if not html_path.exists():

        raise FileNotFoundError(
            f"HTML does not exist: {html_path}"
        )

    if not html_path.is_file():

        raise ValueError(
            f"HTML path is not a file: {html_path}"
        )

    directory = html_path.parent

    handler = functools.partial(
        QuietHTTPRequestHandler,
        directory=str(directory),
    )

    try:

        server = LocalHTTPServer(
            (host, port),
            handler,
        )

    except OSError as exc:

        raise RuntimeError(
            f"Failed to start HTTP server at "
            f"{host}:{port}. "
            "The port may already be in use."
        ) from exc

    actual_port = (
        server.server_address[1]
    )

    # --------------------------------------------------------------------------
    # Human-readable URL
    # --------------------------------------------------------------------------

    if host in {
        "127.0.0.1",
        "0.0.0.0",
        "::",
    }:

        display_host = "localhost"

    else:

        display_host = host

    filename = quote(
        html_path.name
    )

    url = (
        f"http://{display_host}:"
        f"{actual_port}/{filename}"
    )

    print()
    print(
        "=========================================================="
    )
    print(
        " Visualization Server"
    )
    print(
        "=========================================================="
    )
    print(
        f" HTML:\n"
        f"   {html_path}"
    )
    print()
    print(
        f" URL:\n"
        f"   {url}"
    )
    print()
    print(
        " Controls:"
    )
    print(
        "   Left drag     : rotate"
    )
    print(
        "   Right drag    : pan"
    )
    print(
        "   Mouse wheel   : zoom"
    )
    print(
        "   Click legend  : show / hide group"
    )
    print()
    print(
        " Press Ctrl-C to stop."
    )
    print(
        "=========================================================="
    )
    print()

    # --------------------------------------------------------------------------
    # Optional browser launch
    # --------------------------------------------------------------------------

    if open_browser:

        def _open():

            time.sleep(0.3)

            webbrowser.open(
                url
            )

        threading.Thread(
            target=_open,
            daemon=True,
        ).start()

    # --------------------------------------------------------------------------
    # Blocking server
    # --------------------------------------------------------------------------

    try:

        server.serve_forever(
            poll_interval=0.2
        )

    except KeyboardInterrupt:

        print()
        print(
            "[Server] Ctrl-C received."
        )

    finally:

        server.server_close()

        print(
            "[Server] Server stopped."
        )