# ssrf_guard.py
"""Best-effort SSRF guard for user-supplied URLs (public links, remote PDF
imports). Resolves the hostname and rejects it if any resolved address is
private, loopback, link-local, or otherwise non-public. This does not pin
the resolved IP for the actual outbound request, so it does not fully
prevent DNS-rebinding attacks — it blocks the common case of a user
pointing the server at an internal service or the cloud metadata endpoint.
"""
import ipaddress
import socket
from urllib.parse import urlparse

from fastapi import HTTPException

_ALLOWED_SCHEMES = {"http", "https"}


def assert_public_url_safe(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in _ALLOWED_SCHEMES or not parsed.hostname:
        raise HTTPException(status_code=400, detail="Invalid or unsupported URL")

    try:
        addrs = socket.getaddrinfo(parsed.hostname, None)
    except socket.gaierror:
        raise HTTPException(status_code=400, detail="Could not resolve URL host")

    for _family, _type, _proto, _canonname, sockaddr in addrs:
        ip = sockaddr[0]
        try:
            addr = ipaddress.ip_address(ip)
        except ValueError:
            continue
        if (
            addr.is_private
            or addr.is_loopback
            or addr.is_link_local
            or addr.is_reserved
            or addr.is_multicast
            or addr.is_unspecified
        ):
            raise HTTPException(status_code=400, detail="URL resolves to a disallowed address")
