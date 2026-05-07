"""
Web Chat Platform Adapter for Hermes Agent.

A plugin-based gateway adapter that runs an aiohttp HTTP server providing:
- JWT-authenticated API for a React SPA frontend
- Session management with SQLite persistence
- Full markdown + LaTeX rendering support
- Streaming server-sent events for real-time responses
- Static file serving for the built SPA

Exposes the register() function required by the Hermes plugin loader.
"""

from .adapter import register

__all__ = ["register"]
