"""MCP server exposing RHEED Studio's core analysis capabilities to AI agents.

Runs its own rheed_core.RheedSession, independent of any rheed_webapp
process — see the project's architecture notes for why (an agent may
chain in capabilities, like spot detection, that have nothing to do
with what a human has open in the browser).
"""
