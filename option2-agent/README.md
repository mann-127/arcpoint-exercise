# LLM-Based Routing Agent

Explainable routing using Claude for complex decision-making.

## Quick Start

```bash
export ANTHROPIC_API_KEY="your-key"
python3 option2-agent/agent.py
```

## Architecture

- **Context API** (`context_api.py`): REST API exposing system state
- **Agent** (`agent.py`): Claude-based decision maker
- **Prompts** (`prompts.py`): Structured system prompts

## Trade-off vs ML Router

| Aspect | ML Router (src/) | LLM Agent |
|--------|-----------------|-----------|
| Latency | <1ms | ~500ms |
| Interpretability | Feature importance | Full reasoning |
| Best for | 95% of requests | Edge cases, auditing |

## When to Use

- Complex scenarios requiring reasoning
- Regulatory auditing (need explanations)
- Novel failure modes
- Low-frequency, high-stakes requests

See main README.md for full context.
