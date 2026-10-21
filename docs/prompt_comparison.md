# Prompt comparison

Versioned prompts live under `app/prompts/`: a system policy, customer-summary/payment/revenue tasks, and an intentionally weaker baseline. The same synthetic golden/security cases can be run through the explicit evaluation command. Structured schema constraints and deterministic claim checking apply independently of prompt wording.

`python -m scripts.evaluate --mode synthetic` measures only the deterministic synthetic provider and local fixtures. It is useful for regression of routing, tools, validation and rendering. It cannot establish that one prompt is better at reasoning or more injection-resistant on a paid model. No model-backed baseline-versus-grounded improvement, live latency/cost advantage or statistically meaningful prompt win is claimed without a separately recorded real-provider run.

See generated evaluation artifacts and `docs/requirements_matrix.md` for the distinction between implemented behavior, synthetic measurement and external-service gaps.
