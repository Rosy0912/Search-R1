# Same-run port recovery

Independent copy of ed7f976. G_NEG failed at outer index9 before model load with EADDRINUSE; last committed state remains CP88/outer8. Preserve old source/logs, full sampling and teacher diagnostic. Resume original controller with the same run/config; original trainer automatically reuses the existing outer manifest and selects its saved state. No completed Adam is repeated.

Only communication launch uses static c10d rendezvous at an OS-selected free local port. Static backend reuses the agent TCPStore; four ranks and scientific settings are unchanged. A port is selected immediately for launch, not guaranteed permanently reserved against another process; genuine future failures remain explicit. No foreign signals.

The new one-time startup binds actual previous pane/controller exit1 and independent CP88 audit. Former ed7f976 startup must not be reused. Tests remain deferred. The prior batched evaluator still binds old supervisor1861244; update its terminal binding to the real replacement after training finishes, never fake exit0.
