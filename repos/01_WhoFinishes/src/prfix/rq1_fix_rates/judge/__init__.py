"""LLM-as-judge for the RQ1 "does Q fix P?" follow-up-fix signal.

``build_cases`` builds per-pair context + the ``[P-overlap]`` attribution note
(``attribution``); ``prompt`` assembles RUBRIC + LEARNED INSIGHTS + worked-example
anchors; ``engine`` makes each model call; ``run_judge`` runs models at scale over
the net-diff population. Validation is separate: a blind human sample over a
population CSV (``human-annotation/sample_pairs.py judge``). See ``README.md``.
"""
