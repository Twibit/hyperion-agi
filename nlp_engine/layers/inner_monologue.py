import numpy as np


class InnerMonologue:
    """
    System-2 deliberative reasoning loop (inner speech / private monologue).

    Rather than emitting the first thought that comes to mind (System-1), the
    brain runs a *chain of thought*: for each step it (1) retrieves relevant
    long-term knowledge, (2) generates a candidate sub-thought with its
    generative decoder, (3) metacognitively critiques that sub-thought for
    relevance + novelty, and (4) chains the accepted sub-thought into the next
    step's context. A final synthesis pass turns the accumulated narrative into
    an answer.

    This is the explicit deliberative layer the research synthesis calls for:
    a persistent internal verbal loop that offloads and guides executive
    control, supports analogical transfer via the knowledge graph, and tracks
    its own confidence so it can re-sample when a thought is poor.
    """

    def __init__(self, brain, max_steps: int = 4, sub_length: int = 24,
                 temperature: float = 0.6):
        self.brain = brain
        self.max_steps = max_steps
        self.sub_length = sub_length
        self.temperature = temperature
        self.history = []

    # ------------------------------------------------------------------
    def _retrieve(self, prompt):
        """Surface related long-term facts + analogous concepts."""
        facts = []
        try:
            toks = self.brain.thalamus.tokenize(prompt, 'prose')
            for t in toks:
                if len(t) > 3:
                    for triple in self.brain.knowledge_graph.get_related(t, max_results=2):
                        facts.append(f"{triple.subject} {triple.predicate} {triple.object}")
        except Exception:
            pass
        # De-duplicate while keeping order.
        seen, uniq = set(), []
        for f in facts:
            if f not in seen:
                seen.add(f)
                uniq.append(f)
        return uniq[:6]

    def _critique(self, sub_thought, prompt, facts):
        """Metacognitive score in [0, 1]: relevance to context + novelty vs history."""
        sub_toks = set(self.brain.thalamus.tokenize(sub_thought, 'prose'))
        if not sub_toks:
            return 0.0
        ctx = set(self.brain.thalamus.tokenize(prompt, 'prose'))
        for f in facts:
            ctx.update(self.brain.thalamus.tokenize(f, 'prose'))
        overlap = len(sub_toks & ctx)
        relevance = 0.5 * min(1.0, overlap / max(1, len(sub_toks)))

        nov = 1.0
        if self.history:
            prev = set()
            for h in self.history:
                prev |= set(self.brain.thalamus.tokenize(h, 'prose'))
            nov = 1.0 - min(1.0, len(sub_toks & prev) / max(1, len(sub_toks)))
        novelty = 0.5 * nov
        return relevance + novelty

    # ------------------------------------------------------------------
    def think(self, prompt, max_steps=None, return_trace=False):
        steps = max_steps or self.max_steps
        self.history = []
        facts = self._retrieve(prompt)
        context = prompt
        trace = []

        for i in range(steps):
            sub = self.brain._generate_semantic(
                seed_context=context, length=self.sub_length, temperature=self.temperature)
            # Keep only the *newly generated* continuation (strip the seed prefix)
            # so the chained context stays bounded instead of re-prepending.
            gen = sub[len(context):].strip() if sub.startswith(context) else sub.strip()
            crit = self._critique(gen, prompt, facts)
            # Self-correction: if the sub-thought is weak, re-sample with more
            # exploration before accepting it into the chain.
            if crit < 0.25:
                cand = self.brain._generate_semantic(
                    seed_context=context, length=self.sub_length,
                    temperature=min(1.5, self.temperature + 0.4))
                cand_gen = cand[len(context):].strip() if cand.startswith(context) else cand.strip()
                gen2 = self._critique(cand_gen, prompt, facts)
                if gen2 >= crit:
                    gen = cand_gen
                    crit = gen2
            self.history.append(gen)
            trace.append((i + 1, gen, round(crit, 2)))
            # Sliding window: each step conditions on the prompt + the *latest*
            # sub-thought only, so context cannot grow unbounded and loop.
            context = prompt + " " + gen

        synthesis = self.brain._generate_semantic(
            seed_context=prompt + " " + " ".join(self.history[-3:]),
            length=self.sub_length + 10, temperature=self.temperature * 0.8)

        if return_trace:
            lines = [f"  Step {s}: {t}   [critique={c}]" for s, t, c in trace]
            return synthesis, "\n".join(lines)
        return synthesis
