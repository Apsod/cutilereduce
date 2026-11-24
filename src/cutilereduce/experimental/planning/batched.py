"""Batched analytical evaluation, with lazy scalar views and finalist realization."""

from dataclasses import fields
from functools import cached_property

import numpy as np

from cutilereduce.core import symbolic as sym
from cutilereduce.experimental.ir.cost import OperationCost
from .parametric import parameterize, parameter_values


class DeferredImplementation:
    def __init__(self, problem, candidate):
        self.problem, self.candidate = problem, candidate

    @cached_property
    def value(self):
        return self.problem.target.realization_factory(self.problem, self.candidate)


def resolve_implementation(value):
    return value.value if isinstance(value, DeferredImplementation) else value


class CostView:
    """Small numeric row view; full stage/invocation objects are built on demand."""
    def __init__(self, batch, row):
        self.batch, self.row = batch, row

    @cached_property
    def materialized(self):
        return self.batch.template.subs({k: v[self.row].item() for k, v in self.batch.parameters.items()})

    @property
    def work(self):
        return OperationCost(**{f.name: self.batch.columns[f"work.{f.name}"][self.row].item()
                                for f in fields(OperationCost)})

    def stage_seconds(self, indices):
        return sum(float(self.batch.columns[f"stage.{i}"][self.row]) for i in indices)

    def __getattr__(self, name):
        if name in self.batch.columns:
            return self.batch.columns[name][self.row].item()
        return getattr(self.materialized, name)


class PreparedCosts:
    def __init__(self, problem, representative):
        self.template = parameterize(problem, representative)
        names = ("estimated_seconds", "forward_seconds", "backward_seconds", "saved_bytes",
                 "auxiliary_peak_bytes", "resident_bytes_upper_bound")
        self.names = (*names, *(f"work.{f.name}" for f in fields(OperationCost)),
                      *(f"stage.{i}" for i in range(len(self.template.stages))))
        expressions = [getattr(self.template, n) for n in names]
        expressions += [getattr(self.template.work, f.name) for f in fields(OperationCost)]
        expressions += [s.estimated_seconds for s in self.template.stages]
        self.program = sym.cse_many(expressions)

    def evaluate(self, problem, candidates):
        values = [parameter_values(problem, c) for c in candidates]
        parameters = {k: np.asarray([v[k] for v in values]) for k in values[0]}
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            outputs = self.program.evaluate(parameters)
        batch = NumericCosts(self.template, parameters,
            {name: np.broadcast_to(np.asarray(value, dtype=float), (len(candidates),)) for name, value in zip(self.names, outputs, strict=True)})
        return [CostView(batch, i) for i in range(len(candidates))]


class NumericCosts:
    def __init__(self, template, parameters, columns):
        self.template, self.parameters, self.columns = template, parameters, columns


def evaluate_candidates(problem, candidates, prepared, stats):
    """Keep row order and rejection records while evaluating structural groups."""
    from .analysis import CandidateEstimate
    groups, results = {}, [None] * len(candidates)
    for i, candidate in enumerate(candidates):
        try:
            problem.validate_candidate(candidate)
            key = problem.target.batch_key(problem, candidate)
            groups.setdefault(key, []).append((i, candidate))
        except (ValueError, NotImplementedError) as error:
            results[i] = CandidateEstimate(candidate, None, (str(error),))
    stats["groups"].update(groups)
    for key, rows in groups.items():
        if key not in prepared:
            try:
                prepared[key] = PreparedCosts(problem, rows[0][1])
            except (ValueError, NotImplementedError) as error:
                # Structural failures (e.g. unsupported atomic dtype) are shared.
                prepared[key] = str(error)
        evaluator = prepared[key]
        if isinstance(evaluator, str):
            for i, candidate in rows:
                results[i] = CandidateEstimate(candidate, None, (evaluator,))
            continue
        available = parameter_values(problem, rows[0][1])
        if any(str(symbol) not in available for symbol in evaluator.program.free_symbols):
            stats["scalar_fallback_candidates"] += len(rows)
            for i, candidate in rows:
                results[i] = problem.evaluate(candidate)
            continue
        costs = evaluator.evaluate(problem, [c for _, c in rows])
        for (i, candidate), cost in zip(rows, costs, strict=True):
            reasons = []
            for metric, limit in ((cost.resident_bytes_upper_bound, problem.target.max_resident_bytes),
                                  (cost.auxiliary_peak_bytes, problem.target.max_auxiliary_bytes)):
                if limit is not None and metric > limit:
                    reasons.append(f"resource estimate {metric:g} exceeds limit {limit:g}")
            results[i] = CandidateEstimate(candidate, cost, tuple(reasons), DeferredImplementation(problem, candidate))
    return results
