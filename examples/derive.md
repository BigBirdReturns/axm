# Deriving new values with AXM

You can derive new values from existing nodes using the `axm derive` command. The command resolves operands by ID or by fuzzy label match and reports propagated confidence.

```bash
axm derive financials.axm --operator add revenue expenses
```

Example output:

```
Derivation: ADD
  07-01-01-0001: revenue = 200 (confidence 0.90)
  07-01-02-0002: expenses = 50 (confidence 0.70)

Result:
  value: 250
  confidence: 0.70
```

Temporal alignment information stored in the IR is surfaced through the query engine:

```python
from axm import load, query

program = load("financials.axm")
space = query(program)

revenue = space.first(label_contains="revenue")
print(space.temporal_alignments(revenue.id))
```
