# AGENTS.md

## Purpose

`topolsurgery` is the shared, domain-neutral surgery layer for GROMACS
topologies. It exists so `marap-eptido`, `reaxgmx`, and future callers do
not reimplement connectivity edits, bonded-term repair, component
discovery, atom renumbering, and component export.

The repository owns mechanisms, not chemistry policy.

Current product boundary: a session accepts one `nrexcl` and rejects
system-level intermolecular or conditional molecule composition. Do not
describe this as full-system GROMACS support until provenance is implemented.

Keep these concerns in calling packages:

- reaction mapping and AK-atom classification;
- cap chemistry and junction registries;
- amide/aromatic improper rules;
- charge redistribution and atom retyping;
- force-field family selection;
- candidate geometry and reaction recognition.

## Read First

- `README.md` defines the public API and examples.
- `src/topolsurgery/graph.py` owns graph connectivity and graph queries.
- `src/topolsurgery/model.py` owns public immutable data contracts.
- `src/topolsurgery/surgery.py` owns stateful operations and rollback.
- `src/topolsurgery/_records.py` owns public-record conversion.
- `src/topolsurgery/export.py` owns component file output.

## Hard Invariants

1. A surgery atom number is 1-based and belongs to the flattened working
   topology. Sequence operations use the current post-operation numbering.
2. One `cut_bond` or `connect_bond` call changes exactly one connectivity
   edge. Batches belong in caller loops.
3. `[ bonds ]` and `[ constraints ]` are both connectivity. `[ settles ]`
   contributes protected implicit O-H connectivity. Never compute components
   from bonds alone.
4. Graph topology is the source of truth for angles, proper dihedrals, and
   graph-distance-three pairs.
5. A proper dihedral is valid only when all three consecutive edges exist.
   An angle is valid only when both incident edges exist.
6. Impropers are preserved when all referenced atoms remain in one
   connected component. Improper chemistry is a caller concern.
7. Removing atoms compacts survivors and renumbers every retained
   interaction and bond-order key. Survivor renumbering is not a structural
   delete/create in `TopologyEdit`.
8. Public edit methods are transactional. Any exception must restore the
   pre-operation topology, structure, parameter templates, and bond-order
   metadata.
9. Public results must not require callers to inspect `_top`, `_gro`, or
   other private fields. Use `topology`, `graph`, `edit`, `components`,
   `index_map`, and `export`.
10. `bond_order` is optional chemical metadata. It is not a GROMACS
    interaction function and must not alter `[ bonds ]` function columns.
11. A SETTLE group is cut as a unit only by caller policy; `cut_bond` must
    reject its implicit O-H edges. Removing any member removes the row.
12. Mixed moleculetype `nrexcl` values must not be flattened without a
    provenance-preserving merge policy.
13. CMAP rows require all five consecutive graph edges.
14. Unsupported or lossy sections must fail in `from_snapshot()` and
    component export rather than disappearing silently.

## Graph Rules

`AtomicGraph` is the one immutable graph class.

- Nodes may be arbitrary unique integer IDs in graph-only use.
- Surgical topology graphs use contiguous `1..N`.
- Edges are undirected and normalized to `(min, max)`.
- Duplicate edges collapse in first-appearance order.
- Self-loops and unknown endpoints are errors.
- Graph edits return a new graph.
- Bond-order metadata must refer to an existing graph edge.
- Nearest-neighbor and component queries must be deterministic.
- `pairs_at_distance(3)` means exact shortest graph distance.
- `proper_dihedrals_through_edge` enumerates simple three-bond paths that
  contain the requested edge.
- Do not add a second ad hoc adjacency implementation in surgery helpers or
  adapters. Extend `AtomicGraph` or use its public API.

## Public API

Stable imports live in `topolsurgery/__init__.py` and are mirrored in
`topolsurgery/api.py`. Keep both in sync.

The supported surfaces are:

- `TopologySurgery.from_files`, `from_gromologist`, `from_records`,
  `from_snapshot`, `from_result`;
- `TopologySurgery.graph`, `snapshot`, `atoms`, `interactions`, `components`,
  `bond_orders`, `clone`, `save`;
- `TopologySurgery.cut_bond`, `connect_bond`, `remove_atoms`;
- `AtomicGraph` queries and immutable graph transformations;
- `AtomSpec`, `AtomEdge`, `InteractionSpec`, `BondedTerm`,
  `ParameterRequest`, `GroFile`, `GroAtom`;
- `SurgeryResult`, `SurgeryWarning`, `AtomRemovalResult`,
  `TopologySnapshot`, `IndexMap`;
- `SurgeryResult.export`, `SurgeryResult.component_snapshots`, and the same
  methods on `AtomRemovalResult`.

Internal names beginning with `_` may change. Tests should exercise the
public API whenever practical.

Do not import `marap-eptido` or `reaxgmx` from library code. Provide generic
records and document a conversion layer instead of creating package cycles.

## Parameters

Generated interactions use this precedence:

1. exact `parameter_overrides`;
2. injected `parameter_resolver`;
3. explicit templates from the parsed topology;
4. parsed force-field tables;
5. an unparameterized row and a warning where applicable.

Resolver `None` or an empty sequence means "not handled" and continues
fallback. Preserve every returned multi-term Fourier row.

Never silently discard a mandatory bond because a resolver missed it.
Preserve A/B states through `BondedTerm.second_parameters`. Do not choose
among multiple same-type templates; require an exact match or caller policy.

## Testing

From a git checkout, `pixi` provides the pinned multi-version dev environments:

```bash
pixi run check          # format check, lint, tests
pixi run -e py314 test  # oldest/newest matrix locally
```

Without `pixi`, any environment with Python 3.10+, `gromologist`, and
`pytest` works:

```bash
python -m pip install -e '.[test]'
python -m pytest -q
```

Every behavior change needs a regression test. High-risk changes should
also cover:

- rings and non-bridge cuts;
- disconnected graphs and isolated atoms;
- repeated moleculetypes and repeated `[ molecules ]` entries;
- bonds plus constraints on the same connectivity path;
- SETTLE connectivity, cut refusal, atom removal, and `grompp`;
- mixed moleculetype `nrexcl` rejection;
- non-bridge CMAP invalidation;
- A/B parameters and ambiguous atom-type templates;
- improper functions 2 and 4;
- resolver exceptions and rollback;
- bond-order remapping after atom deletion;
- TOP/GRO identity and finite-value validation;
- unsupported/lossy section rejection;
- overwrite protection and CLI exit codes;
- sequential cut/connect operations;
- GRO/ITP/TOP export round trips through `gromologist`.

Prefer exact graph invariants over file-byte snapshots unless bytes are part
of the contract.

## Editing Discipline

- Use focused changes and preserve unrelated work.
- Add regression tests before or with the fix.
- Do not use private parser attributes outside this repository.
- Do not expose mutable internal mappings; return copies or read-only
  mappings.
- Keep the dependency surface small. `gromologist` is the parser/writer;
  no chemistry toolkit should become mandatory.
- Update `README.md` whenever a public signature or semantic changes.
