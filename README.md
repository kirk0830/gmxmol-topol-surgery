<p align="center">
  <img src="assets/icon/topolsurgery-app-icon.svg" width="120" alt="topolsurgery icon">
</p>

<h1 align="center">topolsurgery</h1>

[![CI](https://github.com/kirk0830/gmxmol-topol-surgery/actions/workflows/ci.yml/badge.svg)](https://github.com/kirk0830/gmxmol-topol-surgery/actions/workflows/ci.yml)

`topolsurgery` is a graph-based surgery layer for GROMACS topologies. It
performs one connectivity edit at a time, repairs the bonded interaction
records implied by that edit, finds the resulting component ownership
groups, and exports each group as a matching `.gro` plus `.itp` or `.top`.

The package is intended to be the shared low-level engine for
`marap-eptido` and `reaxgmx`. Reaction mapping, cap chemistry, charge
redistribution, atom retyping, and force-field policy remain in those
packages. They can pass their atoms and records to this package and consume
the edited topology, component map, graph, and structured diff.

Current release boundary: one working topology must use one `nrexcl` value,
must not contain system-level intermolecular interactions, and must not
depend on conditional `[ molecules ]` entries. Standard protein+water+ion
systems with mixed exclusion depths must be split before entering this
version.

## Install

For local package development:

```bash
python -m pip install -e '.[test]'
pytest
```

The repository also checks in Pixi environments pinned across the supported
interpreters, so no virtual environment setup is needed:

```bash
pixi run check           # format check, lint, and tests
pixi run -e py314 test   # one specific interpreter
```

After this repository is published, consumers can install a pinned Git tag
without a local checkout:

```bash
python -m pip install \
  "topolsurgery @ git+https://github.com/kirk0830/gmxmol-topol-surgery.git@v0.5.0"
```

Pixi can consume the same Git revision as a PyPI Git dependency:

```bash
pixi add --pypi topolsurgery \
  --git https://github.com/kirk0830/gmxmol-topol-surgery.git \
  --tag v0.5.0
```

For integration against the moving default branch, replace `--tag v0.5.0`
with `--branch main`. Tagged revisions are preferred for reproducible
application environments. This Git dependency path avoids relying on local
editable package installation inside a Pixi environment.

The runtime dependency is `gromologist`. No chemistry toolkit or GROMACS
installation is required for topology-only operations.

## Quick Start

```python
from topolsurgery import TopologySurgery

surgery = TopologySurgery.from_files("single-nrexcl-system.top", "system.gro")

cut = surgery.cut_bond(12, 34)
assert cut.component_count == 2  # true when 12-34 is a bridge
cut_files = cut.export("cut", format="itp", prefix="fragment")

# Operations mutate the session and always use its current 1-based numbering.
joined = surgery.connect_bond(
    12,
    34,
    bond_parameters=(0.153, 251040.0),
    bond_order=1,
)
for warning in joined.warnings:
    print("warning:", warning)

# save() writes the complete edited topology; export() splits components.
combined_path = surgery.save("edited-combined.top")
joined_files = joined.export("joined", format="top", prefix="product")
```

`cut_bond` accepts one bond or constraint. `connect_bond` accepts one atom
pair that is not already connected. A failed operation restores the
topology, coordinates, parameter templates, and bond-order metadata to their
pre-operation state. Warnings do not raise from the Python API: callers must
inspect `result.warnings` before accepting an unparameterized topology.

## Public Objects

The stable imports are:

```python
from topolsurgery import (
    AtomEdge,
    AtomicGraph,
    AtomRemovalResult,
    AtomSpec,
    BondedTerm,
    Component,
    ExportResult,
    ExportedComponent,
    GroAtom,
    GroFile,
    IndexMap,
    InteractionKind,
    InteractionSpec,
    ParameterOverrides,
    ParameterRequest,
    ParameterResolver,
    ParameterValue,
    SurgeryResult,
    SurgeryWarning,
    TopologyEdit,
    TopologySnapshot,
    TopologySurgery,
)
```

The same stable names are available through `topolsurgery.api`.

## Session and API Operation Matrix

`TopologySurgery` is a mutable session. Its edit methods update that session
and return an immutable result snapshot. Use the result immediately for
inspection/export, or call `TopologySurgery.from_result(result)` to branch
from it. The next operation on the session uses the numbering left by the
previous operation.

| Goal | Public call | Result |
|---|---|---|
| Read TOP/GRO files | `TopologySurgery.from_files(topology, structure=None, ...)` | New session |
| Wrap an existing `gromologist.Top` | `TopologySurgery.from_gromologist(topology, structure=None, ...)` | New session |
| Build in memory | `TopologySurgery.from_records(atoms, interactions=(), ...)` | New session |
| Rebuild from public immutable data | `TopologySurgery.from_snapshot(snapshot, ...)` | New session |
| Branch from a completed edit | `TopologySurgery.from_result(result)` | Independent session |
| Inspect | `snapshot()`, `atoms`, `interactions(kind)`, `graph`, `components` | Immutable or defensive public data |
| Copy a session | `clone()` | Independent session at the same state |
| Cut one edge | `cut_bond(i, j)` | `SurgeryResult` |
| Connect one edge | `connect_bond(i, j, ...)` | `SurgeryResult` |
| Remove atoms | `remove_atoms(atom_numbers)` | `AtomRemovalResult` |
| Save combined topology | `save(path, format=None, molecule_name=None, overwrite=False)` | `pathlib.Path` |
| Split into component files | `result.export(outdir, ...)` | `ExportResult` |
| Split without writing files | `result.component_snapshots()` | `tuple[TopologySnapshot, ...]` |

The frequently used constructor arguments are:

| Argument | Meaning |
|---|---|
| `gmx_dir` | GROMACS share/include directory used while parsing includes. |
| `bond_orders` | Optional chemical graph metadata keyed by atom pairs; values are integers `1..4`. This is not the GROMACS bond function column. |
| `validate_structure` | Compare flattened TOP and GRO residue/atom names. Defaults to `True`. |
| `nrexcl` | Exclusion depth for `from_records`; defaults to `3`. |
| `coordinates`, `velocities`, `box` | Optional in-memory GRO data in nanometres. Coordinates are required for component/GRO export. |
| `parameter_sections` | Inline `[ atomtypes ]`, `[ bondtypes ]`, `[ angletypes ]`, and related force-field text for `from_records`. |

Do not inspect result/session fields beginning with `_`. In particular,
`TopologySurgery.to_gromologist()` returns a defensive clone for
`gromologist` interoperability, while normal integrations should consume
`snapshot()`, `topology`, `graph`, `components`, `edit`, and export objects.

## `connect_bond` Options

`connect_bond` accepts all of these keyword arguments:

| Keyword | Type/default | Meaning |
|---|---|---|
| `bond_function` | `int = 1` | GROMACS `[ bonds ]` function for the requested edge. |
| `bond_parameters` | sequence = `()` | Explicit bond parameter columns. Empty means resolve through overrides, resolver, topology templates, then force-field tables. |
| `generate_angles` | `bool = True` | Add missing angles whose two incident edges contain the new edge. |
| `generate_dihedrals` | `bool = True` | Add missing proper dihedrals whose three-edge path contains the new edge. |
| `angle_function` | `int \| None = None` | Generated angle function. `None` uses the most common existing angle function, then falls back to `1`. |
| `dihedral_function` | `int \| None = None` | Generated proper function. `None` uses the most common existing proper function, then falls back to `9`. |
| `bond_order` | `int \| None = None` | Optional graph metadata from `1..4`; independent of `bond_function`. |
| `update_pairs` | `bool \| None = None` | `None` refreshes pairs only if `[ pairs ]` already exists; `True` creates/refreshes; `False` leaves pairs unchanged. |
| `resolve_parameters` | `bool = True` | Attempt parsed force-field table lookup for missing generated parameters. |
| `parameter_resolver` | callable or `None` | Caller callback for every generated bond, angle, proper dihedral, and new pair. |
| `parameter_overrides` | mapping or `None` | Exact atom-sequence terms, checked before the callback. Keys may use either endpoint/orientation order. |

## Graph API

Bond and constraint records, plus the implicit O-H dependencies of
`[ settles ]`, form one undirected atomic graph:

```python
from topolsurgery import AtomEdge, AtomicGraph

graph = AtomicGraph.from_edges(
    6,
    (
        AtomEdge(1, 2, order=1),
        (2, 3),
        (3, 4),
        (4, 5),
    ),
)

assert graph.components == ((1, 2, 3, 4, 5), (6,))
assert graph.distance(1, 5) == 4
assert graph.vertices_at_distance(1, 3) == (4,)
assert graph.pairs_at_distance(3) == {(1, 4), (2, 5)}
assert graph.angles_through_edge((2, 3)) == {(1, 2, 3), (2, 3, 4)}
assert graph.proper_dihedrals_through_edge((2, 3)) == {(1, 2, 3, 4)}
```

Graph nodes may be any unique integer identifiers. A surgical topology uses
contiguous 1-based numbers. Graph edits are immutable and return a new graph:

```python
joined = graph.with_edge(5, 6, bond_order=1)
cut = graph.without_edge(3, 4)
renumbered = graph.remap({1: 1, 2: 2, 3: 3, 4: 4, 5: 6, 6: 5})
serialized = graph.to_dict()
restored = AtomicGraph.from_dict(serialized)
```

The graph exposes adjacency, components, distances, shortest paths,
cycle rank, distance-three pairs, angle candidates, proper-dihedral
candidates, subgraphs, remapping, and optional chemical bond-order metadata.
It does not infer chemistry from coordinates. Its edges are topology
connectivity. Use `edges` and `pairs_at_distance(3)`; there are no ambiguous
`bonds` or `pairs` aliases.

## In-Memory Topology API

External packages can avoid files entirely:

```python
from topolsurgery import AtomSpec, InteractionSpec, TopologySurgery

atoms = (
    AtomSpec(1, "c3", "C1", residue_name="LIG"),
    AtomSpec(2, "c3", "C2", residue_name="LIG"),
    AtomSpec(3, "c3", "C3", residue_name="LIG"),
)
interactions = (
    InteractionSpec("bonds", (1, 2), 1),
    InteractionSpec("bonds", (2, 3), 1),
)

surgery = TopologySurgery.from_records(
    atoms,
    interactions,
    name="LIG",
    coordinates=((0.0, 0.0, 0.0), (0.153, 0.0, 0.0), (0.306, 0.0, 0.0)),
)

graph = surgery.graph
snapshot = surgery.snapshot()
```

`from_records` also accepts iterables, `velocities`, a 3- or 9-value `box`,
`parameter_sections` containing parsed force-field tables, and
`bond_orders`. Atom numbers must be unique and contiguous from 1; input
record order may differ, because coordinate/velocity rows are paired by each
`AtomSpec.number` before topology construction.

The current session can be inspected without exposing `gromologist`:

```python
surgery.atom_count
surgery.atoms
surgery.interactions()
surgery.interactions("angles")
surgery.snapshot()
surgery.graph
surgery.components
surgery.bond_orders
```

Every completed result exposes `components`, `edit`, `warnings`,
`topology`, `graph`, `component_snapshots()`, `export()`, and
`component_count`. The private parser payload is not part of the public
contract.

`graph.components` is the chemical connectivity partition from bonds,
constraints, and implicit SETTLE edges. Session and result `components`,
plus `TopologySnapshot.component_groups`, are ownership partitions: they
start from the chemical graph and keep virtual sites with their host atoms.
For TIP4P, the oxygen and virtual site can therefore be separate chemical
components but still belong to one ownership group.

### Public Record Shapes

`AtomSpec` has this field order:

```python
AtomSpec(
    number,                    # contiguous 1..N
    atom_type,                 # one GROMACS token
    atom_name,                 # one GROMACS token
    residue_number=1,          # non-negative integer
    residue_name="MOL",        # one GROMACS token
    charge_group=None,
    charge=0.0,
    mass=0.0,
)
```

Atom type/name and residue name must be single non-empty tokens.
`charge_group` is a positive integer or `None`; charge and mass must be
finite, and mass cannot be negative.

`InteractionSpec` has this field order:

```python
InteractionSpec(
    kind,
    atoms,
    function=1,
    parameters=(),
    second_parameters=(),
)
```

Supported `kind` values are `bonds`, `pairs`, `angles`, `dihedrals`,
`pairs_nb`, `position_restraints`, `dihedral_restraints`,
`virtual_sites2`, `virtual_sites3`, `cmap`, `constraints`, `settles`, and
`exclusions`. Atom numbers must be unique within one row and must match that
kind's arity. Exclusion rows use `function=None`; other kinds require a
positive function. `second_parameters` represents state B and is accepted
only together with state A parameters. Parameter values are finite
integers/floats or single-token strings; NumPy integer/real scalars are
normalized to native Python values.

`from_snapshot(snapshot)` reconstructs a topology from `atoms` and
`interactions`; coordinates must be supplied separately when export is
needed. It rejects snapshots with `unsupported_sections` or
`lossy_features`. `from_result(result)` is preferred for branching because
it preserves the result's coordinates, parameter templates, and private
parser payload through the public factory.

## Atom Removal

`remove_atoms` accepts one or more 1-based atoms and compacts the survivors:

```python
removal_session = TopologySurgery.from_files(
    "single-nrexcl-system.top",
    "system.gro",
)
removal = removal_session.remove_atoms((7, 9))

old_to_new = removal.index_map
removed_atoms = removal.removed_atoms
removed_records = removal.edit.removed
remaining_components = removal.components
post_removal_graph = removal.graph
```

Every interaction referencing a removed atom is dropped. Surviving rows and
bond-order metadata are renumbered. Coordinates are selected and serialized
with fresh local numbering. The operation reports renumbered survivors as
survivors, not as a deletion followed by a synthetic insertion.

Removing every atom from a molecule is rejected. A caller that wants to
delete an empty product should remove that molecule from the surrounding
system model itself.

## Parameter Resolution

New bond, angle, proper-dihedral, and pair rows are resolved in this order:

1. Exact atom-sequence `parameter_overrides`.
2. The caller's `parameter_resolver`.
3. Explicit templates from the parsed topology.
4. A topology-table lookup for bonds, angles, and dihedrals.
5. An unparameterized row plus a warning where appropriate.

```python
from topolsurgery import BondedTerm, ParameterRequest


def resolver(request: ParameterRequest) -> list[BondedTerm] | None:
    if request.kind == "bonds" and request.atom_types == ("c3", "c3"):
        return [BondedTerm(1, (0.153, 251040.0))]
    if request.kind == "angles" and request.atom_types == ("c3", "c3", "c3"):
        return [BondedTerm(1, (109.5, 460.24))]
    if request.kind == "dihedrals":
        return [
            BondedTerm(9, (0.0, 0.20, 2)),
            BondedTerm(9, (180.0, 0.10, 3)),
        ]
    return None


result = surgery.connect_bond(
    1,
    3,
    parameter_resolver=resolver,
    parameter_overrides={(3, 1): BondedTerm(1, (0.154, 250000.0))},
    resolve_parameters=False,
)
```

This continues the three-atom in-memory session above: atoms `1` and `3`
are not directly connected, so connecting them is valid.
`result.warnings` is empty because both generated term types receive explicit
parameters.

`None` or an empty resolver result means "not handled", so fallback
continues. It never suppresses a mandatory graph-derived interaction.
Multiple `BondedTerm` values preserve multi-term Fourier torsions. Each
`BondedTerm` may carry `second_parameters` for A/B topology states.
Every `ParameterRequest` exposes `kind`, canonical `atoms`, corresponding
`atom_types`, and `default_function`. A resolver value may be one
`BondedTerm` or a sequence of them; an override value follows the same rule.
For multi-row function-9 dihedrals, every row must provide state B or every
row must omit it, and each state must have unique positive integral
multiplicities.

Atom-type templates are used only when the match is unique. Different rows
for the same type pattern are not guessed; they remain unresolved until an
override, resolver, or force-field table decides the term.

`resolve_parameters=True` additionally fills missing columns from parsed
`bondtypes`, `angletypes`, and `dihedraltypes`. Use `gmx_dir` when an input
topology depends on GROMACS include files. Newly generated interactions that
remain implicit are reported as structured `SurgeryWarning` records in
`result.warnings` regardless of that switch.

## Interaction Repair Rules

For `cut_bond(i, j)`:

- Remove all bond and constraint rows joining the pair.
- Reject cutting an implicit SETTLE O-H dependency.
- Use the post-cut graph as the source of truth.
- Remove angles lacking both incident edges.
- Remove proper dihedrals whose three consecutive edges no longer exist.
- Preserve impropers while all referenced atoms remain in one component.
- Require the full five-atom path for CMAP records.
- Remove graph-derived distance-three pairs that are no longer 1-4 pairs.
- Preserve unrelated explicit pair rows within a surviving component.

For `connect_bond(i, j)`:

- Add exactly the requested bond.
- Add every angle whose two edges include the new edge.
- Add every proper-dihedral path whose three edges include the new edge.
- Add graph-distance-three pairs when requested.
- Preserve existing propers and impropers.
- Do not invent cap chemistry, amide impropers, atom types, or charges.

`update_pairs=None` refreshes pairs only when a `[ pairs ]` section already
exists. Use `True` to create it or `False` to leave it unchanged.

Atom deletion removes a complete SETTLE group when any of its three implicit
members is removed. SETTLE therefore never survives with missing atoms.

## Result Objects and Warnings

A `SurgeryResult` represents the state immediately after one cut or connect:

| Field/property | Meaning |
|---|---|
| `operation` | `"cut"` or `"connect"`. |
| `atom_pair` | Canonical ascending 1-based atom pair in the working topology. |
| `components` | Ordered `Component` ownership groups. |
| `component_count` | Number of ownership groups. |
| `warnings` | Generated `SurgeryWarning` records; warnings do not imply an exception. |
| `edit.created`, `edit.removed` | Exact public `InteractionSpec` rows created/removed by the operation. |
| `topology` | Immutable complete `TopologySnapshot` for this result. |
| `graph` | Chemical `AtomicGraph` for this result. |
| `export(...)`, `component_snapshots()` | Component output helpers. |

Each `Component` exposes `index`, `molecule_name`, and ascending
`atom_numbers` in the original working numbering. `atom_numbers` in
`ExportResult.components` also remain in working numbering, while the
contents of each exported topology are renumbered locally from `1`.

An `AtomRemovalResult` additionally exposes:

| Field/property | Meaning |
|---|---|
| `requested_atoms` | Sorted removed atom numbers. |
| `removed_atoms` | Removed `AtomSpec` records. |
| `index_map.forward` | Old working number to compacted survivor number. |
| `index_map.backward` | Compacted survivor number back to old working number. |
| `edit` | Dropped/renumbered interaction provenance as public records. |

The Python API never converts unresolved generated parameters into an
exception. Inspect `result.warnings`; every unresolved parameter warning has
`code="unresolved_parameter"`, a `kind`, atom numbers, and `blocking=True`.
The CLI converts any warning into exit code `2` unless
`--allow-unresolved` is passed.

## Importing `gromologist`

```python
from topolsurgery import TopologySurgery

session = TopologySurgery.from_gromologist(parsed_top, gro_file)
parsed_copy = session.to_gromologist()
```

`from_files` and `from_gromologist` flatten the active `[ molecules ]`
composition into one working molecule while preserving system atom order.
Repeated copies of a moleculetype therefore use distinct global atom
numbers. Flattening is rejected when active moleculetypes have different
`nrexcl` values because one merged moleculetype cannot preserve both
exclusion regimes.

With coordinates attached, TOP/GRO residue names and atom names are checked
by default. Pass `validate_structure=False` only when the caller has already
proven a valid atom correspondence.

## Integrating `marap-eptido`

`marap-eptido` can convert its `MolecularTopology` records to `AtomSpec`
and `InteractionSpec`, shifting any 0-based chemistry indices to the 1-based
surface used by surgery:

```python
atoms = [
    AtomSpec(
        number=atom.num,
        atom_type=atom.ff_atomtype,
        atom_name=atom.name,
        residue_number=atom.resnum,
        residue_name=atom.res_name,
        charge_group=atom.num,
        charge=atom.charge,
        mass=0.0 if atom.mass is None else atom.mass,
    )
    for atom in molecular_topology.atoms
]

interactions = []
for bond in molecular_topology.bonds:
    interactions.append(InteractionSpec("bonds", (bond.i, bond.j), bond.func))
# Add angles, impropers, and pairs in the same way.
```

The caller keeps its family registry, cap stripping policy, charge merge,
and improper/completeness rules. It replaces its local bond/angle/dihedral
graph repair and numbering implementation with `TopologySurgery`.

## Integrating `reaxgmx`

Convert each `MoleculeTopology` at the reaction-topology boundary:

```python
atoms = tuple(
    AtomSpec(
        number=number,
        atom_type=data["type"],
        atom_name=data.get("atom", data["type"]),
        residue_number=data.get("resi", 1),
        residue_name=data.get("res", "MOL"),
        charge_group=data.get("cgnr", number),
        charge=data.get("charge", 0.0),
        mass=data.get("mass", 0.0),
    )
    for number, data in molecule.atoms.items()
)

surgery = TopologySurgery.from_records(
    atoms,
    existing_interactions,
    name=molecule.name,
    nrexcl=molecule.nrexcl,
)

product = surgery.connect_bond(
    ai,
    aj,
    bond_parameters=(r, k),
    bond_function=funct,
)
```

Reaction mapping, AK-atom selection, atomtype changes, and candidate
geometry remain in `reaxgmx`. Atom deletion, connectivity repair,
component discovery, and renumbered topology output come from this package.

## Downstream Integration Checklist

For `marap-eptido`, `reaxgmx`, or another caller, the normal integration
sequence is:

1. Convert package-native atoms and interactions to contiguous 1-based
   `AtomSpec`/`InteractionSpec` records, or pass a parsed
   `gromologist.Top`/file pair directly.
2. Construct one `TopologySurgery` session. Keep chemistry decisions such as
   atom typing, cap policy, charge redistribution, and improper rules in the
   caller.
3. Run exactly one of `cut_bond`, `connect_bond`, or `remove_atoms`.
4. Inspect `result.warnings`. Treat them as fatal unless the caller has an
   explicit downstream parameter policy.
5. Preserve `result.edit` if reaction provenance is required.
6. Either `session.save(...)` the combined topology or call
   `result.export(...)`/`result.component_snapshots()` to split ownership
   groups.
7. Translate between working and component-local atom numbers using the
   `Component.atom_numbers` order described above.

A strict integration gate can be as small as:

```python
result = session.connect_bond(
    12,
    34,
    bond_parameters=(0.153, 251040.0),
    bond_order=1,
)
if result.warnings:
    raise RuntimeError(
        "topology has unresolved parameters: "
        + "; ".join(str(warning) for warning in result.warnings)
    )

files = result.export("products", format="itp", prefix="product")
```

Use `session.clone()` before speculative edits and
`TopologySurgery.from_result(result)` to branch from a completed result.
Public integrations should never read `_top`, `_gro`, or other underscore
fields.

## Saving the Combined Topology

Use `save()` when the edited topology should remain one working molecule:

```python
top_path = surgery.save(
    "edited-system.top",
    format="top",          # inferable from the suffix
    overwrite=False,
)

itp_path = surgery.save(
    "edited-molecule.itp",
    molecule_name="PRODUCT",
    overwrite=True,
)
```

`format` may be `"itp"` or `"top"`; it is inferred from the target suffix
when omitted. ITP output writes the current moleculetype only. TOP output
requires a parsed `[ system ]` and `[ molecules ]`, so a molecule-only ITP
input cannot be saved as TOP. `molecule_name` renames only the written
moleculetype, not the current session.

`save()` writes the complete combined topology and returns a `pathlib.Path`.
It does not split components. With `overwrite=False`, an existing target is
never replaced, including when another process creates it concurrently.

## Export

```python
split_session = TopologySurgery.from_files(
    "single-nrexcl-system.top",
    "system.gro",
)
result = split_session.cut_bond(12, 34)
files = result.export("out", format="itp", prefix="fragment")
snapshots = result.component_snapshots()

for component in files.components:
    print(component.index)
    print(component.atom_numbers)
    print(component.gro_path)
    print(component.topology_path)
```

`result.export()` accepts:

| Keyword | Default | Meaning |
|---|---|---|
| `outdir` | required positional | Destination directory, created when absent. |
| `format` | `"itp"` | Write molecule ITP or complete TOP files. |
| `prefix` | `"component"` | File and generated molecule-name prefix. |
| `overwrite` | `False` | Replace an existing matching output set only when explicitly enabled. |

Output naming is deterministic:

- One ownership group writes `PREFIX.gro` and `PREFIX.itp`/`PREFIX.top`.
- Two or more groups write `PREFIX_001.gro`, `PREFIX_001.itp`/`.top`,
  `PREFIX_002...`, in `result.components` order.
- The exporter maps non-token prefix characters to `_` and uses the same
  stem as the written molecule name.

Each output topology has local atom numbering `1..N`, and each GRO is
selected in the same order. Export requires coordinates. A complete `.top`
also requires an input with `[ system ]` and `[ molecules ]`; molecule-only
ITP input can only produce ITP components.

`ExportResult` exposes `format`, `components`, `gro_paths`, and
`topology_paths`. To map a working atom number to an exported local number:

```python
for component in result.components:
    local_number = {
        working_number: local
        for local, working_number in enumerate(
            component.atom_numbers,
            start=1,
        )
    }
```

`component_snapshots()` performs the same split in memory. Each returned
`TopologySnapshot` has local atoms `1..N`, the component molecule name, and
local bond-order keys; no coordinate file is written.

Existing output files are never replaced unless `overwrite=True`. Files are
prepared in a staging directory and moved into place only after every
component has been serialized. No-clobber placement is atomic even if
another process creates a target after the initial preflight check.
Component export and component snapshots refuse unsupported sections or
lossy features such as conditional interaction rows and alchemical atom
state rather than silently dropping or flattening them. Raw combined
`save()` remains available when those features must stay in one file.

`TopologySurgery.from_result(result)` creates an independent session from
a completed result, including its bond orders and explicit parameter
templates, which is useful for branching workflows.

## CLI

After installation, the entry point is `topolsurgery`. In a source checkout
without installation, use `PYTHONPATH=src python -m topolsurgery` with the
same arguments.

```bash
topolsurgery --help
topolsurgery cut --help
topolsurgery connect --help
```

The command performs exactly one operation and always ends by exporting the
resulting component ownership groups. It does not call the combined
`save()` API.

```text
topolsurgery cut     TOPOLOGY STRUCTURE --atom-i N --atom-j N --outdir DIR [options]
topolsurgery connect TOPOLOGY STRUCTURE --atom-i N --atom-j N --outdir DIR [options]
```

Common positional arguments and options:

| Argument | Meaning |
|---|---|
| `TOPOLOGY` | Input `.top` or molecule `.itp` file. |
| `STRUCTURE` | Matching `.gro` file. The CLI always requires coordinates. |
| `--atom-i N`, `--atom-j N` | 1-based atoms in flattened `[ molecules ]`/GRO order. |
| `--outdir DIR` | Required component output directory. |
| `--format {itp,top}` | Component topology format; default `itp`. |
| `--prefix NAME` | Output file/molecule prefix; default `component`. |
| `--gmx-dir DIR` | GROMACS share/include directory for topology includes. |
| `--overwrite` | Allow replacement of an existing component output set. |
| `--allow-unresolved` | Export despite generated-parameter warnings. |
| `--no-structure-check` | Skip TOP/GRO residue and atom-name comparison; coordinates are still required. |
| `--no-resolve-parameters` | Do not look up missing parameters in parsed force-field tables. |

`connect` additionally supports:

| Option | Meaning |
|---|---|
| `--bond-function N` | Bond function column; default `1`. |
| `--bond-order N` | Graph bond-order metadata from `1..4`; unrelated to `--bond-function`. |
| `--angle-function N` | Function used for generated angles; omitted means follow existing topology/default `1`. |
| `--dihedral-function N` | Function used for generated propers; omitted means follow existing topology/default `9`. |
| `--bond-parameter VALUE` | One bond parameter column. Repeat once per column and preserve order. |
| `--update-pairs` | Refresh/create graph-distance-three pairs. |
| `--no-update-pairs` | Leave the existing pair policy alone. |

When neither pair flag is supplied, `update_pairs=None`: pairs are refreshed
only if the input already contains `[ pairs ]`. CLI `connect` does not expose
custom Python resolvers, `parameter_overrides`, or the
`generate_angles`/`generate_dihedrals` switches. Use the Python API for
those cases.

Cut example:

```bash
topolsurgery cut system.top system.gro \
  --atom-i 12 \
  --atom-j 34 \
  --outdir output/cut \
  --format itp \
  --prefix fragment
```

Connect example with explicit bond columns and pair regeneration:

```bash
topolsurgery connect system.top system.gro --atom-i 12 --atom-j 34 \
  --bond-function 1 \
  --bond-parameter 0.153 \
  --bond-parameter 251040 \
  --bond-order 1 \
  --angle-function 1 \
  --dihedral-function 9 \
  --update-pairs \
  --outdir output/joined \
  --format top \
  --prefix product
```

If table lookup is intentionally disabled and missing generated terms are
acceptable to the downstream workflow:

```bash
topolsurgery connect system.top system.gro \
  --atom-i 12 \
  --atom-j 34 \
  --outdir output/unresolved \
  --no-resolve-parameters \
  --allow-unresolved
```

Without `--allow-unresolved`, any warning is printed to stderr and the
command exits before creating the output directory set. With the flag, the
same warnings are printed and unparameterized rows are exported.

Output and exit behavior:

- One component writes `DIR/PREFIX.gro` plus `DIR/PREFIX.itp` or `.top`.
- Multiple components write `DIR/PREFIX_001.gro`, `PREFIX_001.itp/.top`,
  `PREFIX_002...`, matching `result.components` order.
- Success prints the operation, atom pair, component count, and each
  GRO/topology path to stdout, then exits `0`.
- Warnings, existing-output protection, invalid input, unsupported topology,
  and export failures print a concise error/warning to stderr and exit `2`.
- `--format top` requires an input containing `[ system ]` and
  `[ molecules ]`. A molecule-only ITP CLI input must use `--format itp`.

## Testing

```bash
PYTHONPATH=src python -m pytest -q
```

The `grompp` integration tests activate when `gmx` is on `PATH` or when
`TOPOLSURGERY_GMX` points to an executable:

```bash
TOPOLSURGERY_GMX=/path/to/gmx PYTHONPATH=src python -m pytest -q
```

## Scope and Safety

- Input atom numbering for surgery is always 1-based and in flattened GRO
  order.
- Sequence operations use the numbering produced by the preceding edit.
- SETTLE is connectivity but is not independently cuttable; rewrite the
  water model explicitly if a settle member must be severed.
- A cut changes interactions only when graph paths become invalid. It is not
  a chemical valence or charge sanitizer.
- Impropers are preserved at component granularity; chemistry-specific
  improper policy stays with the caller.
- Bond-order metadata is not a GROMACS function and is not written into
  `[ bonds ]`.
- Snapshots include supported restrained and virtual-site rows. An unsupported
  atom-addressed section, conditional entry, or alchemical atom state makes
  `from_snapshot()` fail instead of returning a lossy topology.
- `from_records` and export parse through `gromologist`; full MDP validation
  and `grompp` execution are outside this package.

## License

Apache License 2.0; see `LICENSE`.

The one runtime dependency, `gromologist`, is GPLv3-licensed. Review that
license before redistributing a work that links both packages.
