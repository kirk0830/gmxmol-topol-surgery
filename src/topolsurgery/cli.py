"""Command-line interface for one bond operation and component export."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from topolsurgery.surgery import TopologySurgery


def build_parser() -> argparse.ArgumentParser:
    """Build the ``topolsurgery`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="topolsurgery",
        description=(
            "Cut or connect one GROMACS bond, find connected components, "
            "and write GRO plus ITP/TOP files."
        ),
    )
    subparsers = parser.add_subparsers(dest="operation", required=True)
    for operation in ("cut", "connect"):
        command = subparsers.add_parser(operation)
        command.add_argument("topology", type=Path)
        command.add_argument("structure", type=Path)
        command.add_argument("--atom-i", type=int, required=True)
        command.add_argument("--atom-j", type=int, required=True)
        command.add_argument("--outdir", type=Path, required=True)
        command.add_argument("--format", choices=("itp", "top"), default="itp")
        command.add_argument("--prefix", default="component")
        command.add_argument("--gmx-dir", type=Path)
        command.add_argument("--overwrite", action="store_true")
        command.add_argument("--allow-unresolved", action="store_true")
        command.add_argument("--no-structure-check", action="store_true")
        command.add_argument(
            "--no-resolve-parameters",
            action="store_true",
            help="Do not fill explicit parameters from parsed FF tables.",
        )
        if operation == "connect":
            command.add_argument("--bond-function", type=int, default=1)
            command.add_argument("--bond-order", type=int)
            command.add_argument("--angle-function", type=int)
            command.add_argument("--dihedral-function", type=int)
            command.add_argument(
                "--bond-parameter",
                action="append",
                default=[],
                help="Bond parameter column; repeat for each column.",
            )
            command.add_argument(
                "--update-pairs",
                action=argparse.BooleanOptionalAction,
                default=None,
            )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line surgery.

    Parameters
    ----------
    argv : Sequence[str], optional
        Arguments excluding the executable name.

    Returns
    -------
    int
        Process exit code.
    """
    args = build_parser().parse_args(argv)
    try:
        surgery = TopologySurgery.from_files(
            args.topology,
            args.structure,
            gmx_dir=args.gmx_dir,
            validate_structure=not args.no_structure_check,
        )
        if args.operation == "cut":
            result = surgery.cut_bond(args.atom_i, args.atom_j)
        else:
            result = surgery.connect_bond(
                args.atom_i,
                args.atom_j,
                bond_function=args.bond_function,
                bond_order=args.bond_order,
                bond_parameters=tuple(args.bond_parameter),
                angle_function=args.angle_function,
                dihedral_function=args.dihedral_function,
                update_pairs=args.update_pairs,
                resolve_parameters=not args.no_resolve_parameters,
            )
        for warning in result.warnings:
            print(f"warning: {warning}", file=sys.stderr)
        if result.warnings and not args.allow_unresolved:
            print(
                "refusing to export unresolved interactions; pass "
                "--allow-unresolved to override",
                file=sys.stderr,
            )
            return 2
        exported = result.export(
            args.outdir,
            format=args.format,
            prefix=args.prefix,
            overwrite=args.overwrite,
        )
    except (FileExistsError, OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        f"{result.operation} {result.atom_pair}: {result.component_count} component(s)"
    )
    for component in exported.components:
        print(
            f"{component.gro_path}\t{component.topology_path}\t"
            f"atoms={len(component.atom_numbers)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
