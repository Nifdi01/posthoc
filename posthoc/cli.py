from __future__ import annotations

import click

from posthoc.commands.run import run
from posthoc.commands._shared import setup_logging


@click.group()
@click.option("--verbose", is_flag=True, help="Enable debug logging.")
def main(verbose: bool) -> None:
    setup_logging(verbose)


main.add_command(run)

if __name__ == "__main__":
    main()

