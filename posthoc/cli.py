from __future__ import annotations

import click

from posthoc.commands._shared import setup_logging
from posthoc.commands.attribute import attribute
from posthoc.commands.baseline import baseline
from posthoc.commands.pal import pal
from posthoc.commands.simulate_pheno import simulate_pheno


@click.group()
@click.option("--verbose", is_flag=True, help="Enable debug logging.")
def main(verbose: bool) -> None:
    setup_logging(verbose)


main.add_command(attribute)
main.add_command(baseline)
main.add_command(simulate_pheno)
main.add_command(pal)

if __name__ == "__main__":
    main()
