"""Set submodules to the correct refs and ensure the ClimaEarth environment
devs the correct packages after.

Lastly, export a YAML file with the submodule branches and commits for
traceability.

Process:
1. In ClimaCoupler.jl, checkout main, pull, checkout branch for investigation
   so we can commit our Julia manifest there.
   Update ClimaParams? Won't be necessary every time, but was the first time
   I tried.
2. In ClimaAtmos.jl, checkout main, pull, leave on main. In ClimaCoupler.jl
   dev ClimaAtmos.jl.
3. In ClimaCore.jl, checkout main, pull, leave on main. In ClimaCoupler.jl,
   dev ClimaCore.jl and commit manifest.
4. In ClimaCore.jl-mod, checkout main, pull, leave on main since we're not
   investigating a ClimaCore change.
5. In ClimaAtmos.jl-mod, checkout investigation branch, pull.
6. In ClimaCoupler.jl-mod, checkout the baseline investigation branch then
   a mod suffix branch from that. Then dev ClimaCore.jl-mod and
   ClimaAtmos.jl-mod. Commit and push.

TODO: Is this possible to generalize?
"""

from typing import Literal

RepoName = Literal["ClimaCore.jl", "ClimaAtmos.jl", "ClimaCoupler.jl"]


def main(
    investigation_name: str,
    repo_name: RepoName,
) -> None:
    pass
