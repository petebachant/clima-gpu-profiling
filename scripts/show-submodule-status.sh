#!/usr/bin/env bash
# This script shows the status of the git submodules, which is useful for

echo "name                 branch                   HEAD       main      ahead behind  dirty"
echo "--------------------------------------------------------------------------------------"

git submodule foreach -q '
  branch=$(git symbolic-ref --short -q HEAD || echo detached)
  head=$(git rev-parse --short HEAD)
  main=$(git rev-parse --short origin/main 2>/dev/null || echo n/a)

  ab=$(git rev-list --left-right --count HEAD...origin/main 2>/dev/null || echo "n/a n/a")
  ab=$(printf "%s" "$ab" | tr "\t" " ")
  set -- $ab
  ahead=$1
  behind=$2

  dirty=$(test -n "$(git status --porcelain)" && echo dirty || echo clean)
  printf "%-20s %-24s %-10s %-10s %5s %6s %s\n" "$name" "$branch" "$head" "$main" "$ahead" "$behind" "$dirty"
'
