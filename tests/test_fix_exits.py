"""Tests for fix_exits.py.

`fix_exits.py` is the deterministic half of the pipeline -- it has no LLM and
no network, so the same input must always produce the same output. Re-running a
stage and getting a different map is a bug, not a variation.
"""

import json
import subprocess
import sys

import pytest

sys.path.insert(0, '.')
from fix_exits import connect_disconnected_rooms as connect_subgraphs


@pytest.fixture
def disconnected_rooms():
    """Three mutually disconnected clusters, each with several rooms.

    Several rooms per cluster matters: with one room each there is only one
    representative to pick, and a set-ordering bug cannot show itself.
    """
    def cluster(prefix, peers):
        return [
            {
                "name": f"{prefix}{i}",
                "description": "d",
                "exits": [f"{prefix}{j}" for j in peers if j != i],
            }
            for i in peers
        ]

    return {
        "rooms": cluster("A", [1, 2, 3]) + cluster("B", [1, 2, 3]) + cluster("C", [1, 2, 3]),
    }


def _exit_signature(data):
    return sorted(
        (room["name"], tuple(sorted(room.get("exits", []))))
        for room in data["rooms"]
    )


class TestConnectSubgraphsDeterminism:
    def test_repeated_runs_agree_in_process(self, disconnected_rooms):
        """Same input, same output -- within a single interpreter."""
        first = _exit_signature(connect_subgraphs(json.loads(json.dumps(disconnected_rooms))))
        for _ in range(5):
            again = _exit_signature(connect_subgraphs(json.loads(json.dumps(disconnected_rooms))))
            assert again == first

    def test_output_is_stable_across_hash_seeds(self, tmp_path, disconnected_rooms):
        """The real test: Python randomises string hashing per process.

        Iterating a set of room names therefore yields a different order in
        each run, so a set-order dependency only shows up across processes.
        """
        source = tmp_path / "rooms.json"
        source.write_text(json.dumps(disconnected_rooms))

        script = (
            "import json, sys;"
            "sys.path.insert(0, '.');"
            "from fix_exits import connect_disconnected_rooms as connect_subgraphs;"
            "d = json.load(open(sys.argv[1]));"
            "r = connect_subgraphs(d);"
            "print(json.dumps(sorted((x['name'], tuple(sorted(x.get('exits', [])))) "
            "for x in r['rooms'])))"
        )

        outputs = set()
        for seed in ("0", "1", "42", "12345"):
            proc = subprocess.run(
                [sys.executable, "-c", script, str(source)],
                capture_output=True, text=True, timeout=60, check=False,
                env={"PYTHONHASHSEED": seed, "PATH": "/usr/bin:/bin"},
            )
            assert proc.returncode == 0, proc.stderr
            outputs.add(proc.stdout.strip().splitlines()[-1])

        assert len(outputs) == 1, (
            f"connect_subgraphs produced {len(outputs)} different maps across "
            "hash seeds; it depends on set iteration order"
        )


class TestFullPipelineDeterminism:
    """The whole repair chain, on real data, across hash seeds.

    The subgraph fix alone was not enough: the normalized-name lookup was also
    built by iterating a set, which decided fuzzy-match tie-breaks. Only a
    full-pipeline check on a realistic map catches that, so this test runs what
    the CLI runs.
    """

    def test_repair_chain_is_stable_across_hash_seeds(self, tmp_path):
        script = (
            "import json, sys;"
            "sys.path.insert(0, '.');"
            "from fix_exits import fix_exits, add_bidirectional_connections, "
            "connect_disconnected_rooms;"
            "d = json.load(open(sys.argv[1]));"
            "d = fix_exits(d);"
            "d = add_bidirectional_connections(d);"
            "d = connect_disconnected_rooms(d);"
            "print(json.dumps(sorted((x['name'], tuple(sorted(x.get('exits', [])))) "
            "for x in d['rooms'])))"
        )
        fixture = "tests/fixtures/crime_and_punishment_gold.json"

        outputs = set()
        for seed in ("0", "1", "42", "12345"):
            proc = subprocess.run(
                [sys.executable, "-c", script, fixture],
                capture_output=True, text=True, timeout=300, check=False,
                env={"PYTHONHASHSEED": seed, "PATH": "/usr/bin:/bin"},
            )
            assert proc.returncode == 0, proc.stderr
            outputs.add(proc.stdout.strip().splitlines()[-1])

        assert len(outputs) == 1, (
            f"the repair chain produced {len(outputs)} different maps from one "
            "input; something still depends on set iteration order"
        )


class TestConnectSubgraphsBehaviour:
    def test_all_clusters_become_reachable(self, disconnected_rooms):
        result = connect_subgraphs(disconnected_rooms)

        rooms = {r["name"]: r.get("exits", []) for r in result["rooms"]}
        seen, stack = set(), ["A1"]
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            stack.extend(rooms.get(current, []))

        assert seen == set(rooms), "every room should be reachable after connecting"

    def test_already_connected_map_is_untouched(self):
        data = {"rooms": [
            {"name": "A", "description": "d", "exits": ["B"]},
            {"name": "B", "description": "d", "exits": ["A"]},
        ]}
        before = _exit_signature(data)

        result = connect_subgraphs(json.loads(json.dumps(data)))

        assert _exit_signature(result) == before
