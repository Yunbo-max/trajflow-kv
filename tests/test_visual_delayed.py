import json

from PIL import Image

from trajflow_kv.counterfactual import load_counterfactual_jsonl, write_counterfactual_jsonl
from trajflow_kv.visual_delayed import (
    DistractorCreditTask,
    HiddenMemoryTask,
    build_visual_counterfactual_dataset,
)


def _group(rows):
    groups = {}
    for row in rows:
        groups.setdefault(row["prefix_id"], []).append(row)
    return groups


def test_distractor_credit_has_equal_noncritical_q_and_critical_fork(tmp_path):
    task = DistractorCreditTask()
    rows = build_visual_counterfactual_dataset([0], output_dir=tmp_path / "images", tasks=[task], horizon=8)
    groups = _group(rows)
    assert rows
    assert all(row["visual"] for row in rows)
    assert all(row["benchmark"] == "tango_visual_delayed_gui" for row in rows)
    x_rows = next(group for group in groups.values() if group[0]["prefix"]["state"]["phase"] == "x")
    assert {row["Q"] for row in x_rows} == {x_rows[0]["Q"]}
    fork_rows = next(group for group in groups.values() if group[0]["prefix"]["state"]["phase"] == "fork")
    q = {row["action"]: row["Q"] for row in fork_rows}
    assert q["choose_A"] != q["choose_B"]
    assert all(row["is_critical_action"] for row in fork_rows)
    assert not any(row["is_critical_action"] for row in x_rows)


def test_hidden_memory_hides_cue_in_choice_screenshot_and_executes(tmp_path):
    task = HiddenMemoryTask()
    rows = build_visual_counterfactual_dataset([1], output_dir=tmp_path / "images", tasks=[task], horizon=8)
    choose = next(group for group in _group(rows).values() if group[0]["prefix"]["state"]["phase"] == "choose")
    assert all(row["critical_step"] for row in choose)
    assert all(row["is_critical_action"] for row in choose)
    image_path = choose[0]["image"]
    assert Image.open(image_path).size == (960, 600)
    # Seed 1 selects blue; the state machine is directly executable.
    state = task.initial_state(1)
    state, done = task.step(state, "continue")
    assert not done and state["phase"] == "choose"
    state, done = task.step(state, "choose_blue")
    assert not done and state["phase"] == "submit"
    state, done = task.step(state, "submit")
    assert done and state["terminal_return"] == 1.0


def test_visual_rows_roundtrip_existing_counterfactual_schema(tmp_path):
    rows = build_visual_counterfactual_dataset([0], output_dir=tmp_path / "images", tasks=[HiddenMemoryTask()])
    path = tmp_path / "visual.jsonl"
    assert write_counterfactual_jsonl(rows, path) == len(rows)
    loaded = load_counterfactual_jsonl(path)
    assert len(loaded) == len(rows)
    assert {row["task_family"] for row in loaded} == {"hidden_memory"}
    assert all(json.loads(json.dumps(row["prefix"]))["screenshot_path"] for row in loaded)

