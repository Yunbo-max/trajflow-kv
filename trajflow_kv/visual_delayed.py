"""Small screenshot-backed delayed-consequence GUI benchmark.

This module turns the deterministic counterfactual tasks into actual visual
GUI observations without requiring a browser or an Android emulator.  The
state machine is executable through :class:`DelayedConsequenceTask.step`,
while :func:`build_visual_counterfactual_dataset` writes the existing
``tango.counterfactual.v1`` JSONL rows plus one deterministic PNG per prefix.

The benchmark is intentionally controlled: ``distractor_credit`` has two
harmless action stages around one hidden critical fork, and
``hidden_memory`` exposes a cue which disappears before the decision.  Each
row keeps an immutable same-prefix group and annotates the ground-truth
critical step so credit localization can be evaluated independently of
policy success.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw, ImageFont

from .counterfactual import DelayedConsequenceTask, build_counterfactual_examples


_PALETTE = {
    "red": (214, 61, 61),
    "blue": (54, 103, 198),
    "green": (48, 157, 92),
    "orange": (224, 132, 42),
}


def _font(size: int = 22) -> ImageFont.ImageFont:
    # Prefer a common container font so screenshots are legible to a VLM, but
    # retain the bitmap fallback for minimal CI images.
    for font_path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        if Path(font_path).exists():
            return ImageFont.truetype(font_path, size=size)
    return ImageFont.load_default()


def _draw_button(draw: ImageDraw.ImageDraw, xy: tuple[int, int, int, int], label: str, *, fill=(238, 242, 248)) -> None:
    draw.rounded_rectangle(xy, radius=10, fill=fill, outline=(60, 70, 85), width=2)
    left, top, right, bottom = xy
    bbox = draw.textbbox((0, 0), label, font=_font())
    draw.text(
        ((left + right - bbox[2]) / 2, (top + bottom - bbox[3]) / 2 - 2),
        label,
        fill=(25, 31, 40),
        font=_font(),
    )


class VisualDelayedTask(DelayedConsequenceTask):
    """Extension point for screenshot rendering and critical-step labels."""

    benchmark = "tango_visual_delayed_gui"

    def critical_actions(self, state: dict[str, Any]) -> tuple[str, ...]:
        return ()

    def optimal_actions(self, state: dict[str, Any]) -> tuple[str, ...]:
        return ()

    def render_screenshot(self, state: dict[str, Any], path: str | Path) -> None:
        raise NotImplementedError


class DistractorCreditTask(VisualDelayedTask):
    """Two irrelevant action stages surround a hidden, consequential fork.

    A transient target cue is shown on the first screen and disappears before
    the ``choose_a``/``choose_b`` fork.  ``click_x``/``skip_x`` and
    ``click_y``/``skip_y`` are deliberately return-equivalent distractors.
    """

    task_family = "distractor_credit"
    instruction = "Remember the target option, pass the harmless steps, choose it, then submit."
    options = ("A", "B")

    def initial_state(self, seed: int) -> dict[str, Any]:
        target = self.options[int(seed) % len(self.options)]
        return {"phase": "x", "target": target, "choice": None, "history": [], "seed": int(seed)}

    def available_actions(self, state: dict[str, Any]) -> tuple[str, ...]:
        phase = state["phase"]
        if phase == "x":
            return ("click_x", "skip_x")
        if phase == "fork":
            return ("choose_A", "choose_B")
        if phase == "y":
            return ("click_y", "skip_y")
        if phase == "submit":
            return ("submit", "cancel")
        return ()

    def observe(self, state: dict[str, Any]) -> dict[str, Any]:
        phase = state["phase"]
        if phase == "x":
            screen = f"Temporary banner: TARGET OPTION = {state['target']}. Harmless step X."
        elif phase == "fork":
            screen = "Target banner disappeared. Choose the correct option, then continue."
        elif phase == "y":
            screen = "Harmless step Y. The selected option is not shown here."
        elif phase == "submit":
            screen = f"Ready to submit. Selected option: {state['choice']}."
        else:
            screen = "Task finished."
        return {"screen": screen, "candidates": list(self.available_actions(state))}

    def critical_actions(self, state: dict[str, Any]) -> tuple[str, ...]:
        return tuple(self.available_actions(state)) if state["phase"] == "fork" else ()

    def optimal_actions(self, state: dict[str, Any]) -> tuple[str, ...]:
        if state["phase"] == "fork":
            return (f"choose_{state['target']}",)
        if state["phase"] == "submit" and state.get("choice") == state.get("target"):
            return ("submit",)
        return tuple(self.available_actions(state)) if state["phase"] in {"x", "y"} else ()

    def step(self, state: dict[str, Any], action: str) -> tuple[dict[str, Any], bool]:
        state = self.clone(state)
        state["history"].append(action)
        phase = state["phase"]
        if phase == "x" and action in self.available_actions(state):
            state["phase"] = "fork"
            return state, False
        if phase == "fork" and action in self.available_actions(state):
            state["choice"] = action.rsplit("_", 1)[-1]
            state["phase"] = "y"
            return state, False
        if phase == "y" and action in self.available_actions(state):
            state["phase"] = "submit"
            return state, False
        if phase == "submit":
            state.update(
                phase="terminal",
                terminal_return=float(action == "submit" and state.get("choice") == state.get("target")),
            )
            return state, True
        state.update(phase="terminal", terminal_return=0.0)
        return state, True

    def render_screenshot(self, state: dict[str, Any], path: str | Path) -> None:
        image = Image.new("RGB", (960, 600), (247, 249, 252))
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, 960, 68), fill=(35, 48, 68))
        draw.text((28, 22), "TANGO delayed workflow", fill="white", font=_font())
        draw.text((28, 100), self.instruction, fill=(30, 38, 50), font=_font())
        phase = state["phase"]
        if phase == "x":
            draw.rounded_rectangle((40, 150, 920, 220), radius=12, fill=(255, 239, 192), outline=(206, 155, 39), width=2)
            draw.text((65, 177), f"Remember this: target option is {state['target']}", fill=(105, 72, 10), font=_font())
            labels = ("Click harmless X", "Skip harmless X")
        elif phase == "fork":
            draw.text((50, 168), "Choose the option required by the earlier target.", fill=(30, 38, 50), font=_font())
            labels = ("Choose A", "Choose B")
        elif phase == "y":
            draw.text((50, 168), "This step has no effect on the final result.", fill=(30, 38, 50), font=_font())
            labels = ("Click harmless Y", "Skip harmless Y")
        elif phase == "submit":
            draw.text((50, 168), f"Selected option: {state['choice']}", fill=(30, 38, 50), font=_font())
            labels = ("Submit", "Cancel")
        else:
            draw.text((50, 168), "Task finished.", fill=(30, 38, 50), font=_font())
            labels = ()
        for index, label in enumerate(labels):
            left = 60 + index * 430
            _draw_button(draw, (left, 275, left + 350, 355), label)
        draw.text((50, 470), f"Visible history length: {len(state.get('history', []))}", fill=(100, 108, 120), font=_font())
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        image.save(path, format="PNG")


class HiddenMemoryTask(VisualDelayedTask):
    """A visible cue disappears before the visual choice screen."""

    task_family = "hidden_memory"
    instruction = "Remember the banner color, then select that color after it disappears."
    colors = ("red", "blue", "green")

    def initial_state(self, seed: int) -> dict[str, Any]:
        return {"phase": "cue", "cue": self.colors[int(seed) % len(self.colors)], "choice": None, "history": [], "seed": int(seed)}

    def available_actions(self, state: dict[str, Any]) -> tuple[str, ...]:
        if state["phase"] == "cue":
            return ("continue",)
        if state["phase"] == "choose":
            return tuple(f"choose_{color}" for color in self.colors)
        if state["phase"] == "submit":
            return ("submit", "cancel")
        return ()

    def observe(self, state: dict[str, Any]) -> dict[str, Any]:
        phase = state["phase"]
        if phase == "cue":
            screen = f"A banner shows the color {state['cue']}. Press continue."
        elif phase == "choose":
            screen = "The banner disappeared. Select the color you remember."
        elif phase == "submit":
            screen = f"Selected color: {state['choice']}. Press submit."
        else:
            screen = "Task finished."
        return {"screen": screen, "candidates": list(self.available_actions(state))}

    def critical_actions(self, state: dict[str, Any]) -> tuple[str, ...]:
        return tuple(self.available_actions(state)) if state["phase"] == "choose" else ()

    def optimal_actions(self, state: dict[str, Any]) -> tuple[str, ...]:
        if state["phase"] == "choose":
            return (f"choose_{state['cue']}",)
        if state["phase"] == "submit" and state.get("choice") == state.get("cue"):
            return ("submit",)
        return tuple(self.available_actions(state)) if state["phase"] == "cue" else ()

    def step(self, state: dict[str, Any], action: str) -> tuple[dict[str, Any], bool]:
        state = self.clone(state)
        state["history"].append(action)
        phase = state["phase"]
        if phase == "cue" and action == "continue":
            state["phase"] = "choose"
            return state, False
        if phase == "choose" and action in self.available_actions(state):
            state["choice"] = action[len("choose_"):]
            state["phase"] = "submit"
            return state, False
        if phase == "submit":
            state.update(phase="terminal", terminal_return=float(action == "submit" and state.get("choice") == state.get("cue")))
            return state, True
        state.update(phase="terminal", terminal_return=0.0)
        return state, True

    def render_screenshot(self, state: dict[str, Any], path: str | Path) -> None:
        image = Image.new("RGB", (960, 600), (248, 250, 253))
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, 960, 68), fill=(35, 48, 68))
        draw.text((28, 22), "TANGO hidden-memory task", fill="white", font=_font())
        draw.text((28, 100), self.instruction, fill=(30, 38, 50), font=_font())
        phase = state["phase"]
        if phase == "cue":
            color = _PALETTE[state["cue"]]
            draw.rounded_rectangle((50, 150, 910, 240), radius=12, fill=color)
            draw.text((75, 185), f"MEMORIZE: {state['cue'].upper()}", fill="white", font=_font())
            labels = ("Continue",)
        elif phase == "choose":
            draw.text((50, 170), "The banner is gone. Which color was shown?", fill=(30, 38, 50), font=_font())
            labels = tuple(color.title() for color in self.colors)
        elif phase == "submit":
            draw.text((50, 170), f"Selected color: {state['choice']}", fill=(30, 38, 50), font=_font())
            labels = ("Submit", "Cancel")
        else:
            draw.text((50, 170), "Task finished.", fill=(30, 38, 50), font=_font())
            labels = ()
        width = 260 if len(labels) >= 3 else 350
        gap = 25
        for index, label in enumerate(labels):
            left = 55 + index * (width + gap)
            fill = _PALETTE.get(label.lower(), (238, 242, 248)) if phase == "choose" else (238, 242, 248)
            _draw_button(draw, (left, 275, left + width, 355), label, fill=fill)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        image.save(path, format="PNG")


def visual_delayed_tasks() -> tuple[VisualDelayedTask, ...]:
    """Return the controlled visual families used by the pilot."""
    return (DistractorCreditTask(), HiddenMemoryTask())


def build_visual_counterfactual_dataset(
    seeds: Iterable[int] = range(500),
    *,
    output_dir: str | Path = "data/visual_delayed/images",
    tasks: Iterable[VisualDelayedTask] | None = None,
    horizon: int = 8,
    aggregation: str = "mean",
) -> list[dict[str, Any]]:
    """Build visual same-prefix rows and render one PNG for every prefix."""
    destination = Path(output_dir)
    selected_tasks = tuple(tasks) if tasks is not None else visual_delayed_tasks()
    rows: list[dict[str, Any]] = []
    for task in selected_tasks:
        for seed in tuple(int(item) for item in seeds):
            examples = build_counterfactual_examples(task, seed, horizon=horizon, aggregation=aggregation)
            for row in examples:
                image_path = destination / f"{task.task_family}_s{seed}_{row['prefix_id'].rsplit('-', 1)[-1]}.png"
                task.render_screenshot(row["prefix"]["state"], image_path)
                observation = row["prefix"]["observation"]
                critical_actions = list(task.critical_actions(row["prefix"]["state"]))
                optimal_actions = list(task.optimal_actions(row["prefix"]["state"]))
                row.update(
                    benchmark=task.benchmark,
                    visual=True,
                    image=str(image_path),
                    screenshot_path=str(image_path),
                    critical_step=bool(critical_actions),
                    critical_actions=critical_actions,
                    optimal_actions=optimal_actions,
                    is_critical_action=row["action"] in critical_actions,
                    prompt=(
                        "You are a GUI policy. Inspect the screenshot and choose exactly one candidate action. "
                        f"Task: {row['instruction']} Candidates: {row['candidate_actions']}"
                    ),
                )
                row["prefix"]["image"] = str(image_path)
                row["prefix"]["screenshot_path"] = str(image_path)
                row["prefix"]["critical_step"] = bool(critical_actions)
                row["prefix"]["critical_actions"] = critical_actions
                row["prefix"]["optimal_actions"] = optimal_actions
                # The text observation remains available for non-vision
                # baselines; the screenshot is the primary VLM input.
                observation["screenshot_path"] = str(image_path)
                rows.append(row)
    return rows
