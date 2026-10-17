import os
from collections.abc import Callable
from functools import partial

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.events import Resize
from textual.timer import Timer
from textual.widgets import Input, Static

from angularai import __version__
from angularai.errors import AngularAIError
from angularai.session import COMMANDS, cortex_status, session_command
from angularai.store import Store

Speaker = Callable[[Store, str], str]
_REPLAY = 12
_EMPTY = (
    "The cortex is empty.\nSay something. What fits is sent. What it keeps is written here."
)
_FAILED = "The pass failed. Nothing new was written."
_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
_REVEAL_AT = 48

_CSS = """
Screen {
    background: #0c0c0c;
    color: #e8e6e3;
}

#top {
    height: 1;
    padding: 0 1;
    color: #6f6a64;
    background: #0c0c0c;
}

#transcript {
    height: 1fr;
    padding: 1 2 0 2;
    background: #0c0c0c;
    scrollbar-size: 1 1;
    scrollbar-color: #2a2a2a;
    scrollbar-background: #0c0c0c;
}

#empty {
    height: auto;
    color: #6f6a64;
}

#palette {
    height: auto;
    padding: 0 2;
    background: #0c0c0c;
    display: none;
}

.hint {
    height: 1;
    color: #6f6a64;
    background: #0c0c0c;
}

.hint.on {
    color: #e8e6e3;
    background: #191919;
}

#pending {
    height: 1;
    color: #6f6a64;
}

#prompt.busy {
    color: #6f6a64;
}

#dock {
    height: 3;
    border-top: solid #2a2a2a;
    background: #0c0c0c;
}

#composer {
    height: 1;
    padding: 0 1;
}

#prompt {
    width: 2;
    height: 1;
    color: #e8e6e3;
}

#input {
    width: 1fr;
    height: 1;
    min-height: 1;
    border: none;
    padding: 0;
    background: #0c0c0c;
    color: #e8e6e3;
}

#input:focus {
    border: none;
}

#status {
    height: 1;
    padding: 0 1;
    background: #141414;
    color: #7a756e;
}

Block {
    height: auto;
    margin: 0 0 1 0;
}

.role {
    height: 1;
    color: #6f6a64;
}

.body {
    height: auto;
    color: #e8e6e3;
}

.note .body {
    color: #8a847c;
}
"""


class Block(Vertical):
    def __init__(
        self, role: str, body: str, *, note: bool = False, shown: str | None = None
    ) -> None:
        super().__init__(classes="note" if note else "said")
        self.role_text = role
        self.body_text = body
        self._shown = body if shown is None else shown

    def compose(self) -> ComposeResult:
        yield Static(self.role_text, classes="role", markup=False)
        yield Static(self._shown, classes="body", markup=False)


class AngularApp(App[None]):
    """Fullscreen session. The status line is the budget. The file is the memory."""

    CSS = _CSS
    ENABLE_COMMAND_PALETTE = False
    BINDINGS = [
        Binding("up", "history_previous", show=False, priority=True),
        Binding("down", "history_next", show=False, priority=True),
        Binding("tab", "complete", show=False, priority=True),
        Binding("ctrl+c", "quit", show=False),
        Binding("ctrl+d", "quit", show=False, priority=True),
        Binding("ctrl+l", "clear_screen", show=False, priority=True),
    ]

    def __init__(self, store: Store, speak: Speaker) -> None:
        super().__init__()
        self._store = store
        self._speak = speak
        self._busy = False
        self._history: list[str] = []
        self._history_at: int | None = None
        self._draft = ""
        self._selected = 0
        self._spin_i = 0
        self._reveal_timer: Timer | None = None
        self._spinner: Timer | None = None
        self.title = "angularai"

    def compose(self) -> ComposeResult:
        yield Static("", id="top", markup=False)
        yield VerticalScroll(id="transcript")
        yield Vertical(id="palette")
        with Vertical(id="dock"):
            with Horizontal(id="composer"):
                yield Static("›", id="prompt", markup=False)
                yield Input(id="input", compact=True)
            yield Static("", id="status", markup=False)

    def on_mount(self) -> None:
        self._spinner = self.set_interval(0.08, self._spin, pause=True)
        self.query_one("#input", Input).focus()
        self._replay()
        self._paint()
        self.query_one("#transcript", VerticalScroll).scroll_end(animate=False)

    def on_resize(self, _event: Resize) -> None:
        self._paint()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        event.stop()
        text = self._accept(event.value.strip())
        event.input.value = ""
        if not text or self._busy:
            return
        self._take(text)

    def on_input_changed(self, event: Input.Changed) -> None:
        self._show_commands(event.value)

    def action_clear_screen(self) -> None:
        if self._busy:
            return
        self._clear_transcript()

    def action_complete(self) -> None:
        field = self.query_one("#input", Input)
        if not field.value.startswith("/"):
            return
        head = field.value.split(" ", 1)[0]
        matches = [name for name, _hint in COMMANDS if name.startswith(head)]
        if len(matches) != 1:
            return
        field.value = f"{matches[0]} "
        field.cursor_position = len(field.value)

    def action_history_previous(self) -> None:
        if self._palette_open():
            self._nudge(-1)
            return
        if self._busy or not self._history:
            return
        field = self.query_one("#input", Input)
        if self._history_at is None:
            self._draft = field.value
            self._history_at = len(self._history) - 1
        elif self._history_at > 0:
            self._history_at -= 1
        field.value = self._history[self._history_at]
        field.cursor_position = len(field.value)

    def action_history_next(self) -> None:
        if self._palette_open():
            self._nudge(1)
            return
        if self._history_at is None:
            return
        field = self.query_one("#input", Input)
        if self._history_at + 1 >= len(self._history):
            self._history_at = None
            field.value = self._draft
        else:
            self._history_at += 1
            field.value = self._history[self._history_at]
        field.cursor_position = len(field.value)

    def _take(self, text: str) -> None:
        action, payload = session_command(
            self._store, text, cortex_status(self._store, os.environ)
        )
        if action == "quit":
            self.exit()
            return
        if action == "clear":
            self._clear_transcript()
            return
        if action == "continue":
            if payload:
                self._append("angularai", payload, note=True)
            return
        if payload is None:
            return
        self._history.append(payload)
        self._history_at = None
        self._append("you", payload)
        self._set_busy(True)
        self.run_worker(
            partial(self._work, payload),
            name="pass",
            thread=True,
            exclusive=True,
            exit_on_error=False,
        )

    def _work(self, text: str) -> None:
        try:
            reply = self._speak(self._store, text)
        except AngularAIError as exc:
            self.call_from_thread(self._show_error, str(exc))
            return
        except Exception:
            self.call_from_thread(self._show_error, _FAILED)
            return
        self.call_from_thread(self._show_reply, reply)

    def _show_reply(self, reply: str) -> None:
        self._set_busy(False)
        self._append("angularai", reply, reveal=True)

    def _show_error(self, message: str) -> None:
        self._set_busy(False)
        self._append("angularai", message, note=True)

    def _replay(self) -> None:
        records = self._store.passes()[-_REPLAY:]
        if not records:
            self.query_one("#transcript", VerticalScroll).mount(
                Static(_EMPTY, id="empty", markup=False)
            )
            return
        for record in records:
            self._append("you", record.user, motion=False)
            if record.reply is not None:
                self._append("angularai", record.reply, note=record.outcome != "ok", motion=False)
            elif record.error:
                self._append("angularai", record.error, note=True, motion=False)

    def _append(
        self,
        role: str,
        body: str,
        *,
        note: bool = False,
        motion: bool = True,
        reveal: bool = False,
    ) -> Block:
        empty = self.query("#empty")
        if empty:
            empty.first().remove()
        block = Block(role, body, note=note, shown="" if reveal else None)
        transcript = self.query_one("#transcript", VerticalScroll)
        if motion:
            block.styles.opacity = 0.0
        transcript.mount(block)
        if motion or reveal:
            self.call_after_refresh(partial(self._arrive, block, motion=motion, reveal=reveal))
        if motion:
            transcript.scroll_end(animate=True, duration=0.12)
        else:
            transcript.scroll_end(animate=False)
        return block

    def _arrive(self, block: Block, *, motion: bool, reveal: bool) -> None:
        if not block.is_mounted:
            return
        if motion:
            block.styles.animate("opacity", value=1.0, duration=0.16, easing="out_cubic")
        if reveal:
            self._reveal(block)

    def _reveal(self, block: Block) -> None:
        text = block.body_text
        body = block.query_one(".body", Static)
        self._stop_reveal()
        if len(text) < _REVEAL_AT:
            body.update(text)
            return
        index = 0
        step = max(6, len(text) // 18)

        def tick() -> None:
            nonlocal index
            index = min(len(text), index + step)
            if block.is_mounted:
                body.update(text[:index])
            if index >= len(text):
                self._stop_reveal()

        self._reveal_timer = self.set_interval(0.016, tick)

    def _stop_reveal(self) -> None:
        if self._reveal_timer is not None:
            self._reveal_timer.stop()
            self._reveal_timer = None

    def _clear_transcript(self) -> None:
        self._stop_reveal()
        self._dismiss_pending()
        transcript = self.query_one("#transcript", VerticalScroll)
        transcript.remove_children()
        if not self._store.passes():
            transcript.mount(Static(_EMPTY, id="empty", markup=False))
        self._selected = 0
        self.query_one("#input", Input).focus()

    def _show_commands(self, raw: str) -> None:
        self._selected = 0
        self._paint_palette(self._matches(raw))

    def _matches(self, raw: str) -> list[tuple[str, str]]:
        text = raw.strip()
        if not text.startswith("/"):
            return []
        head = text.split(" ", 1)[0]
        return [(name, hint) for name, hint in COMMANDS if name.startswith(head)]

    def _paint_palette(self, matches: list[tuple[str, str]]) -> None:
        palette = self.query_one("#palette", Vertical)
        palette.remove_children()
        if not matches:
            palette.display = False
            return
        self._selected = min(self._selected, len(matches) - 1)
        for index, (name, hint) in enumerate(matches):
            mark = "›" if index == self._selected else " "
            classes = "hint on" if index == self._selected else "hint"
            palette.mount(Static(f"{mark} {name:<10}{hint}", classes=classes, markup=False))
        palette.display = True

    def _palette_open(self) -> bool:
        palette = self.query_one("#palette", Vertical)
        return bool(palette.display and palette.children)

    def _nudge(self, delta: int) -> None:
        matches = self._matches(self.query_one("#input", Input).value)
        if not matches:
            return
        self._selected = (self._selected + delta) % len(matches)
        self._paint_palette(matches)

    def _accept(self, text: str) -> str:
        if not text.startswith("/"):
            return text
        head, _, rest = text.partition(" ")
        names = [name for name, _hint in COMMANDS if name.startswith(head)]
        if head in names or not names:
            return text
        chosen = names[min(self._selected, len(names) - 1)]
        tail = rest.strip()
        if tail:
            return f"{chosen} {tail}"
        return chosen

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        field = self.query_one("#input", Input)
        field.disabled = busy
        prompt = self.query_one("#prompt", Static)
        prompt.set_class(busy, "busy")
        if self._spinner is not None:
            if busy:
                self._spinner.resume()
            else:
                self._spinner.pause()
        if busy:
            self._arm_pending()
        else:
            self._dismiss_pending()
            field.focus()
        self._paint()

    def _arm_pending(self) -> None:
        self._dismiss_pending()
        self.query_one("#transcript", VerticalScroll).mount(
            Static(_FRAMES[self._spin_i], id="pending", markup=False)
        )

    def _dismiss_pending(self) -> None:
        pending = self.query("#pending")
        if pending:
            pending.first().remove()

    def _spin(self) -> None:
        self._spin_i = (self._spin_i + 1) % len(_FRAMES)
        if self.query("#pending"):
            self.query_one("#pending", Static).update(_FRAMES[self._spin_i])
        self._paint()

    def _paint(self) -> None:
        width = self.size.width
        frame = _FRAMES[self._spin_i] if self._busy else ""
        self.query_one("#top", Static).update(_pair("angularai", __version__, max(width - 2, 0)))
        self.query_one("#status", Static).update(
            status_line(self._store, width, self._busy, frame=frame)
        )


def run_tui(store: Store, speak: Speaker) -> int:
    AngularApp(store, speak).run()
    return 0


def status_line(store: Store, width: int, busy: bool, frame: str = "") -> str:
    config = store.config()
    last = store.last_pass()
    traces = len(store.view().current_traces())
    kind = config.substrate or "unwired"
    model = config.model or "no model"
    if busy:
        meter = f"{frame} integrating".strip()
    elif last is None:
        meter = f"—/{config.workspace_budget}"
    else:
        meter = f"{last.used}/{last.budget}"
    left = f"{kind}  {model}    {meter}    {traces} traces"
    room = width - 2
    if room <= 0:
        return ""
    path = str(store.root)
    gap = room - len(left) - len(path)
    if gap >= 2:
        return f"{left}{' ' * gap}{path}"
    if len(left) <= room:
        return left
    return left[:room]


def _pair(left: str, right: str, width: int) -> str:
    if width <= 0:
        return ""
    gap = width - len(left) - len(right)
    if gap < 1:
        return left[:width]
    return f"{left}{' ' * gap}{right}"
