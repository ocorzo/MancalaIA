from __future__ import annotations

import argparse
import json
import random
import threading
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from alphazero_mancala import AlphaZeroMCTS, AlphaZeroNetwork
from mancala_dqn_selfplay import (
    action_to_label,
    check_for_winner,
    get_new_board,
    legal_actions,
    make_move,
)


PLAYER_1_PITS = ("A", "B", "C", "D", "E", "F")
PLAYER_2_PITS = ("G", "H", "I", "J", "K", "L")
TOP_ROW_ENGINE = ("L", "K", "J", "I", "H", "G")
TOP_ROW_DISPLAY = ("L", "K", "J", "I", "H", "G")
TOP_ROW_VALUE_PITS = ("G", "H", "I", "J", "K", "L")
DISPLAY_TO_ENGINE = dict(zip(TOP_ROW_DISPLAY, TOP_ROW_VALUE_PITS))
BOTTOM_ROW = PLAYER_1_PITS
MODEL: AlphaZeroNetwork | None = None
MODEL_LOCK = threading.Lock()
GAMES: dict[str, "GameState"] = {}
SAVE_DIR = Path("saved_games")


def display_label_for_pit(pit: str) -> str:
    return pit


def top_row_slots() -> list[dict[str, str]]:
    return [
        {"pit": value_pit, "label": label}
        for label, value_pit in zip(TOP_ROW_DISPLAY, TOP_ROW_VALUE_PITS)
    ]


class GameState:
    def __init__(self, human_player: str, simulations: int, seed: int | None = None) -> None:
        self.id = uuid.uuid4().hex
        self.board = get_new_board()
        self.human_player = human_player
        self.bot_player = "2" if human_player == "1" else "1"
        self.player_turn = "1"
        self.simulations = simulations
        self.rng = random.Random(seed if seed is not None else random.randrange(2**63))
        self.winner = "no winner"
        self.history: list[dict[str, Any]] = []
        self.created_at = time.time()
        self.updated_at = self.created_at

    def legal(self) -> list[int]:
        if self.winner != "no winner":
            return []
        return legal_actions(self.player_turn, self.board)

    def apply_action(self, actor: str, action: int, visits: dict[int, int] | None = None) -> None:
        before_player = self.player_turn
        before_store = (self.board["1"], self.board["2"])
        self.player_turn = make_move(self.board, self.player_turn, action_to_label(action))
        self.winner = check_for_winner(self.board)
        after_store = (self.board["1"], self.board["2"])
        self.history.append(
            {
                "actor": actor,
                "player": before_player,
                "move": action_to_label(action),
                "move_display": display_label_for_pit(action_to_label(action)),
                "extra_turn": self.winner == "no winner" and self.player_turn == before_player,
                "store_delta": {
                    "1": self.board["1"] - before_store[0],
                    "2": self.board["2"] - before_store[1],
                },
                "stores": {"1": after_store[0], "2": after_store[1]},
                "visits": visits or {},
            }
        )
        self.updated_at = time.time()


def model() -> AlphaZeroNetwork:
    if MODEL is None:
        raise RuntimeError("Modelo no cargado")
    return MODEL


def run_bot_turns(game: GameState) -> None:
    while game.winner == "no winner" and game.player_turn == game.bot_player:
        run_bot_step(game)


def run_bot_step(game: GameState) -> bool:
    if game.winner != "no winner" or game.player_turn != game.bot_player:
        return False
    searcher = AlphaZeroMCTS(
        network=model(),
        simulations=game.simulations,
        rng=random.Random(game.rng.randrange(2**63)),
    )
    with MODEL_LOCK:
        result = searcher.search(game.board, game.player_turn, temperature=0.0, sample=False)
    game.apply_action("bot", result.action, visits=result.visits)
    return True


def auto_play_enabled(payload: dict[str, Any]) -> bool:
    return bool(payload.get("autoPlayBot", False))


def finish_bot_turns_if_requested(game: GameState, payload: dict[str, Any]) -> None:
    if auto_play_enabled(payload):
        run_bot_turns(game)


def public_state(game: GameState) -> dict[str, Any]:
    legal = game.legal()
    return {
        "id": game.id,
        "board": game.board,
        "topRow": top_row_slots(),
        "bottomRow": BOTTOM_ROW,
        "humanPlayer": game.human_player,
        "botPlayer": game.bot_player,
        "playerTurn": game.player_turn,
        "winner": game.winner,
        "legalActions": legal,
        "legalMoves": [display_label_for_pit(action_to_label(action)) for action in legal],
        "humanCanMove": game.winner == "no winner" and game.player_turn == game.human_player,
        "simulations": game.simulations,
        "history": game.history[-20:],
    }


def game_snapshot(game: GameState) -> dict[str, Any]:
    return {
        **public_state(game),
        "history": game.history,
        "createdAt": game.created_at,
        "updatedAt": game.updated_at,
        "savedAt": time.time(),
        "model": "alphazero_mancala_10000_best_mcts.npz",
        "rules": "mancala_dqn_selfplay",
    }


def save_game(game: GameState) -> Path:
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    outcome = game.winner if game.winner != "no winner" else "in_progress"
    path = SAVE_DIR / f"mancala_{timestamp}_{outcome}_{game.id[:8]}.json"
    path.write_text(json.dumps(game_snapshot(game), indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def parse_json(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    content_length = int(handler.headers.get("Content-Length", "0"))
    if content_length <= 0:
        return {}
    raw = handler.rfile.read(content_length)
    return json.loads(raw.decode("utf-8"))


def json_response(handler: BaseHTTPRequestHandler, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
    body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


class MancalaHandler(BaseHTTPRequestHandler):
    server_version = "MancalaAlphaZero/1.0"

    def log_message(self, format: str, *args: Any) -> None:
        print(f"{self.address_string()} - {format % args}")

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            body = HTML.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if path == "/api/health":
            json_response(self, {"ok": True, "games": len(GAMES)})
            return
        json_response(self, {"error": "Ruta no encontrada"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            if path == "/api/new":
                payload = parse_json(self)
                human_player = str(payload.get("humanPlayer", "1"))
                if human_player not in {"1", "2"}:
                    raise ValueError("humanPlayer debe ser '1' o '2'")
                simulations = int(payload.get("simulations", 50))
                simulations = max(1, min(300, simulations))
                game = GameState(human_player=human_player, simulations=simulations)
                GAMES[game.id] = game
                finish_bot_turns_if_requested(game, payload)
                json_response(self, {"game": public_state(game)})
                return

            if path == "/api/move":
                payload = parse_json(self)
                game_id = str(payload.get("gameId", ""))
                move = str(payload.get("move", "")).upper()
                game = GAMES.get(game_id)
                if game is None:
                    raise ValueError("Partida no encontrada")
                if game.winner != "no winner":
                    raise ValueError("La partida ya terminó")
                if game.player_turn != game.human_player:
                    raise ValueError("Todavía no es tu turno")
                legal = {action_to_label(action): action for action in game.legal()}
                if move not in legal and game.player_turn == "2":
                    move = DISPLAY_TO_ENGINE.get(move, move)
                if move not in legal:
                    raise ValueError("Movimiento ilegal")
                game.apply_action("human", legal[move])
                finish_bot_turns_if_requested(game, payload)
                json_response(self, {"game": public_state(game)})
                return

            if path == "/api/bot-step":
                payload = parse_json(self)
                game_id = str(payload.get("gameId", ""))
                game = GAMES.get(game_id)
                if game is None:
                    raise ValueError("Partida no encontrada")
                moved = run_bot_step(game)
                json_response(self, {"game": public_state(game), "moved": moved})
                return

            if path == "/api/state":
                payload = parse_json(self)
                game_id = str(payload.get("gameId", ""))
                game = GAMES.get(game_id)
                if game is None:
                    raise ValueError("Partida no encontrada")
                json_response(self, {"game": public_state(game)})
                return

            if path == "/api/save":
                payload = parse_json(self)
                game_id = str(payload.get("gameId", ""))
                game = GAMES.get(game_id)
                if game is None:
                    raise ValueError("Partida no encontrada")
                saved_path = save_game(game)
                json_response(
                    self,
                    {
                        "game": public_state(game),
                        "snapshot": game_snapshot(game),
                        "savedPath": str(saved_path),
                    },
                )
                return

            json_response(self, {"error": "Ruta no encontrada"}, HTTPStatus.NOT_FOUND)
        except Exception as exc:
            json_response(self, {"error": str(exc)}, HTTPStatus.BAD_REQUEST)


HTML = r"""<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Mancala AlphaZero</title>
  <style>
    :root {
      --ink: #16211f;
      --muted: #66716e;
      --paper: #f6f1e8;
      --line: rgba(22, 33, 31, 0.16);
      --wood-a: #b86d35;
      --wood-b: #7a4427;
      --pit: #f0c97d;
      --pit-dark: #9c5c30;
      --blue: #1f6f8b;
      --green: #2d7d58;
      --red: #b84a3d;
      --gold: #d49a2f;
      font-family: Avenir Next, Gill Sans, Trebuchet MS, sans-serif;
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      color: var(--ink);
      background:
        radial-gradient(circle at 24% 18%, rgba(212, 154, 47, 0.18), transparent 26rem),
        linear-gradient(135deg, #f8f1e2 0%, #e7d3b0 52%, #d8b684 100%);
      min-height: 100vh;
    }

    main {
      width: min(1180px, calc(100vw - 32px));
      margin: 0 auto;
      padding: 28px 0 40px;
    }

    header {
      display: grid;
      grid-template-columns: 1fr auto;
      align-items: end;
      gap: 20px;
      margin-bottom: 22px;
    }

    h1 {
      margin: 0;
      font-size: clamp(2rem, 5vw, 4.2rem);
      line-height: 0.95;
      letter-spacing: 0;
      max-width: 720px;
    }

    .subhead {
      margin: 10px 0 0;
      color: var(--muted);
      font-size: 1.02rem;
    }

    .controls {
      display: flex;
      gap: 10px;
      align-items: center;
      justify-content: flex-end;
      flex-wrap: wrap;
    }

    button, select, input {
      font: inherit;
    }

    .control {
      border: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.6);
      min-height: 42px;
      padding: 8px 10px;
    }

    button {
      border: 0;
      cursor: pointer;
      min-height: 42px;
      padding: 10px 14px;
      background: var(--ink);
      color: white;
      font-weight: 700;
      box-shadow: 0 10px 24px rgba(22, 33, 31, 0.16);
    }

    button:disabled {
      cursor: not-allowed;
      opacity: 0.45;
      box-shadow: none;
    }

    .layout {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 320px;
      gap: 22px;
      align-items: start;
    }

    .board-wrap {
      background:
        linear-gradient(90deg, rgba(255,255,255,0.1), transparent),
        linear-gradient(135deg, var(--wood-a), var(--wood-b));
      border: 1px solid rgba(70, 39, 20, 0.35);
      box-shadow: 0 22px 50px rgba(65, 40, 21, 0.24);
      padding: clamp(14px, 2.8vw, 28px);
      position: relative;
      overflow: hidden;
    }

    .board {
      display: grid;
      grid-template-columns: minmax(72px, 0.9fr) repeat(6, minmax(58px, 1fr)) minmax(72px, 0.9fr);
      grid-template-rows: repeat(2, minmax(96px, 1fr));
      gap: clamp(9px, 1.5vw, 16px);
      min-height: 280px;
    }

    .store {
      grid-row: span 2;
      min-height: 220px;
      border-radius: 999px;
      background: linear-gradient(160deg, #f6d796, #bd743b 72%);
      box-shadow: inset 0 14px 22px rgba(80, 42, 18, 0.34), 0 3px 0 rgba(255,255,255,0.25);
      display: grid;
      place-items: center;
      position: relative;
    }

    .store.p2 { grid-column: 1; }
    .store.p1 { grid-column: 8; }

    .pit {
      border: 0;
      padding: 0;
      min-height: 96px;
      border-radius: 999px;
      color: var(--ink);
      background: radial-gradient(circle at 42% 34%, var(--pit), var(--pit-dark));
      box-shadow: inset 0 12px 20px rgba(80, 42, 18, 0.34), 0 3px 0 rgba(255,255,255,0.25);
      display: grid;
      place-items: center;
      position: relative;
      transition: transform 140ms ease, outline 140ms ease, filter 140ms ease;
    }

    .pit.legal {
      outline: 4px solid rgba(31, 111, 139, 0.72);
      filter: saturate(1.1);
    }

    .pit.legal:hover {
      transform: translateY(-4px);
    }

    .count {
      font-size: clamp(1.8rem, 4vw, 3rem);
      font-weight: 900;
      text-shadow: 0 1px 0 rgba(255,255,255,0.38);
    }

    .label {
      position: absolute;
      left: 50%;
      bottom: 9px;
      transform: translateX(-50%);
      font-size: 0.82rem;
      font-weight: 800;
      color: rgba(22, 33, 31, 0.72);
    }

    .store .label {
      bottom: 14px;
    }

    .status, .panel {
      background: rgba(255, 255, 255, 0.64);
      border: 1px solid var(--line);
      padding: 16px;
      box-shadow: 0 12px 32px rgba(65, 40, 21, 0.12);
    }

    .status {
      margin-top: 16px;
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
    }

    .badge {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      color: white;
      background: var(--blue);
      padding: 8px 10px;
      font-weight: 800;
      white-space: nowrap;
    }

    .badge.win { background: var(--green); }
    .badge.loss { background: var(--red); }
    .badge.wait { background: var(--gold); color: var(--ink); }

    .status-actions {
      display: flex;
      gap: 10px;
      align-items: center;
      justify-content: flex-end;
      flex-wrap: wrap;
    }

    .secondary {
      background: var(--blue);
    }

    .panel h2 {
      margin: 0 0 10px;
      font-size: 1.1rem;
    }

    .stats {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
      margin-bottom: 16px;
    }

    .stat {
      border: 1px solid var(--line);
      padding: 10px;
      background: rgba(255,255,255,0.45);
    }

    .stat strong {
      display: block;
      font-size: 1.7rem;
    }

    .history {
      display: grid;
      gap: 8px;
      max-height: 420px;
      overflow: auto;
      padding-right: 4px;
    }

    .move {
      display: grid;
      grid-template-columns: 54px 1fr;
      gap: 8px;
      border-bottom: 1px solid var(--line);
      padding-bottom: 8px;
      color: var(--muted);
    }

    .move b { color: var(--ink); }

    .error {
      color: var(--red);
      margin-top: 10px;
      min-height: 22px;
      font-weight: 700;
    }

    @media (max-width: 900px) {
      header, .layout {
        grid-template-columns: 1fr;
      }
      .controls {
        justify-content: start;
      }
      .board {
        grid-template-columns: minmax(46px, 0.7fr) repeat(6, minmax(38px, 1fr)) minmax(46px, 0.7fr);
        gap: 7px;
        min-height: 220px;
      }
      .pit { min-height: 72px; }
      .store { min-height: 166px; }
      .label { bottom: 5px; font-size: 0.72rem; }
    }
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>Mancala AlphaZero</h1>
        <p class="subhead">Juega contra el checkpoint campeón con búsqueda MCTS.</p>
      </div>
      <div class="controls">
        <select id="side" class="control" title="Lado humano">
          <option value="1">Yo juego abajo</option>
          <option value="2">Yo juego arriba</option>
        </select>
        <label>
          <input id="sims" class="control" type="number" min="1" max="300" value="50" title="Simulaciones MCTS" />
        </label>
        <button id="newGame">Nueva partida</button>
        <button id="saveGame" class="secondary" disabled>Guardar partida</button>
      </div>
    </header>

    <section class="layout">
      <div>
        <div class="board-wrap">
          <div id="board" class="board"></div>
        </div>
        <div class="status">
          <div>
            <strong id="headline">Preparando tablero...</strong>
            <div id="detail" class="subhead"></div>
          </div>
          <div class="status-actions">
            <button id="botMove" class="secondary" disabled>Tirar agente</button>
            <span id="badge" class="badge wait">Sin partida</span>
          </div>
        </div>
        <div id="error" class="error"></div>
      </div>
      <aside class="panel">
        <h2>Marcador</h2>
        <div class="stats">
          <div class="stat">Tú<strong id="humanScore">0</strong></div>
          <div class="stat">Modelo<strong id="botScore">0</strong></div>
        </div>
        <h2>Historial</h2>
        <div id="history" class="history"></div>
      </aside>
    </section>
  </main>

  <script>
    const boardEl = document.querySelector("#board");
    const sideEl = document.querySelector("#side");
    const simsEl = document.querySelector("#sims");
    const newGameEl = document.querySelector("#newGame");
    const saveGameEl = document.querySelector("#saveGame");
    const botMoveEl = document.querySelector("#botMove");
    const headlineEl = document.querySelector("#headline");
    const detailEl = document.querySelector("#detail");
    const badgeEl = document.querySelector("#badge");
    const errorEl = document.querySelector("#error");
    const historyEl = document.querySelector("#history");
    const humanScoreEl = document.querySelector("#humanScore");
    const botScoreEl = document.querySelector("#botScore");
    let game = null;
    let busy = false;

    async function api(path, payload) {
      const response = await fetch(path, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload || {})
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "Error de servidor");
      return data;
    }

    function pitButton(slot, row) {
      const pit = typeof slot === "string" ? slot : slot.pit;
      const label = typeof slot === "string" ? slot : slot.label;
      const value = game.board[pit];
      const actionIndex = "ABCDEFGHIJKL".indexOf(pit);
      const legal = game.humanCanMove && game.legalActions.includes(actionIndex);
      const button = document.createElement("button");
      button.className = `pit ${legal ? "legal" : ""}`;
      button.style.gridRow = row;
      button.disabled = !legal || busy;
      button.innerHTML = `<span class="count">${value}</span><span class="label">${label}</span>`;
      if (legal) button.addEventListener("click", () => playMove(pit));
      return button;
    }

    function store(player) {
      const div = document.createElement("div");
      div.className = `store p${player}`;
      div.innerHTML = `<span class="count">${game.board[player]}</span><span class="label">Jugador ${player}</span>`;
      return div;
    }

    function renderBoard() {
      boardEl.innerHTML = "";
      boardEl.appendChild(store("2"));
      for (const slot of game.topRow) boardEl.appendChild(pitButton(slot, 1));
      for (const slot of game.bottomRow) boardEl.appendChild(pitButton(slot, 2));
      boardEl.appendChild(store("1"));
    }

    function statusText() {
      if (game.winner !== "no winner") {
        if (game.winner === "tie") return ["Empate", "La partida terminó pareja.", "wait"];
        if (game.winner === game.humanPlayer) return ["Ganaste", "Le quitaste la corona al modelo.", "win"];
        return ["Ganó el modelo", "AlphaZero cerró la partida.", "loss"];
      }
      if (game.humanCanMove) return ["Tu turno", `Jugadas legales: ${game.legalMoves.join(", ")}`, "win"];
      return ["Le toca al agente", "Revisa el tablero y presiona “Tirar agente” cuando quieras ver su jugada.", "wait"];
    }

    function renderStatus() {
      const [headline, detail, kind] = statusText();
      headlineEl.textContent = headline;
      detailEl.textContent = detail;
      badgeEl.textContent = game.winner !== "no winner" ? "Fin" : `Turno ${game.playerTurn}`;
      badgeEl.className = `badge ${kind}`;
      humanScoreEl.textContent = game.board[game.humanPlayer];
      botScoreEl.textContent = game.board[game.botPlayer];
      saveGameEl.disabled = busy || !game || !game.history.length;
      botMoveEl.disabled = busy || game.winner !== "no winner" || game.playerTurn !== game.botPlayer;
      if (game.winner === "no winner" && game.playerTurn === game.botPlayer) {
        const last = game.history[game.history.length - 1];
        botMoveEl.textContent = last && last.actor === "bot" && last.extra_turn ? "Tirar agente otra vez" : "Tirar agente";
      } else {
        botMoveEl.textContent = "Tirar agente";
      }
    }

    function renderHistory() {
      historyEl.innerHTML = "";
      const items = [...game.history].reverse();
      if (!items.length) {
        historyEl.innerHTML = `<div class="subhead">Todavía no hay jugadas.</div>`;
        return;
      }
      for (const item of items) {
        const row = document.createElement("div");
        row.className = "move";
        const actor = item.actor === "human" ? "Tú" : "Modelo";
        const extra = item.extra_turn ? " · repite turno" : "";
        row.innerHTML = `<b>${item.move_display || item.move}</b><span>${actor}, jugador ${item.player}${extra}</span>`;
        historyEl.appendChild(row);
      }
    }

    function render() {
      if (!game) return;
      renderBoard();
      renderStatus();
      renderHistory();
    }

    async function newGame() {
      busy = true;
      errorEl.textContent = "";
      newGameEl.disabled = true;
      saveGameEl.disabled = true;
      try {
        const data = await api("/api/new", {
          humanPlayer: sideEl.value,
          simulations: Number(simsEl.value || 50),
          autoPlayBot: false
        });
        game = data.game;
        render();
      } catch (error) {
        errorEl.textContent = error.message;
      } finally {
        busy = false;
        newGameEl.disabled = false;
        render();
      }
    }

    async function saveGame() {
      if (!game || busy) return;
      busy = true;
      errorEl.textContent = "";
      saveGameEl.disabled = true;
      try {
        const data = await api("/api/save", { gameId: game.id });
        const filename = `${data.savedPath.split("/").pop()}`;
        const blob = new Blob([JSON.stringify(data.snapshot, null, 2)], { type: "application/json" });
        const url = URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.href = url;
        link.download = filename;
        document.body.appendChild(link);
        link.click();
        link.remove();
        URL.revokeObjectURL(url);
        detailEl.textContent = `Partida guardada en servidor: ${data.savedPath}`;
      } catch (error) {
        errorEl.textContent = error.message;
      } finally {
        busy = false;
        render();
      }
    }

    async function playMove(move) {
      if (!game || busy) return;
      busy = true;
      errorEl.textContent = "";
      render();
      try {
        const data = await api("/api/move", { gameId: game.id, move, autoPlayBot: false });
        game = data.game;
        render();
      } catch (error) {
        errorEl.textContent = error.message;
      } finally {
        busy = false;
        render();
      }
    }

    async function playBotMove() {
      if (!game || busy || game.winner !== "no winner" || game.playerTurn !== game.botPlayer) return;
      busy = true;
      errorEl.textContent = "";
      headlineEl.textContent = "AlphaZero está calculando";
      detailEl.textContent = "El tablero se moverá cuando termine la búsqueda MCTS.";
      botMoveEl.disabled = true;
      try {
        const data = await api("/api/bot-step", { gameId: game.id });
        game = data.game;
      } catch (error) {
        errorEl.textContent = error.message;
      } finally {
        busy = false;
        render();
      }
    }

    newGameEl.addEventListener("click", newGame);
    saveGameEl.addEventListener("click", saveGame);
    botMoveEl.addEventListener("click", playBotMove);
    newGame();
  </script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Interfaz web para jugar Mancala contra AlphaZero + MCTS.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8088)
    parser.add_argument("--model", type=Path, default=Path("alphazero_mancala_10000_best_mcts.npz"))
    args = parser.parse_args()

    global MODEL
    MODEL = AlphaZeroNetwork()
    MODEL.load(args.model)
    server = ThreadingHTTPServer((args.host, args.port), MancalaHandler)
    print(f"Mancala AlphaZero web en http://{args.host}:{args.port}")
    print(f"Modelo: {args.model}")
    server.serve_forever()


if __name__ == "__main__":
    main()
