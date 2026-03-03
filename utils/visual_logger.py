"""
Visual dashboard system for XAKCN trading bot.
Unicode layout with PT-BR labels (without accents).
"""

import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union


LineSpec = Union[str, Tuple[str, str]]


class VisualLogger:
    """Terminal visual logger with reusable panel rendering."""

    def __init__(self, use_colors: bool = True, width: int = 100, use_unicode: bool = True):
        self.use_colors = use_colors and self._supports_color()
        self.width = width
        self.use_unicode = use_unicode and self._supports_unicode()

        if self.use_unicode:
            self.BOX = {
                "h": "─",
                "v": "│",
                "tl": "┌",
                "tr": "┐",
                "bl": "└",
                "br": "┘",
                "ml": "├",
                "mr": "┤",
                "d_h": "═",
                "d_v": "║",
                "d_tl": "╔",
                "d_tr": "╗",
                "d_bl": "╚",
                "d_br": "╝",
                "d_ml": "╠",
                "d_mr": "╣",
            }
        else:
            self.BOX = {
                "h": "-",
                "v": "|",
                "tl": "+",
                "tr": "+",
                "bl": "+",
                "br": "+",
                "ml": "+",
                "mr": "+",
                "d_h": "=",
                "d_v": "|",
                "d_tl": "+",
                "d_tr": "+",
                "d_bl": "+",
                "d_br": "+",
                "d_ml": "+",
                "d_mr": "+",
            }

        self.SYMBOLS = {
            "buy": "▲" if self.use_unicode else "[B]",
            "sell": "▼" if self.use_unicode else "[S]",
            "hold": "◆" if self.use_unicode else "[H]",
            "up": "↑" if self.use_unicode else "/\\",
            "down": "↓" if self.use_unicode else "\\/",
            "flat": "→" if self.use_unicode else "--",
            "money": "¤" if self.use_unicode else "$",
            "clock": "◷" if self.use_unicode else "@",
        }

        self.COLORS = {
            "reset": "\033[0m",
            "bold": "\033[1m",
            "dim": "\033[2m",
            "green": "\033[32m",
            "bright_green": "\033[92m",
            "red": "\033[31m",
            "bright_red": "\033[91m",
            "yellow": "\033[33m",
            "bright_yellow": "\033[93m",
            "blue": "\033[34m",
            "bright_blue": "\033[94m",
            "cyan": "\033[36m",
            "bright_cyan": "\033[96m",
            "magenta": "\033[35m",
            "bright_magenta": "\033[95m",
            "white": "\033[37m",
            "bright_white": "\033[97m",
        }

    def _supports_color(self) -> bool:
        try:
            return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
        except Exception:
            return False

    def _supports_unicode(self) -> bool:
        """Check if stdout encoding supports core box drawing chars."""
        try:
            encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
            sample = "╔═╦╗║╚╝┌─┐│└┘▲▼█░"
            sample.encode(encoding)
            return True
        except Exception:
            return False

    def _color(self, text: str, color: str) -> str:
        if self.use_colors and color in self.COLORS:
            return f"{self.COLORS[color]}{text}{self.COLORS['reset']}"
        return text

    def _pad(self, text: str, width: int, align: str = "left") -> str:
        text = str(text)[:width]
        if align == "center":
            left = (width - len(text)) // 2
            return (" " * left) + text + (" " * (width - len(text) - left))
        if align == "right":
            return (" " * (width - len(text))) + text
        return text + (" " * (width - len(text)))

    def _format_float(self, value: Any, decimals: int = 2) -> str:
        try:
            number = float(value)
            if number != number:
                return "n/a"
            return f"{number:,.{decimals}f}"
        except Exception:
            return "n/a"

    def _create_bar(self, value: float, min_val: float, max_val: float, width: int) -> str:
        if max_val == min_val:
            return "[" + (" " * width) + "]"
        ratio = (value - min_val) / (max_val - min_val)
        ratio = max(0.0, min(1.0, ratio))
        filled = int(ratio * width)
        if self.use_unicode:
            return "[" + ("█" * filled) + ("░" * (width - filled)) + "]"
        return "[" + ("#" * filled) + ("-" * (width - filled)) + "]"

    def _score_bar(self, score: float, width: int = 24) -> str:
        normalized = (max(-1.0, min(1.0, score)) + 1.0) / 2.0
        return self._create_bar(normalized, 0.0, 1.0, width)

    def _adx_strength(self, adx: float) -> str:
        if adx >= 35:
            return "FORTE"
        if adx >= 22:
            return "MODERADO"
        return "FRACO"

    def _panel(
        self,
        title: str,
        lines: List[LineSpec],
        border_color: str = "white",
        title_color: str = "bright_white",
        double: bool = False,
    ):
        b = self.BOX
        w = self.width
        prefix = "d_" if double else ""
        h = b[f"{prefix}h"]
        v = b[f"{prefix}v"]
        tl = b[f"{prefix}tl"]
        tr = b[f"{prefix}tr"]
        bl = b[f"{prefix}bl"]
        br = b[f"{prefix}br"]
        ml = b[f"{prefix}ml"]
        mr = b[f"{prefix}mr"]

        top = tl + (h * (w - 2)) + tr
        mid = ml + (h * (w - 2)) + mr
        bot = bl + (h * (w - 2)) + br

        print()
        print(self._color(top, border_color))
        title_line = self._pad(f" {title} ", w - 2, "center")
        print(self._color(v, border_color) + self._color(title_line, title_color) + self._color(v, border_color))
        print(self._color(mid, border_color))

        for item in lines:
            if isinstance(item, tuple):
                text, line_color = item
            else:
                text, line_color = item, "white"
            padded = self._pad(f" {text}", w - 2, "left")
            print(self._color(v, border_color) + self._color(padded, line_color) + self._color(v, border_color))

        print(self._color(bot, border_color))

    def _box_lines(
        self,
        title: str,
        lines: List[LineSpec],
        box_width: int,
        border_color: str = "white",
        title_color: str = "bright_white",
        double: bool = False,
    ) -> List[str]:
        """Create boxed lines to be rendered side-by-side."""
        b = self.BOX
        prefix = "d_" if double else ""
        h = b[f"{prefix}h"]
        v = b[f"{prefix}v"]
        tl = b[f"{prefix}tl"]
        tr = b[f"{prefix}tr"]
        bl = b[f"{prefix}bl"]
        br = b[f"{prefix}br"]
        ml = b[f"{prefix}ml"]
        mr = b[f"{prefix}mr"]

        top = tl + (h * (box_width - 2)) + tr
        mid = ml + (h * (box_width - 2)) + mr
        bot = bl + (h * (box_width - 2)) + br

        out: List[str] = []
        out.append(self._color(top, border_color))
        title_line = self._pad(f" {title} ", box_width - 2, "center")
        out.append(self._color(v, border_color) + self._color(title_line, title_color) + self._color(v, border_color))
        out.append(self._color(mid, border_color))

        for item in lines:
            if isinstance(item, tuple):
                text, line_color = item
            else:
                text, line_color = item, "white"
            padded = self._pad(f" {text}", box_width - 2, "left")
            out.append(self._color(v, border_color) + self._color(padded, line_color) + self._color(v, border_color))

        out.append(self._color(bot, border_color))
        return out

    def _print_two_boxes(
        self,
        left_title: str,
        left_lines: List[LineSpec],
        right_title: str,
        right_lines: List[LineSpec],
        left_color: str = "white",
        right_color: str = "white",
        double: bool = False,
    ):
        """Render two fixed-width boxes in one row."""
        gap = 2
        left_width = (self.width - gap) // 2
        right_width = self.width - gap - left_width

        left_body = list(left_lines)
        right_body = list(right_lines)
        max_body = max(len(left_body), len(right_body))
        if len(left_body) < max_body:
            left_body.extend([""] * (max_body - len(left_body)))
        if len(right_body) < max_body:
            right_body.extend([""] * (max_body - len(right_body)))

        left_box = self._box_lines(
            title=left_title,
            lines=left_body,
            box_width=left_width,
            border_color=left_color,
            title_color="bright_white",
            double=double,
        )
        right_box = self._box_lines(
            title=right_title,
            lines=right_body,
            box_width=right_width,
            border_color=right_color,
            title_color="bright_white",
            double=double,
        )

        print()
        for left, right in zip(left_box, right_box):
            print(f"{left}{' ' * gap}{right}")

    def clear_screen(self):
        os.system("cls" if os.name == "nt" else "clear")

    # ==========================================================================
    # Header / Footer
    # ==========================================================================

    def print_header(self, title: str, subtitle: str = ""):
        lines: List[LineSpec] = []
        if subtitle:
            lines.append((subtitle, "bright_cyan"))
        self._panel(title, lines, border_color="bright_blue", title_color="bright_white", double=True)

    def print_footer(self, status: str = "", next_update: str = ""):
        now = datetime.now().strftime("%H:%M:%S")
        footer = f"{self.SYMBOLS['clock']} {now}"
        if status:
            footer += f" | {status}"
        if next_update:
            footer += f" | Prox: {next_update}"
        footer += " | Ctrl+C para sair"
        self._panel("STATUS", [(footer, "dim")], border_color="dim", title_color="dim")

    # ==========================================================================
    # Market Panels
    # ==========================================================================

    def print_price_panel(
        self,
        symbol: str,
        price: float,
        change_24h: float = 0,
        high_24h: float = 0,
        low_24h: float = 0,
        volume: float = 0,
        variation_label: str = "24H",
    ):
        arrow = self.SYMBOLS["up"] if change_24h >= 0 else self.SYMBOLS["down"]
        change_color = "bright_green" if change_24h >= 0 else "bright_red"
        label = str(variation_label).upper()
        lines: List[LineSpec] = [
            (f"ATIVO: {symbol:<10} PRECO: {self._format_float(price)} USDT", "bright_white"),
            (f"VAR {label}: {arrow} {change_24h:+.2f}% | MAX: {self._format_float(high_24h)} | MIN: {self._format_float(low_24h)}", change_color),
        ]
        if volume > 0:
            lines.append((f"VOLUME: {self._format_float(volume, 0)}", "white"))
        self._panel("MERCADO", lines, border_color="bright_cyan", title_color="bright_white", double=True)

    def _signal_meta(self, signal: str) -> Tuple[str, str, str, str]:
        mapping = {
            "STRONG_BUY": (self.SYMBOLS["buy"], "COMPRA FORTE", "bright_green", "100%"),
            "BUY": (self.SYMBOLS["buy"], "COMPRA", "green", "75%"),
            "WEAK_BUY": (self.SYMBOLS["buy"], "COMPRA FRACA", "yellow", "50%"),
            "STRONG_SELL": (self.SYMBOLS["sell"], "VENDA FORTE", "bright_red", "100%"),
            "SELL": (self.SYMBOLS["sell"], "VENDA", "red", "75%"),
            "WEAK_SELL": (self.SYMBOLS["sell"], "VENDA FRACA", "yellow", "50%"),
            "HOLD": (self.SYMBOLS["hold"], "AGUARDAR", "white", "0%"),
            "NEUTRAL": (self.SYMBOLS["hold"], "NEUTRO", "white", "0%"),
        }
        return mapping.get(signal, mapping["HOLD"])

    def _confidence_pt(self, confidence: str) -> str:
        mapping = {
            "HIGH": "ALTA",
            "MEDIUM": "MEDIA",
            "LOW": "BAIXA",
            "NEUTRAL": "NEUTRA",
        }
        return mapping.get(str(confidence).upper(), str(confidence).upper())

    def print_signal_panel(
        self,
        signal: str,
        confidence: str,
        score: float,
        ensemble_score: float = 0,
        ml_prob: float = 0,
    ):
        sym, signal_text, color, size = self._signal_meta(signal)
        conf_text = self._confidence_pt(confidence)
        lines: List[LineSpec] = [
            (f"SINAL: {sym} {signal_text:<14} TAMANHO: {size}", color),
            (f"CONFIANCA: {conf_text:<7} SCORE: {score:+.3f} {self._score_bar(score)}", "white"),
        ]
        if ensemble_score != 0 or ml_prob != 0:
            lines.append((f"ENSEMBLE: {ensemble_score:+.3f} | ML: {ml_prob:.1%}", "bright_white"))
        self._panel("SINAL OPERACIONAL", lines, border_color=color, title_color="bright_white", double=True)

    def print_indicators_table(self, indicators: Dict[str, Any]):
        rsi = float(indicators.get("rsi", 50) or 50)
        macd = float(indicators.get("macd", 0) or 0)
        adx = float(indicators.get("adx", 0) or 0)
        obv = float(indicators.get("obv", 0) or 0)
        stoch = float(indicators.get("stoch_k", indicators.get("stoch", 0)) or 0)
        williams = float(indicators.get("williams_r", indicators.get("williams", 0)) or 0)

        if rsi > 70:
            rsi_status = "SOBRECOMPRA"
        elif rsi < 30:
            rsi_status = "SOBREVENDA"
        else:
            rsi_status = "NEUTRO"

        lines: List[LineSpec] = [
            f"RSI: {rsi:6.1f} {self._create_bar(rsi, 0, 100, 16)} {rsi_status}",
            f"ADX: {adx:6.1f} ({self._adx_strength(adx)}) | MACD: {macd:10.3f}",
            f"OBV: {self._format_float(obv, 0):>12} | STOCH: {stoch:6.1f} | WILLIAMS: {williams:6.1f}",
        ]
        self._panel("INDICADORES CHAVE", lines, border_color="bright_blue", title_color="bright_white")

    def print_ma_table(self, mas: Dict[str, float], current_price: float = 0):
        lines: List[LineSpec] = []
        for key, label in [("ema_10", "EMA10"), ("ema_20", "EMA20"), ("ema_50", "EMA50"), ("sma_200", "SMA200")]:
            value = float(mas.get(key, 0) or 0)
            if current_price > 0 and value > 0:
                trend = self.SYMBOLS["up"] if current_price >= value else self.SYMBOLS["down"]
            else:
                trend = self.SYMBOLS["flat"]
            lines.append(f"{label:<7} {self._format_float(value):>12} | PRECO x {label}: {trend}")

        if mas.get("golden_cross", False):
            lines.append(("SINAL: GOLDEN CROSS", "bright_green"))
        if mas.get("death_cross", False):
            lines.append(("SINAL: DEATH CROSS", "bright_red"))

        self._panel("MEDIAS MOVEIS", lines, border_color="bright_magenta", title_color="bright_white")

    def print_regime_panel(self, regime: str, vol_regime: str, vol_ratio: float):
        regime_text = str(regime).upper()
        vol_text = str(vol_regime).upper()

        if vol_ratio >= 1.5:
            leitura = "VOL ALTA"
            leitura_color = "bright_red"
        elif vol_ratio <= 0.8:
            leitura = "VOL BAIXA"
            leitura_color = "bright_green"
        else:
            leitura = "VOL NORMAL"
            leitura_color = "yellow"

        lines: List[LineSpec] = [
            (f"REGIME: {regime_text}", "bright_white"),
            (f"VOLATILIDADE: {vol_text} | RAZAO: {vol_ratio:.2f}", "white"),
            (f"LEITURA: {leitura}", leitura_color),
        ]
        self._panel("REGIME DE MERCADO", lines, border_color="bright_yellow", title_color="bright_white")

    def print_position_panel(self, position: Dict[str, Any]):
        if not position:
            return
        if "status" in position and position.get("status") != "open":
            return

        side = str(position.get("side", position.get("type", "LONG"))).upper()
        entry = float(position.get("entry_price", position.get("open_price", 0)) or 0)
        current = float(position.get("current_price", 0) or 0)
        qty = float(position.get("quantity", position.get("volume", 0)) or 0)
        sl = position.get("stop_loss", position.get("sl", None))
        tp = position.get("take_profit", position.get("tp", None))
        pnl_pct = position.get("unrealized_pnl_pct")
        if pnl_pct is None:
            try:
                pnl_pct = ((current - entry) / entry) * 100 if entry > 0 else 0.0
            except Exception:
                pnl_pct = 0.0

        pnl_color = "bright_green" if float(pnl_pct) >= 0 else "bright_red"
        pnl_arrow = self.SYMBOLS["up"] if float(pnl_pct) >= 0 else self.SYMBOLS["down"]

        lines: List[LineSpec] = [
            f"LADO: {side} | QTD: {qty:.6f}",
            f"ENTRADA: {self._format_float(entry)} | ATUAL: {self._format_float(current)}",
            (f"PNL: {pnl_arrow} {float(pnl_pct):+.2f}%", pnl_color),
            f"STOP: {self._format_float(sl)} | ALVO: {self._format_float(tp)}",
        ]
        self._panel("POSICAO ABERTA", lines, border_color="bright_cyan", title_color="bright_white", double=True)

    def print_trade_history(self, trades: List[Dict], max_trades: int = 5):
        if not trades:
            return

        recent = trades[-max_trades:] if len(trades) > max_trades else trades
        lines: List[LineSpec] = [
            "HORARIO   | SINAL        | PRECO        | TAMANHO",
            "-" * (self.width - 6),
        ]

        for trade in recent:
            raw_time = trade.get("time")
            if isinstance(raw_time, datetime):
                time_str = raw_time.strftime("%H:%M:%S")
            else:
                time_str = "--:--:--"

            signal = str(trade.get("signal", "HOLD"))
            price = float(trade.get("price", 0) or 0)
            size = float(trade.get("size", trade.get("volume", 0)) or 0)
            line = f"{time_str:<8} | {signal:<12} | {price:>12,.2f} | {size:>8.4f}"
            lines.append(line)

        self._panel(
            f"ULTIMAS OPERACOES ({len(trades)})",
            lines,
            border_color="dim",
            title_color="white",
        )

    def print_stats_panel(self, equity: float, initial: float, trades: int, wins: int = 0, losses: int = 0):
        pnl = equity - initial
        pnl_pct = ((pnl / initial) * 100) if initial > 0 else 0
        pnl_color = "bright_green" if pnl >= 0 else "bright_red"
        pnl_arrow = self.SYMBOLS["up"] if pnl >= 0 else self.SYMBOLS["down"]

        lines: List[LineSpec] = [
            f"CAPITAL INICIAL: {self._format_float(initial)} USDT",
            f"CAPITAL ATUAL:   {self._format_float(equity)} USDT",
            (f"RESULTADO:       {pnl_arrow} {pnl:+,.2f} ({pnl_pct:+.2f}%)", pnl_color),
            f"OPERACOES:       {trades}",
        ]

        if trades > 0:
            win_rate = (wins / trades) * 100 if trades > 0 else 0
            lines.append(f"VITORIAS:        {wins}")
            lines.append(f"DERROTAS:        {losses}")
            lines.append(f"TAXA ACERTO:     {win_rate:.1f}%")

        self._panel("DESEMPENHO", lines, border_color="bright_green", title_color="bright_white", double=True)

    # ==========================================================================
    # Backtest / Train / Optimize
    # ==========================================================================

    def print_backtest_header(self, symbol: str, days: int):
        self.print_header("MODO BACKTEST", f"{symbol} | {days} dias")

    def print_backtest_results(self, results: Dict[str, Any]):
        total_ret = float(results.get("total_return", 0) or 0)
        sharpe = results.get("sharpe_ratio", 0)
        max_dd = float(results.get("max_drawdown", 0) or 0)
        trades = int(results.get("total_trades", 0) or 0)
        win_rate = results.get("win_rate", 0)

        ret_color = "bright_green" if total_ret >= 0 else "bright_red"
        ret_arrow = self.SYMBOLS["up"] if total_ret >= 0 else self.SYMBOLS["down"]

        lines: List[LineSpec] = [
            (f"RETORNO TOTAL:   {ret_arrow} {total_ret:+.2f}%", ret_color),
            f"SHARPE:          {self._format_float(sharpe, 2)}",
            f"DRAWDOWN MAX:    {max_dd:.2f}%",
            f"TOTAL TRADES:    {trades}",
            f"WIN RATE:        {self._format_float(win_rate, 1)}%",
        ]
        self._panel("RESULTADO BACKTEST", lines, border_color="bright_green", title_color="bright_white", double=True)

    def print_training_header(self):
        self.print_header("MODO TREINO ML", "XGBOOST")

    def print_training_progress(self, fold: int, total: int, accuracy: float):
        bar = self._create_bar(fold, 0, max(total, 1), 32)
        lines: List[LineSpec] = [
            f"FOLD: {fold}/{total}",
            f"PROGRESSO: {bar}",
            f"ACURACIA: {accuracy:.3f}",
        ]
        self._panel("TREINO EM ANDAMENTO", lines, border_color="bright_blue", title_color="bright_white")

    def print_training_complete(self, metrics: Dict[str, float]):
        lines: List[LineSpec] = [
            f"ACURACIA:  {float(metrics.get('accuracy', 0) or 0):.3f}",
            f"PRECISAO:  {float(metrics.get('precision', 0) or 0):.3f}",
            f"RECALL:    {float(metrics.get('recall', 0) or 0):.3f}",
            f"F1 SCORE:  {float(metrics.get('f1', 0) or 0):.3f}",
        ]
        self._panel("TREINO FINALIZADO", lines, border_color="bright_green", title_color="bright_white", double=True)

    def print_optimization_header(self):
        self.print_header("MODO OTIMIZACAO", "OPTUNA")

    def print_optimization_progress(self, trial: int, total: int, best_value: float):
        bar = self._create_bar(trial, 0, max(total, 1), 32)
        lines: List[LineSpec] = [
            f"TRIAL: {trial}/{total}",
            f"PROGRESSO: {bar}",
            f"MELHOR VALOR: {best_value:.4f}",
        ]
        self._panel("OTIMIZACAO EM ANDAMENTO", lines, border_color="bright_magenta", title_color="bright_white")

    def print_optimization_complete(self, best_params: Dict[str, float], best_value: float):
        lines: List[LineSpec] = [(f"MELHOR VALOR: {best_value:.4f}", "bright_yellow")]
        for key, value in list(best_params.items())[:8]:
            lines.append(f"{key:<24} {float(value):.4f}")
        self._panel("OTIMIZACAO FINALIZADA", lines, border_color="bright_green", title_color="bright_white", double=True)

    # ==========================================================================
    # Full Demo Dashboard
    # ==========================================================================

    def print_terminal_trader_dashboard(self, data: Dict[str, Any]):
        """Render compact terminal-trader layout with fixed two-column blocks."""
        self.clear_screen()

        symbol = str(data.get("symbol", "BTCUSDT"))
        timestamp = data.get("timestamp", datetime.now())
        if isinstance(timestamp, datetime):
            ts_text = timestamp.strftime("%H:%M:%S")
        else:
            ts_text = datetime.now().strftime("%H:%M:%S")

        mode = str(data.get("mode", "DEMO")).upper()
        cycle = data.get("cycle", "")
        next_update = str(data.get("next_update", "60s"))

        price = float(data.get("price", 0) or 0)
        change = float(data.get("change_24h", 0) or 0)
        high_24h = float(data.get("high_24h", 0) or 0)
        low_24h = float(data.get("low_24h", 0) or 0)
        volume = float(data.get("volume", 0) or 0)
        variation_label = str(data.get("variation_label", "24H")).upper()

        signal = str(data.get("signal", "HOLD"))
        confidence = str(data.get("confidence", "LOW"))
        score = float(data.get("score", 0) or 0)
        ensemble_score = float(data.get("ensemble_score", 0) or 0)
        ml_prob = float(data.get("ml_prob", 0.5) or 0.5)

        regime = str(data.get("regime", "UNKNOWN"))
        vol_regime = str(data.get("vol_regime", data.get("volatility_regime", "NORMAL")))
        vol_ratio = float(data.get("vol_ratio", data.get("volatility_ratio", 1.0)) or 1.0)

        equity = float(data.get("equity", 0) or 0)
        initial = float(data.get("initial", 0) or 0)
        trades = int(data.get("trades", 0) or 0)

        self.print_header(
            "XAKCN TERMINAL TRADER",
            f"{symbol} | {mode} | {ts_text}",
        )

        change_arrow = self.SYMBOLS["up"] if change >= 0 else self.SYMBOLS["down"]
        market_lines: List[LineSpec] = [
            f"ATIVO: {symbol}",
            f"PRECO: {self._format_float(price)} USDT",
            (f"VAR {variation_label}: {change_arrow} {change:+.2f}%", "bright_green" if change >= 0 else "bright_red"),
            f"MAX/MIN {variation_label}: {self._format_float(high_24h)} / {self._format_float(low_24h)}",
        ]
        if volume > 0:
            market_lines.append(f"VOLUME: {self._format_float(volume, 0)}")

        sym, signal_text, signal_color, size = self._signal_meta(signal)
        signal_lines: List[LineSpec] = [
            (f"SINAL: {sym} {signal_text}", signal_color),
            f"TAMANHO BASE: {size}",
            f"CONFIANCA: {self._confidence_pt(confidence)}",
            (f"SCORE: {score:+.3f}", signal_color),
            f"FORCA: {self._score_bar(score, width=18)}",
            f"ENS/ML: {ensemble_score:+.3f} / {ml_prob:.1%}",
        ]
        self._print_two_boxes(
            left_title="MERCADO",
            left_lines=market_lines,
            right_title="SINAL",
            right_lines=signal_lines,
            left_color="bright_cyan",
            right_color=signal_color,
            double=True,
        )

        indicators = data.get("indicators", {}) or {}
        components = data.get("components", {}) or {}
        ind_lines: List[LineSpec] = []

        if indicators:
            rsi = float(indicators.get("rsi", 50) or 50)
            adx = float(indicators.get("adx", 0) or 0)
            macd = float(indicators.get("macd", 0) or 0)
            obv = float(indicators.get("obv", 0) or 0)
            ind_lines = [
                f"RSI: {rsi:6.1f} {self._create_bar(rsi, 0, 100, 12)}",
                f"ADX: {adx:6.1f} ({self._adx_strength(adx)})",
                f"MACD: {macd:+.3f}",
                f"OBV: {self._format_float(obv, 0)}",
            ]
        elif components:
            ind_lines = [
                f"RSI score: {float(components.get('rsi', 0) or 0):+.2f}",
                f"MACD score: {float(components.get('macd', 0) or 0):+.2f}",
                f"EMA score: {float(components.get('ema', 0) or 0):+.2f}",
                f"ADX score: {float(components.get('adx', 0) or 0):+.2f}",
                f"OBV score: {float(components.get('obv', 0) or 0):+.2f}",
            ]
        else:
            ind_lines = ["Sem indicadores para exibir"]

        regime_lines: List[LineSpec] = [
            f"REGIME: {regime}",
            f"VOL REGIME: {vol_regime}",
            f"VOL RAZAO: {vol_ratio:.2f}",
            ("LEITURA: VOL ALTA", "bright_red") if vol_ratio >= 1.5
            else ("LEITURA: VOL BAIXA", "bright_green") if vol_ratio <= 0.8
            else ("LEITURA: VOL NORMAL", "yellow"),
        ]
        self._print_two_boxes(
            left_title="INDICADORES",
            left_lines=ind_lines,
            right_title="REGIME",
            right_lines=regime_lines,
            left_color="bright_blue",
            right_color="bright_yellow",
        )

        pnl = equity - initial
        pnl_pct = ((pnl / initial) * 100) if initial > 0 else 0.0
        pnl_color = "bright_green" if pnl >= 0 else "bright_red"
        pnl_arrow = self.SYMBOLS["up"] if pnl >= 0 else self.SYMBOLS["down"]
        stats_lines: List[LineSpec] = [
            f"CAPITAL INICIAL: {self._format_float(initial)} USDT",
            f"CAPITAL ATUAL:   {self._format_float(equity)} USDT",
            (f"PNL: {pnl_arrow} {pnl:+,.2f} ({pnl_pct:+.2f}%)", pnl_color),
            f"TRADES: {trades}",
        ]

        last_trade_lines: List[LineSpec] = []
        history = data.get("trade_history", []) or []
        if history:
            for trade in history[-3:]:
                raw_time = trade.get("time")
                t = raw_time.strftime("%H:%M:%S") if isinstance(raw_time, datetime) else "--:--:--"
                sig = str(trade.get("signal", "HOLD"))
                pr = float(trade.get("price", 0) or 0)
                sz = float(trade.get("size", trade.get("volume", 0)) or 0)
                last_trade_lines.append(f"{t} {sig:<11} {pr:>10,.2f} {sz:>8.4f}")
        else:
            last_trade_lines.append("Sem operacoes recentes")

        self._print_two_boxes(
            left_title="DESEMPENHO",
            left_lines=stats_lines,
            right_title="ULTIMAS OPERACOES",
            right_lines=last_trade_lines,
            left_color="bright_green",
            right_color="dim",
            double=True,
        )

        status = f"Modo {mode}"
        if cycle != "":
            status += f" | Ciclo {cycle}"
        self.print_footer(status=status, next_update=next_update)

    def print_demo_dashboard(self, data: Dict[str, Any]):
        dashboard_data = dict(data)
        dashboard_data["mode"] = "DEMO"
        dashboard_data["timestamp"] = datetime.now()
        dashboard_data.setdefault("initial", 0.0)
        dashboard_data.setdefault("equity", 0.0)
        dashboard_data.setdefault("trades", 0)
        dashboard_data.setdefault("trade_history", [])
        dashboard_data.setdefault("ensemble_score", 0.0)
        dashboard_data.setdefault("ml_prob", 0.5)
        self.print_terminal_trader_dashboard(dashboard_data)


# Global instance
visual = VisualLogger()
