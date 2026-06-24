"""Pure-computation deterministic tools: calculator and unit conversion.

These tools never call a model or the network. Their results are passed to
the selected model as trusted facts so arithmetic is never hallucinated.
"""

from __future__ import annotations

import ast
import operator
import re
from collections.abc import Callable

from switchboard.app.models.capabilities import Capability, ToolResult

_BIN_OPS: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPS: dict[type[ast.unaryop], Callable[[float], float]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

_PERCENT_OF = re.compile(r"(\d+(?:\.\d+)?)\s*(?:%|percent)\s+of\s+([-+]?\d+(?:\.\d+)?)")
_SQUARE_ROOT = re.compile(r"square root of\s+(\d+(?:\.\d+)?)")
_EXPRESSION = re.compile(
    r"(?<![\w.])\(*[-+]?\d+(?:\.\d+)?(?:\s*[-+*/x×^]\s*\(*[-+]?\d+(?:\.\d+)?\)*)+(?![\w.])"
)

# Spoken-operator normalization so "87 divided by 4" parses like "87 / 4".
_WORD_OPERATORS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bdivided by\b"), " / "),
    (re.compile(r"\bmultiplied by\b"), " * "),
    (re.compile(r"\btimes\b"), " * "),
    (re.compile(r"\bplus\b"), " + "),
    (re.compile(r"\bminus\b"), " - "),
    (re.compile(r"\bto the power of\b"), " ^ "),
)

_MAX_POW_OPERAND = 1000


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, int | float):
            return float(node.value)
        raise ValueError("Only numbers are allowed.")
    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _BIN_OPS:
            raise ValueError("Operator not allowed.")
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        if op_type is ast.Pow and (abs(left) > _MAX_POW_OPERAND or abs(right) > 64):
            raise ValueError("Exponent too large.")
        return _BIN_OPS[op_type](left, right)
    if isinstance(node, ast.UnaryOp):
        unary_type = type(node.op)
        if unary_type not in _UNARY_OPS:
            raise ValueError("Operator not allowed.")
        return _UNARY_OPS[unary_type](_safe_eval(node.operand))
    raise ValueError("Expression not allowed.")


def format_number(value: float) -> str:
    if value == int(value) and abs(value) < 1e15:
        return f"{int(value):,}"
    return f"{value:,.6f}".rstrip("0").rstrip(".")


class CalculatorTool:
    """Evaluates arithmetic deterministically via a restricted AST walk."""

    def answer(self, prompt: str) -> ToolResult:
        text = prompt.lower().replace(",", "")
        for pattern, replacement in _WORD_OPERATORS:
            text = pattern.sub(replacement, text)
        try:
            percent = _PERCENT_OF.search(text)
            if percent:
                fraction = float(percent.group(1))
                base = float(percent.group(2))
                result = fraction * base / 100.0
                statement = f"{format_number(fraction)}% of {format_number(base)}"
                return self._success(statement, result)

            root = _SQUARE_ROOT.search(text)
            if root:
                value = float(root.group(1))
                return self._success(
                    f"the square root of {format_number(value)}", value**0.5
                )

            match = _EXPRESSION.search(text)
            if not match:
                return self._failure("No arithmetic expression found.")
            expression = (
                match.group(0).replace("x", "*").replace("×", "*").replace("^", "**")
            )
            result = _safe_eval(ast.parse(expression, mode="eval"))
            return self._success(match.group(0).strip(), result)
        except ZeroDivisionError:
            return self._failure("Division by zero.")
        except (ValueError, SyntaxError, OverflowError) as exc:
            return self._failure(f"Could not evaluate expression: {exc}")

    def _success(self, statement: str, result: float) -> ToolResult:
        return ToolResult(
            tool_name="calculator",
            capability=Capability.CALCULATION,
            answer=f"Calculated locally: {statement} = {format_number(result)}.",
            display_model_or_label="Calculator",
            metadata={"calculator_result": format_number(result)},
        )

    def _failure(self, error: str) -> ToolResult:
        return ToolResult(
            tool_name="calculator",
            capability=Capability.CALCULATION,
            answer="",
            success=False,
            error=error,
            display_model_or_label="Calculator",
        )


# Canonical unit table: name -> (kind, factor to canonical unit of that kind).
# Canonical units: meter (length), kilogram (mass), liter (volume).
_UNITS: dict[str, tuple[str, float]] = {
    "km": ("length", 1000.0),
    "kilometer": ("length", 1000.0),
    "kilometers": ("length", 1000.0),
    "kilometre": ("length", 1000.0),
    "kilometres": ("length", 1000.0),
    "m": ("length", 1.0),
    "meter": ("length", 1.0),
    "meters": ("length", 1.0),
    "metre": ("length", 1.0),
    "metres": ("length", 1.0),
    "cm": ("length", 0.01),
    "centimeter": ("length", 0.01),
    "centimeters": ("length", 0.01),
    "centimetre": ("length", 0.01),
    "centimetres": ("length", 0.01),
    "mile": ("length", 1609.344),
    "miles": ("length", 1609.344),
    "mi": ("length", 1609.344),
    "ft": ("length", 0.3048),
    "foot": ("length", 0.3048),
    "feet": ("length", 0.3048),
    "in": ("length", 0.0254),
    "inch": ("length", 0.0254),
    "inches": ("length", 0.0254),
    "kg": ("mass", 1.0),
    "kilogram": ("mass", 1.0),
    "kilograms": ("mass", 1.0),
    "kilos": ("mass", 1.0),
    "g": ("mass", 0.001),
    "gram": ("mass", 0.001),
    "grams": ("mass", 0.001),
    "lb": ("mass", 0.45359237),
    "lbs": ("mass", 0.45359237),
    "pound": ("mass", 0.45359237),
    "pounds": ("mass", 0.45359237),
    "oz": ("mass", 0.028349523125),
    "ounce": ("mass", 0.028349523125),
    "ounces": ("mass", 0.028349523125),
    "l": ("volume", 1.0),
    "liter": ("volume", 1.0),
    "liters": ("volume", 1.0),
    "litre": ("volume", 1.0),
    "litres": ("volume", 1.0),
    "gal": ("volume", 3.785411784),
    "gallon": ("volume", 3.785411784),
    "gallons": ("volume", 3.785411784),
}

_TEMPERATURE_UNITS = {"c", "celsius", "centigrade", "f", "fahrenheit"}

_CONVERSION = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:degrees?\s+)?([a-z]+)\s*(?:in|to|into|as)\s+(?:degrees?\s+)?([a-z]+)"
)

# Number-free conversions ("convert feet into miles", "ounces to grams"):
# answered with the unit rate (value = 1).
_RATE_CONVERSION = re.compile(r"\b([a-z]+)\s+(?:in|to|into|as)\s+([a-z]+)\b")


class UnitConversionTool:
    """Converts between common units deterministically."""

    def answer(self, prompt: str) -> ToolResult:
        text = prompt.lower().replace(",", "")
        match = _CONVERSION.search(text)
        if match:
            value = float(match.group(1))
            source = match.group(2)
            target = match.group(3)
        else:
            rate = self._find_rate_conversion(text)
            if rate is None:
                return self._failure("No conversion expression found.")
            value, source, target = 1.0, rate[0], rate[1]

        if source in _TEMPERATURE_UNITS and target in _TEMPERATURE_UNITS:
            return self._temperature(value, source, target)

        source_unit = _UNITS.get(source)
        target_unit = _UNITS.get(target)
        if source_unit is None or target_unit is None:
            return self._failure(f"Unknown unit: {source if source_unit is None else target}.")
        if source_unit[0] != target_unit[0]:
            return self._failure(
                f"Cannot convert {source} ({source_unit[0]}) to {target} ({target_unit[0]})."
            )
        result = value * source_unit[1] / target_unit[1]
        return self._success(value, source, result, target)

    def _find_rate_conversion(self, text: str) -> tuple[str, str] | None:
        """Find a unit->unit pair with no quantity; both sides must be known
        units of the same kind so ordinary prose never matches."""
        for match in _RATE_CONVERSION.finditer(text):
            source, target = match.group(1), match.group(2)
            source_unit = _UNITS.get(source)
            target_unit = _UNITS.get(target)
            if (
                source_unit is not None
                and target_unit is not None
                and source_unit[0] == target_unit[0]
                and source != target
            ):
                return source, target
        return None

    def _temperature(self, value: float, source: str, target: str) -> ToolResult:
        source_is_celsius = source in {"c", "celsius", "centigrade"}
        target_is_celsius = target in {"c", "celsius", "centigrade"}
        if source_is_celsius == target_is_celsius:
            result = value
        elif source_is_celsius:
            result = value * 9 / 5 + 32
        else:
            result = (value - 32) * 5 / 9
        source_label = "°C" if source_is_celsius else "°F"
        target_label = "°C" if target_is_celsius else "°F"
        return ToolResult(
            tool_name="unit_conversion",
            capability=Capability.UNIT_CONVERSION,
            answer=(
                f"Converted locally: {format_number(value)}{source_label} = "
                f"{format_number(round(result, 4))}{target_label}."
            ),
            display_model_or_label="Converter",
        )

    def _success(self, value: float, source: str, result: float, target: str) -> ToolResult:
        return ToolResult(
            tool_name="unit_conversion",
            capability=Capability.UNIT_CONVERSION,
            answer=(
                f"Converted locally: {format_number(value)} {source} = "
                f"{format_number(round(result, 6))} {target}."
            ),
            display_model_or_label="Converter",
        )

    def _failure(self, error: str) -> ToolResult:
        return ToolResult(
            tool_name="unit_conversion",
            capability=Capability.UNIT_CONVERSION,
            answer="",
            success=False,
            error=error,
            display_model_or_label="Converter",
        )
