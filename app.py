
import os
import re
import ast
import math
import time
import base64
import operator
import tempfile
import logging
import threading
import uuid

import gradio as gr
import requests
import numpy as np
import sympy as sp
import pint

from scipy import constants
from PIL import Image
from google import genai
from google.genai import types


# ============================================================
# 1. CONFIGURATION
# ============================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("alpha")

PORT = int(os.environ.get("PORT", "10000"))

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
GEMINI_BACKUP_MODEL = os.environ.get(
    "GEMINI_BACKUP_MODEL",
    "gemini-3.5-flash-lite"
)

CLOUDFLARE_API_TOKEN = os.environ.get("CLOUDFLARE_API_TOKEN", "")
CLOUDFLARE_ACCOUNT_ID = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "")

IMAGE_MODEL = os.environ.get(
    "IMAGE_MODEL",
    "@cf/black-forest-labs/flux-1-schnell"
)

CHAT_HISTORY_LIMIT = 8
MAX_OUTPUT_TOKENS = 900
GEMINI_TIMEOUT_MS = 45000

SYSTEM_PROMPT = """
You are Alpha, a helpful and accurate AI assistant.

- Answer the actual question directly.
- Use simple and clear language.
- Avoid unnecessary introductions.
- Explain difficult topics step by step.
- Do not invent facts or sources.
- If uncertain, clearly say so.
- Keep simple answers concise.
"""

NUMERICAL_PROMPT = """
You are Alpha's Universal Numerical Solver.

You solve numerical questions from all subjects and fields:

- Mathematics
- Arithmetic and algebra
- Geometry and trigonometry
- Calculus
- Statistics and probability
- Physics
- Chemistry
- Engineering
- Astronomy
- Finance
- Computer science
- Everyday calculations
- Other scientific and technical fields

For numerical questions:

1. Identify the given values.
2. Identify what needs to be calculated.
3. Select the correct formula.
4. Substitute the values.
5. Calculate carefully.
6. Include correct units.
7. Verify the result.
8. Give the final answer clearly.

Never invent missing information.
Never pretend an uncertain answer is verified.
If information is missing, ask for clarification.
"""

# Reusable clients
client = None
client_lock = threading.Lock()

unit_registry = pint.UnitRegistry()

if GEMINI_API_KEY:
    try:
        client = genai.Client(
            api_key=GEMINI_API_KEY,
            http_options=types.HttpOptions(
                timeout=GEMINI_TIMEOUT_MS
            )
        )
        logger.info("Gemini client initialized.")

    except Exception:
        logger.exception("Gemini initialization failed.")


# ============================================================
# 2. SAFE MATHEMATICAL EXPRESSION ENGINE
# ============================================================

OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

SYMPY_OPERATORS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.Pow: lambda a, b: a ** b,
}

SYMPY_FUNCTIONS = {
    "sqrt": sp.sqrt,
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "asin": sp.asin,
    "acos": sp.acos,
    "atan": sp.atan,
    "log": sp.log,
    "ln": sp.log,
    "exp": sp.exp,
    "abs": sp.Abs,
    "factorial": sp.factorial,
}

SYMPY_CONSTANTS = {
    "pi": sp.pi,
    "e": sp.E,
}

SCIENTIFIC_CONSTANTS = {
    "c": constants.c,
    "G": constants.G,
    "h": constants.h,
    "k_B": constants.k,
    "N_A": constants.Avogadro,
    "R": constants.R,
}


def clean_expression(expression):
    expression = expression.strip()

    expression = expression.replace("^", "**")
    expression = expression.replace("×", "*")
    expression = expression.replace("÷", "/")

    # Convert 2x into 2*x.
    expression = re.sub(
        r"(\d)([a-zA-Z])",
        r"\1*\2",
        expression
    )

    # Convert x(x+1) into x*(x+1).
    expression = re.sub(
        r"([a-zA-Z0-9)])\(",
        r"\1*(",
        expression
    )

    return expression


def safe_sympy_node(node):
    if isinstance(node, ast.Expression):
        return safe_sympy_node(node.body)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return sp.sympify(node.value)

        raise ValueError("Invalid constant")

    if isinstance(node, ast.Name):
        name = node.id

        if name in SYMPY_CONSTANTS:
            return SYMPY_CONSTANTS[name]

        if name in SCIENTIFIC_CONSTANTS:
            return sp.Float(SCIENTIFIC_CONSTANTS[name])

        if re.fullmatch(r"[a-zA-Z]", name):
            return sp.Symbol(name)

        raise ValueError("Unsupported variable")

    if isinstance(node, ast.UnaryOp):
        operation = node.op

        if isinstance(operation, ast.USub):
            return -safe_sympy_node(node.operand)

        if isinstance(operation, ast.UAdd):
            return safe_sympy_node(node.operand)

        raise ValueError("Unsupported unary operation")

    if isinstance(node, ast.BinOp):
        operation = SYMPY_OPERATORS.get(type(node.op))

        if operation is None:
            raise ValueError("Unsupported operation")

        left = safe_sympy_node(node.left)
        right = safe_sympy_node(node.right)

        if isinstance(node.op, ast.Pow):
            if right.is_number and abs(float(right)) > 20:
                raise ValueError("Exponent too large")

        return operation(left, right)

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Unsupported function")

        function = SYMPY_FUNCTIONS.get(node.func.id)

        if function is None:
            raise ValueError("Unsupported function")

        if node.keywords or len(node.args) != 1:
            raise ValueError("Invalid function arguments")

        return function(safe_sympy_node(node.args[0]))

    raise ValueError("Unsupported mathematical expression")


def parse_safe_expression(expression):
    expression = clean_expression(expression)

    if len(expression) > 250:
        raise ValueError("Expression too long")

    if not re.fullmatch(
        r"[0-9a-zA-Z_+\-*/().,%\s]+",
        expression
    ):
        raise ValueError("Invalid expression characters")

    tree = ast.parse(expression, mode="eval")

    return safe_sympy_node(tree)


# ============================================================
# 3. FAST ARITHMETIC
# ============================================================

def safe_arithmetic_node(node):
    if isinstance(node, ast.Expression):
        return safe_arithmetic_node(node.body)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value

        raise ValueError("Invalid number")

    if isinstance(node, ast.UnaryOp):
        operation = OPERATORS.get(type(node.op))

        if operation is None:
            raise ValueError("Invalid operation")

        return operation(safe_arithmetic_node(node.operand))

    if isinstance(node, ast.BinOp):
        operation = OPERATORS.get(type(node.op))

        if operation is None:
            raise ValueError("Invalid operation")

        left = safe_arithmetic_node(node.left)
        right = safe_arithmetic_node(node.right)

        if isinstance(node.op, ast.Pow):
            if abs(right) > 10 or abs(left) > 100000:
                raise ValueError("Calculation too large")

        result = operation(left, right)

        if isinstance(result, complex):
            raise ValueError("Invalid result")

        if isinstance(result, float) and not math.isfinite(result):
            raise ValueError("Invalid result")

        if abs(result) > 10**100:
            raise ValueError("Result too large")

        return result

    raise ValueError("Unsupported expression")


def extract_math_expression(message):
    text = message.strip().lower()

    if re.fullmatch(r"[\d\s.+\-*/()%]+", text):
        return text

    patterns = [
        r"^(?:what is|calculate|solve|compute)\s+(.+?)\??$",
        r"^(?:please calculate|please solve)\s+(.+?)\??$",
    ]

    for pattern in patterns:
        match = re.match(pattern, text)

        if match:
            expression = match.group(1).strip()

            if re.fullmatch(r"[\d\s.+\-*/()%]+", expression):
                return expression

    return None


def format_result(result):
    if isinstance(result, sp.Equality):
        return sp.pretty(result)

    if isinstance(result, sp.Basic):
        if result.is_number:
            return str(sp.N(result, 12))

        return sp.pretty(result)

    if isinstance(result, float):
        if not math.isfinite(result):
            raise ValueError("Invalid result")

        if result.is_integer():
            return str(int(result))

        return str(round(result, 10))

    return str(result)


def calculate_arithmetic(message):
    expression = extract_math_expression(message)

    if not expression:
        return None

    try:
        tree = ast.parse(expression, mode="eval")
        result = safe_arithmetic_node(tree)

        return format_result(result)

    except Exception:
        return None


# ============================================================
# 4. UNIVERSAL LOCAL NUMERICAL ENGINE
# ============================================================

def solve_equation_locally(message):
    text = message.strip()

    match = re.search(
        r"(?:solve(?:\s+for\s+[a-zA-Z])?[:\s]+)(.+?=\s*.+)$",
        text,
        re.IGNORECASE
    )

    if not match:
        return None

    equation_text = match.group(1).strip()

    if equation_text.count("=") != 1:
        return None

    left_text, right_text = equation_text.split("=", 1)

    try:
        left = parse_safe_expression(left_text)
        right = parse_safe_expression(right_text)

        equation = sp.Eq(left, right)

        variables = sorted(
            equation.free_symbols,
            key=lambda item: item.name
        )

        if not variables:
            return None

        if len(variables) > 3:
            return None

        solutions = sp.solve(
            equation,
            variables,
            dict=True
        )

        if not solutions:
            return "No solution was found."

        lines = ["**Solution:**"]

        for index, solution in enumerate(solutions[:10], 1):
            values = []

            for variable in variables:
                if variable in solution:
                    values.append(
                        f"{variable} = {format_result(solution[variable])}"
                    )

            if values:
                lines.append(
                    f"{index}. " + ", ".join(values)
                )

        return "\n".join(lines)

    except Exception as error:
        logger.debug("Local equation solver skipped: %s", error)
        return None


def solve_calculus_locally(message):
    text = message.strip().lower()

    differentiate_patterns = [
        r"^(?:differentiate|find the derivative of|derivative of)\s+(.+)$",
    ]

    integrate_patterns = [
        r"^(?:integrate|find the integral of|integral of)\s+(.+)$",
    ]

    for pattern in differentiate_patterns:
        match = re.match(pattern, text)

        if match:
            expression_text = match.group(1)

            try:
                expression = parse_safe_expression(expression_text)
                variables = sorted(
                    expression.free_symbols,
                    key=lambda item: item.name
                )

                variable = variables[0] if variables else sp.Symbol("x")

                result = sp.diff(expression, variable)

                return (
                    "**Derivative:**\n\n"
                    + f"`{sp.pretty(result)}`"
                )

            except Exception:
                return None

    for pattern in integrate_patterns:
        match = re.match(pattern, text)

        if match:
            expression_text = match.group(1)

            try:
                expression = parse_safe_expression(expression_text)
                variables = sorted(
                    expression.free_symbols,
                    key=lambda item: item.name
                )

                variable = variables[0] if variables else sp.Symbol("x")

                result = sp.integrate(expression, variable)

                return (
                    "**Integral:**\n\n"
                    + f"`{sp.pretty(result)} + C`"
                )

            except Exception:
                return None

    return None


def solve_statistics_locally(message):
    text = message.strip().lower()

    match = re.search(
        r"(?:mean|average|median|standard deviation|std)\s+(?:of\s+)?"
        r"([-+\d.,\s]+)$",
        text
    )

    if not match:
        return None

    try:
        values = [
            float(value)
            for value in re.findall(
                r"[-+]?\d+(?:\.\d+)?",
                match.group(1)
            )
        ]

        if not values or len(values) > 10000:
            return None

        data = np.array(values, dtype=float)

        if "median" in text:
            result = np.median(data)
            label = "Median"

        elif "standard deviation" in text or "std" in text:
            result = np.std(data)
            label = "Standard deviation"

        else:
            result = np.mean(data)
            label = "Mean"

        return f"**{label}:** {format_result(float(result))}"

    except Exception:
        return None


def convert_units_locally(message):
    match = re.match(
        r"^\s*convert\s+([-+]?\d*\.?\d+)\s+(.+?)\s+to\s+(.+?)\s*$",
        message,
        re.IGNORECASE
    )

    if not match:
        return None

    value, source_unit, target_unit = match.groups()

    try:
        quantity = unit_registry.Quantity(
            float(value),
            source_unit.strip()
        )

        converted = quantity.to(target_unit.strip())

        return f"**Answer:** {converted:~P}"

    except Exception:
        return None


def solve_local_numerical(message):
    # Try fast arithmetic first.
    result = calculate_arithmetic(message)

    if result is not None:
        return result

    # Scientific unit conversion.
    result = convert_units_locally(message)

    if result is not None:
        return result

    # Statistics.
    result = solve_statistics_locally(message)

    if result is not None:
        return result

    # Calculus.
    result = solve_calculus_locally(message)

    if result is not None:
        return result

    # Algebra and equations.
    result = solve_equation_locally(message)

    if result is not None:
        return result

    return None


# ============================================================
# 5. NUMERICAL QUESTION DETECTOR
# ============================================================

NUMERICAL_KEYWORDS = [
    "calculate", "compute", "solve", "evaluate",
    "find the value", "find x", "find the answer",
    "numerical", "equation", "percentage", "percent",
    "ratio", "proportion", "fraction", "average",
    "probability", "statistics", "mean", "median",
    "standard deviation", "algebra", "calculus",
    "differentiate", "derivative", "integrate",
    "integration", "integral",
    "velocity", "acceleration", "distance",
    "displacement", "force", "work done", "power",
    "energy", "momentum", "current", "voltage",
    "resistance", "wavelength", "frequency",
    "pressure", "density", "temperature",
    "heat", "specific heat", "calorimetry",
    "mole", "molarity", "molality",
    "concentration", "equilibrium",
    "enthalpy", "entropy", "gibbs energy",
    "stoichiometry", "mass", "volume",
    "surface area", "perimeter", "area",
    "trigonometry", "logarithm",
    "compound interest", "simple interest",
    "unit conversion", "significant figures"
]


def is_numerical_question(message):
    text = message.lower()

    has_number = bool(re.search(r"\d", text))

    has_keyword = any(
        keyword in text for keyword in NUMERICAL_KEYWORDS
    )

    # Explicitly numerical requests can also contain
    # equations without digits, such as "solve x + 2 = 5".
    equation_request = bool(
        re.search(r"\b(solve|calculate|compute|evaluate)\b", text)
        and "=" in text
    )

    return has_keyword or (has_number and has_keyword) or equation_request


# ============================================================
# 6. GEMINI API
# ============================================================

def check_client():
    if client is None:
        raise RuntimeError(
            "Gemini is not configured. Check GEMINI_API_KEY in Render."
        )


def generation_config(numerical=False):
    prompt = NUMERICAL_PROMPT if numerical else SYSTEM_PROMPT

    return types.GenerateContentConfig(
        system_instruction=prompt,
        max_output_tokens=MAX_OUTPUT_TOKENS
    )


def is_temporary_error(error):
    message = str(error).upper()

    indicators = [
        "503",
        "UNAVAILABLE",
        "429",
        "RESOURCE_EXHAUSTED",
        "500",
        "502",
        "504",
        "TIMEOUT",
        "INTERNAL"
    ]

    return any(item in message for item in indicators)


def get_models():
    models = [GEMINI_MODEL]

    if GEMINI_BACKUP_MODEL and GEMINI_BACKUP_MODEL != GEMINI_MODEL:
        models.append(GEMINI_BACKUP_MODEL)

    return models


def generate_with_fallback(contents, numerical=False):
    check_client()

    last_error = None
    models = get_models()

    for index, model_name in enumerate(models):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=contents,
                config=generation_config(numerical)
            )

            if response and response.text:
                return response.text

            raise RuntimeError("Empty Gemini response.")

        except Exception as error:
            last_error = error

            logger.warning(
                "Gemini model %s failed: %s",
                model_name,
                str(error)[:250]
            )

            if not is_temporary_error(error):
                break

            if index < len(models) - 1:
                time.sleep(0.2)

    raise RuntimeError(
        "Alpha could not get a response from Gemini. "
        + str(last_error)[:250]
    )


def stream_with_fallback(contents, numerical=False):
    check_client()

    last_error = None
    models = get_models()

    for index, model_name in enumerate(models):
        received_text = False

        try:
            stream = client.models.generate_content_stream(
                model=model_name,
                contents=contents,
                config=generation_config(numerical)
            )

            for chunk in stream:
                chunk_text = getattr(chunk, "text", None)

                if chunk_text:
                    received_text = True
                    yield chunk_text

            if not received_text:
                raise RuntimeError("Empty streaming response.")

            return

        except Exception as error:
            last_error = error

            logger.warning(
                "Streaming failed for %s: %s",
                model_name,
                str(error)[:250]
            )

            if received_text:
                raise RuntimeError(
                    "The response stream stopped unexpectedly."
                )

            if not is_temporary_error(error):
                break

            if index < len(models) - 1:
                time.sleep(0.2)

    raise RuntimeError(
        
