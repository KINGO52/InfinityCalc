import numpy as np
import sympy as sp
from scipy.special import lambertw
from typing import Union, List, Tuple, Dict, Callable
import math
import mpmath
from mpmath import mp
from functools import lru_cache

class CalculatorCore:
    def __init__(self):
        self.x = sp.Symbol('x')
        self.memory = 0
        self.precision = 50  # Default precision
        mp.dps = self.precision
        mp.pretty = True
        self._setup_math_dict()
        
    def set_precision(self, digits: int) -> None:
        """Set the precision for calculations."""
        self.precision = max(10, min(1000000, digits))  # Limit between 10 and 1M digits
        mp.dps = self.precision
        self._setup_math_dict()  # Refresh math dictionary with new precision
        
    def _is_simple_calculation(self, expression: str) -> bool:
        """Check if this is a simple calculation that can use standard float."""
        simple_ops = {'+', '-', '*', '/', '**', '^', '(', ')', '.', 'e'}
        return all(c.isdigit() or c.isspace() or c in simple_ops for c in expression)
        
    def _setup_math_dict(self) -> None:
        """Initialize math dictionary once for better performance."""
        self.safe_dict: Dict[str, Callable] = {
            "sin": lambda x: float(mp.sin(x)) if abs(float(x)) < 1000 else mp.sin(x),
            "cos": lambda x: float(mp.cos(x)) if abs(float(x)) < 1000 else mp.cos(x),
            "tan": lambda x: float(mp.tan(x)) if abs(float(x)) < 1000 else mp.tan(x),
            "asin": lambda x: float(mp.asin(x)) if abs(float(x)) <= 1 else mp.asin(x),
            "acos": lambda x: float(mp.acos(x)) if abs(float(x)) <= 1 else mp.acos(x),
            "atan": lambda x: float(mp.atan(x)) if abs(float(x)) < 1000 else mp.atan(x),
            "sinh": lambda x: float(mp.sinh(x)) if abs(float(x)) < 10 else mp.sinh(x),
            "cosh": lambda x: float(mp.cosh(x)) if abs(float(x)) < 10 else mp.cosh(x),
            "tanh": lambda x: float(mp.tanh(x)) if abs(float(x)) < 10 else mp.tanh(x),
            "exp": lambda x: float(mp.exp(x)) if abs(float(x)) < 100 else mp.exp(x),
            "log": lambda x: float(mp.log10(x)) if 0 < float(x) < 1e100 else mp.log10(x),
            "ln": lambda x: float(mp.log(x)) if 0 < float(x) < 1e100 else mp.log(x),
            "loga": lambda a, b: float(mp.log(b) / mp.log(a)) if 0 < float(b) < 1e100 and float(a) > 0 else mp.log(b) / mp.log(a),
            "sqrt": lambda x: float(mp.sqrt(x)) if float(x) >= 0 and float(x) < 1e100 else mp.sqrt(x),
            "pi": mp.pi,
            "e": mp.e,
            "abs": abs,
            "factorial": self._safe_factorial,
            "W": lambda x: float(lambertw(float(x), 0).real),
            "tetration": self._cached_tetration,
            "power": self._safe_power
        }
        
    @staticmethod
    def _safe_factorial(x: Union[int, float]) -> Union[float, str]:
        """Optimized factorial with size limit."""
        try:
            n = int(x)
            if n < 0:
                return "Error: Factorial undefined for negative numbers"
            if n > 1000:
                return float('inf')
            return mp.factorial(n)
        except:
            return "Error: Invalid input for factorial"
    
    @staticmethod
    def _safe_power(x: Union[float, mp.mpf], y: Union[float, mp.mpf]) -> Union[float, mp.mpf]:
        """Optimized power function with quick overflow checks."""
        try:
            if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                abs_x, abs_y = abs(x), abs(y)
                if abs_x > 10 and abs_y > 1000000:
                    return float('inf')
                if abs_x > 1000 and abs_y > 10000:
                    return float('inf')
            return mp.power(x, y)
        except:
            return float('inf')

    @lru_cache(maxsize=1000)
    def _cached_tetration(self, a: float, n: int) -> Union[float, str]:
        """Cached tetration computation with optimizations."""
        try:
            # Quick checks for common values
            if n == 0: return 1
            if n == 1: return a
            if n < 0: return "Error: Negative height"
            if a == 0: return 0
            if a == 1: return 1
            
            # Convert to mpmath for arbitrary precision
            a = mp.mpf(str(a))
            result = a
            
            for _ in range(n - 1):
                # Use mpmath's power function for arbitrary precision
                result = mp.power(a, result)
                
                # Check if result is too large for representation
                if mp.isnan(result) or mp.isinf(result):
                    return float('inf')
                
                # Check if we can still handle the next iteration
                try:
                    log_estimate = mp.log(result)
                    if log_estimate > mp.mpf('1e100000'):  # Much higher limit
                        return float('inf')
                except:
                    return float('inf')
            
            # Try to convert back to float if possible
            try:
                if result > mp.mpf('1e1000000'):
                    return float('inf')
                return float(result)
            except:
                return float('inf')
                
        except Exception as e:
            return f"Error: {str(e)}"

    def format_large_number(self, num: Union[float, mp.mpf]) -> str:
        """Optimized number formatting with special handling for tetration results."""
        try:
            if isinstance(num, float) and math.isinf(num):
                return "Infinity" if num > 0 else "-Infinity"
            
            if isinstance(num, (int, float)):
                abs_num = abs(num)
                if abs_num < 1e-10 and abs_num != 0:
                    return f"{num:.2e}"
                if abs_num > 1e10:
                    exp = int(math.log10(abs_num))
                    if exp > 1e6:
                        # For extremely large numbers from tetration
                        log_exp = math.log10(exp)
                        if log_exp > 100:
                            return f"10↑↑{int(log_exp)}"
                        mantissa = num / (10 ** exp)
                        return f"{mantissa:.2f}e{exp}"
                    mantissa = num / (10 ** exp)
                    return f"{mantissa:.6f}e{exp}"
                return f"{num:.10g}".rstrip('0').rstrip('.')
            
            # mpmath number handling
            return str(mp.nstr(num, 10)).rstrip('0').rstrip('.')
                
        except:
            return str(num)

    def _normalize_expression(self, expr: str) -> str:
        """Normalize the expression by converting operators and handling special cases."""
        # Handle tetration first (must be done before ^ replacement)
        if "↑↑" in expr:
            parts = expr.split("↑↑")
            if len(parts) != 2:
                raise ValueError("Invalid tetration format")
            base = float(self.evaluate_expression(parts[0]))
            height = int(float(self.evaluate_expression(parts[1])))
            return str(self._compute_tetration(base, height))
        
        # Replace ^ with ** for exponentiation
        expr = expr.replace('^', '**')
        return expr

    def _compute_tetration(self, base: float, height: int) -> float:
        """Compute tetration (a↑↑n) with dynamic precision based on size."""
        if height < 0:
            raise ValueError("Tetration height must be non-negative")
        if height == 0:
            return 1
        if height == 1:
            return base
        
        # Use logarithms for large numbers to prevent overflow
        try:
            result = base
            for i in range(height - 1):
                if result > 1e10:
                    # Switch to logarithmic calculation for very large numbers
                    log_result = math.log(result)
                    result = math.exp(log_result * base)
                    
                    # If the result is too large, return an approximation
                    if result > 1e100:
                        # Calculate number of digits in the result
                        digits = int(log_result * base / math.log(10))
                        if digits > 1e6:
                            return float('inf')
                        # Return approximation in scientific notation
                        mantissa = base * math.log10(math.e) % 1
                        return float(f"{math.pow(10, mantissa):.2f}e{digits}")
                else:
                    result = pow(base, result)
            return result
        except OverflowError:
            # If overflow occurs, try logarithmic calculation
            try:
                log_result = math.log(result)
                digits = int(log_result * base / math.log(10))
                if digits > 1e6:
                    return float('inf')
                mantissa = base * math.log10(math.e) % 1
                return float(f"{math.pow(10, mantissa):.2f}e{digits}")
            except:
                return float('inf')

    def evaluate_expression(self, expression: str) -> float:
        """Evaluates a mathematical expression."""
        try:
            # Normalize the expression first
            expr = self._normalize_expression(expression)
            # Handle special cases
            if expr == "inf" or expr == "Infinity":
                return float('inf')
            if expr == "-inf" or expr == "-Infinity":
                return float('-inf')
            
            return float(eval(expr, {"__builtins__": None},
                            {"sin": np.sin, "cos": np.cos, "tan": np.tan,
                             "exp": np.exp, "log": np.log10, "ln": np.log,
                             "pi": np.pi, "e": np.e, "sqrt": np.sqrt,
                             "W": lambda x: float(lambertw(x, 0).real),
                             "loga": lambda a, b: np.log(b) / np.log(a)}))
        except Exception as e:
            return f"Error: {str(e)}"

    def calculate_derivative(self, expression: str, variable: str = 'x') -> str:
        """Calculates the derivative of an expression."""
        try:
            # Normalize the expression first
            expr = self._normalize_expression(expression)
            parsed_expr = sp.parse_expr(expr)
            var = sp.Symbol(variable)
            derivative = sp.diff(parsed_expr, var)
            # Convert back to ^ notation for display
            return str(derivative).replace('**', '^')
        except Exception as e:
            return f"Error: {str(e)}"

    def calculate_integral(self, expression: str, variable: str = 'x') -> str:
        """Calculates the indefinite integral of an expression."""
        try:
            # Normalize the expression first
            expr = self._normalize_expression(expression)
            parsed_expr = sp.parse_expr(expr)
            var = sp.Symbol(variable)
            integral = sp.integrate(parsed_expr, var)
            # Convert back to ^ notation for display
            return str(integral).replace('**', '^')
        except Exception as e:
            return f"Error: {str(e)}"

    def definite_integral(self, expression: str, lower: float, upper: float, variable: str = 'x') -> float:
        """Calculates the definite integral of an expression between limits."""
        try:
            # Normalize the expression first
            expr = self._normalize_expression(expression)
            parsed_expr = sp.parse_expr(expr)
            var = sp.Symbol(variable)
            integral = sp.integrate(parsed_expr, (var, lower, upper))
            return float(integral.evalf())
        except Exception as e:
            return f"Error: {str(e)}"

    def lambert_w(self, x: float) -> float:
        """Computes the Lambert W function (principal branch)."""
        try:
            return float(lambertw(x, 0).real)
        except Exception as e:
            return f"Error: {str(e)}"

    def solve_equation(self, equation: str) -> List[complex]:
        """Solves an equation and returns all roots."""
        try:
            # Normalize the equation first
            equation = self._normalize_expression(equation)
            if '=' in equation:
                left, right = equation.split('=')
                equation = f"({left})-({right})"
            eq = sp.parse_expr(equation)
            return [complex(root) for root in sp.solve(eq)]
        except Exception as e:
            return f"Error: {str(e)}"

    def evaluate_function(self, expression: str, x_val: float) -> float:
        """Evaluates a function at a specific x value."""
        try:
            # Normalize the expression first
            expr = self._normalize_expression(expression)
            parsed_expr = sp.parse_expr(expr)
            return float(parsed_expr.subs(self.x, x_val))
        except Exception as e:
            return f"Error: {str(e)}"

    def generate_plot_points(self, expression: str, x_min: float = -10, x_max: float = 10, 
                            points: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """Generates points for plotting a function."""
        try:
            # Normalize the expression first
            expr = self._normalize_expression(expression)
            x_vals = np.linspace(x_min, x_max, points)
            parsed_expr = sp.parse_expr(expr)
            y_vals = [float(parsed_expr.subs(self.x, x_val)) for x_val in x_vals]
            return x_vals, np.array(y_vals)
        except Exception as e:
            return None, f"Error: {str(e)}"

    def matrix_operations(self, operation: str, matrix_a: np.ndarray, 
                        matrix_b: np.ndarray = None) -> np.ndarray:
        """Performs matrix operations."""
        try:
            if operation == "determinant":
                return np.linalg.det(matrix_a)
            elif operation == "inverse":
                return np.linalg.inv(matrix_a)
            elif operation == "multiply" and matrix_b is not None:
                return np.matmul(matrix_a, matrix_b)
            elif operation == "transpose":
                return np.transpose(matrix_a)
            else:
                raise ValueError("Invalid matrix operation")
        except Exception as e:
            return f"Error: {str(e)}"

    def statistical_operations(self, data: List[float], operation: str) -> float:
        """Performs statistical operations on a dataset."""
        try:
            if operation == "mean":
                return np.mean(data)
            elif operation == "median":
                return np.median(data)
            elif operation == "std":
                return np.std(data)
            elif operation == "variance":
                return np.var(data)
            else:
                raise ValueError("Invalid statistical operation")
        except Exception as e:
            return f"Error: {str(e)}" 