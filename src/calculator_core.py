import numpy as np
import sympy as sp
from scipy.special import lambertw
from typing import Union, List, Tuple
import math
import decimal
from decimal import Decimal, getcontext

# Set precision for extremely large numbers
getcontext().prec = 1000000  # Increased precision
MAX_NUMBER = Decimal(10) ** (Decimal(10) ** Decimal(100))  # Maximum number size (10^10^100)

class CalculatorCore:
    def __init__(self):
        self.x = sp.Symbol('x')
        self.memory = 0
        
    def format_large_number(self, num: float) -> str:
        """Formats large numbers in scientific notation or tetration notation."""
        try:
            # Handle infinity
            if math.isinf(num):
                return "Infinity" if num > 0 else "-Infinity"
                
            # Convert to Decimal for higher precision
            num_decimal = Decimal(str(num))
            
            # Check if number exceeds our maximum
            if abs(num_decimal) > MAX_NUMBER:
                return "Overflow Error: Number too large"
            
            # Convert to string first to check number of digits
            num_str = f"{abs(num):.10f}".rstrip('0').rstrip('.')
            num_digits = len(num_str.replace('.', ''))
            
            if num_digits > 10:  # Only format if more than 10 digits
                exp = int(math.log10(abs(num)))
                if exp > 1e6:  # Use tetration notation for extremely large numbers
                    return f"10↑↑{int(math.log10(exp))}"
                else:
                    mantissa = num / (10 ** exp)
                    return f"{mantissa:.6f}e{exp}"
            
            # For smaller numbers, return with appropriate decimal places
            if isinstance(num, int):
                return str(num)
            else:
                # Remove trailing zeros after decimal point
                return f"{num:.10f}".rstrip('0').rstrip('.')
                
        except Exception:
            return str(num)

    def tetration(self, a: float, n: int) -> float:
        """Computes tetration (iterated exponentiation)."""
        try:
            if n == 0:
                return 1
            if n < 0:
                raise ValueError("Negative tetration height not supported")
                
            # Convert to Decimal for higher precision
            a_decimal = Decimal(str(a))
            result = a_decimal
            
            for _ in range(n - 1):
                # Check if next operation would exceed our maximum
                if result > Decimal('1e1000000'):  # Higher intermediate limit
                    log_result = result.ln()
                    if log_result * Decimal(str(a)) > MAX_NUMBER.ln():
                        return float('inf')
                result = a_decimal ** result
                
                # Check if result exceeds maximum
                if result > MAX_NUMBER:
                    return float('inf')
                    
            return float(result)
        except Exception as e:
            return f"Error: {str(e)}"

    def evaluate_expression(self, expression: str) -> Union[float, str]:
        """Evaluates a mathematical expression."""
        try:
            # Replace scientific notation
            expression = expression.replace("e", str(math.e))
            
            # Handle tetration notation (↑↑)
            if "↑↑" in expression:
                parts = expression.split("↑↑")
                if len(parts) == 2:
                    base = float(self.evaluate_expression(parts[0]))
                    height = int(float(self.evaluate_expression(parts[1])))
                    return self.tetration(base, height)
            
            # Create safe math environment with Decimal support
            safe_dict = {
                "sin": np.sin,
                "cos": np.cos,
                "tan": np.tan,
                "asin": np.arcsin,
                "acos": np.arccos,
                "atan": np.arctan,
                "sinh": np.sinh,
                "cosh": np.cosh,
                "tanh": np.tanh,
                "exp": np.exp,
                "log": lambda x: np.log10(x),  # log base 10
                "ln": np.log,  # natural log
                "loga": lambda a, b: np.log(b) / np.log(a),  # log base a of b
                "sqrt": np.sqrt,
                "pi": np.pi,
                "e": np.e,
                "abs": abs,
                "factorial": math.factorial,
                "W": lambda x: float(lambertw(x, 0).real),  # Lambert W function
                "tetration": self.tetration,  # Add tetration function
                "Decimal": Decimal  # Add Decimal support
            }
            
            # Replace ^ with ** for exponentiation
            expression = expression.replace("^", "**")
            
            # Evaluate with safe environment
            result = float(eval(expression, {"__builtins__": None}, safe_dict))
            
            # Format large numbers
            if isinstance(result, (int, float, Decimal)):
                return self.format_large_number(float(result))
            return result
        except Exception as e:
            return f"Error: {str(e)}"

    def calculate_derivative(self, expression: str, variable: str = 'x') -> str:
        """Calculates the derivative of an expression."""
        try:
            # Replace scientific notation and functions
            expression = expression.replace("e", str(math.e))
            expression = expression.replace("log(", "log10(")
            expr = sp.parse_expr(expression)
            var = sp.Symbol(variable)
            derivative = sp.diff(expr, var)
            return str(derivative)
        except Exception as e:
            return f"Error: {str(e)}"

    def calculate_integral(self, expression: str, variable: str = 'x') -> str:
        """Calculates the indefinite integral of an expression."""
        try:
            # Replace scientific notation and functions
            expression = expression.replace("e", str(math.e))
            expression = expression.replace("log(", "log10(")
            expr = sp.parse_expr(expression)
            var = sp.Symbol(variable)
            integral = sp.integrate(expr, var)
            return str(integral)
        except Exception as e:
            return f"Error: {str(e)}"

    def definite_integral(self, expression: str, lower: float, upper: float, variable: str = 'x') -> float:
        """Calculates the definite integral of an expression between limits."""
        try:
            # Replace scientific notation and functions
            expression = expression.replace("e", str(math.e))
            expression = expression.replace("log(", "log10(")
            expr = sp.parse_expr(expression)
            var = sp.Symbol(variable)
            integral = sp.integrate(expr, (var, lower, upper))
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
            # Replace scientific notation and functions
            equation = equation.replace("e", str(math.e))
            equation = equation.replace("log(", "log10(")
            eq = sp.parse_expr(equation)
            return [complex(root) for root in sp.solve(eq)]
        except Exception as e:
            return f"Error: {str(e)}"

    def evaluate_function(self, expression: str, x_val: float) -> float:
        """Evaluates a function at a specific x value."""
        try:
            # Replace scientific notation and functions
            expression = expression.replace("e", str(math.e))
            expression = expression.replace("log(", "log10(")
            expr = sp.parse_expr(expression)
            return float(expr.subs(self.x, x_val))
        except Exception as e:
            return f"Error: {str(e)}"

    def generate_plot_points(self, expression: str, x_min: float = -10, x_max: float = 10, 
                           points: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """Generates points for plotting a function."""
        try:
            # Replace scientific notation and functions
            expression = expression.replace("e", str(math.e))
            expression = expression.replace("log(", "log10(")
            expression = expression.replace("W(", "lambertw(")
            
            x_vals = np.linspace(x_min, x_max, points)
            expr = sp.parse_expr(expression)
            
            # Convert sympy expression to numpy function for vectorized evaluation
            f = sp.lambdify(self.x, expr, modules=['numpy', {'lambertw': lambda x: lambertw(x, 0).real}])
            
            try:
                y_vals = f(x_vals)
                # Handle infinities and NaN values
                y_vals = np.ma.masked_invalid(y_vals)
                return x_vals, y_vals
            except Exception as e:
                # If vectorized evaluation fails, fall back to point-by-point evaluation
                y_vals = []
                for x_val in x_vals:
                    try:
                        y = float(expr.subs(self.x, x_val))
                        y_vals.append(y)
                    except:
                        y_vals.append(np.nan)
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