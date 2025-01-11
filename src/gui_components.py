import customtkinter as ctk
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import numpy as np
import sympy as sp
from scipy.optimize import fsolve
from sympy.solvers import solve
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
from sympy import Eq, solve_linear_system, Matrix

class CalculatorFrame(ctk.CTkFrame):
    def __init__(self, parent, calculator_core):
        super().__init__(parent)
        self.calculator_core = calculator_core
        self.pack(fill='both', expand=True)

class InfoFrame(CalculatorFrame):
    def __init__(self, parent, calculator_core):
        super().__init__(parent, calculator_core)
        
        # Create scrollable text widget
        self.info_text = ctk.CTkTextbox(self, height=500, width=700)
        self.info_text.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Add information text
        info = """Advanced Scientific Calculator - Function Guide

Precision Control:
• Adjustable calculation precision (10 to 1,000,000 digits)
• Quick presets:
  - Low (15 digits): Fast, for basic calculations
  - Medium (50 digits): Default, good for most uses
  - High (1000 digits): For high-precision calculations
• Custom precision: Enter any value between 10 and 1M

Basic Operations:
• +, -, *, / : Basic arithmetic
• ^ or ** : Power/Exponent
• ↑↑ : Tetration (iterated exponentiation)
• √ : Square root
• ( ) : Parentheses for grouping
• ± : Change sign
• Ans : Recall last answer

Memory Functions:
• M+ : Add to memory
• M- : Subtract from memory
• MR : Recall memory
• MC : Clear memory

Scientific Functions:
• sin(x), cos(x), tan(x) : Trigonometric functions
• asin(x), acos(x), atan(x) : Inverse trigonometric functions
• sinh(x), cosh(x), tanh(x) : Hyperbolic functions
• log(x) : Base-10 logarithm
• ln(x) : Natural logarithm (base e)
• loga(a,b) : Logarithm with base a of b
• abs(x) : Absolute value
• e : Euler's number (2.71828...)
• π (pi) : Pi (3.14159...)
• W(x) : Lambert W function (principal branch)

Advanced Features:
• Tetration (↑↑): Super-exponentiation
  Example: 2↑↑3 = 2^(2^2) = 2^4 = 16
• Scientific Notation: Automatic for large numbers
• Large Number Support: Up to 10^1000000
• Arbitrary Precision: Adjustable calculation accuracy

Graphing Calculator:
• Plot any mathematical function of x
• Real-time coordinate display
• Function analysis:
  - Derivatives
  - Critical points
  - Inflection points
  - Asymptotes
  - Integrals
• Zoom and pan capabilities
• Examples:
  - e^((x^2+2)/sqrt(abs(x)))
  - x^5+x^3+9
  - 1/(x^2)
  - W(x) : Lambert W function
  - log(x) : Base-10 logarithm
  - loga(2,x) : Log base 2 of x

Calculus Functions:
• Derivative: Calculates the derivative of a function
• Integral: Calculates the indefinite integral
• Examples:
  - x^2 + 2*x + 1
  - sin(x)*e^x
  - ln(x)/x
  - W(x^2)
  - log(x) + loga(2,x)

Tips:
• Use parentheses to ensure correct order of operations
• For complex expressions, break them into smaller parts
• Adjust precision for speed vs. accuracy trade-off
• Scientific notation: Use 'e' (e.g., 1e-3 for 0.001)
• For logarithms:
  - log(x) is base 10
  - ln(x) is natural log (base e)
  - loga(a,b) is log base a of b"""
        
        self.info_text.insert('1.0', info)
        self.info_text.configure(state='disabled')  # Make text read-only

class PrecisionFrame(ctk.CTkFrame):
    def __init__(self, parent, calculator_core):
        super().__init__(parent)
        self.calculator_core = calculator_core
        
        # Create precision control
        self.precision_label = ctk.CTkLabel(self, text="Precision (digits):")
        self.precision_label.pack(side="left", padx=5)
        
        self.precision_var = tk.StringVar(value="50")
        self.precision_entry = ctk.CTkEntry(self, width=70, textvariable=self.precision_var)
        self.precision_entry.pack(side="left", padx=5)
        
        self.apply_button = ctk.CTkButton(self, text="Apply", width=60, 
                                        command=self._apply_precision)
        self.apply_button.pack(side="left", padx=5)
        
        # Quick preset buttons
        self.preset_frame = ctk.CTkFrame(self)
        self.preset_frame.pack(side="left", padx=10)
        
        presets = [("Low", "15"), ("Medium", "50"), ("High", "1000")]
        for text, value in presets:
            btn = ctk.CTkButton(self.preset_frame, text=text, width=60,
                              command=lambda v=value: self._set_preset(v))
            btn.pack(side="left", padx=2)
            
    def _set_preset(self, value):
        self.precision_var.set(value)
        self._apply_precision()
        
    def _apply_precision(self):
        try:
            precision = int(self.precision_var.get())
            precision = max(10, min(1000000, precision))
            self.calculator_core.set_precision(precision)
            self.precision_var.set(str(precision))  # Update with clamped value
        except ValueError:
            self.precision_var.set("50")  # Reset to default if invalid

class MainCalculatorFrame(CalculatorFrame):
    def __init__(self, parent, calculator_core):
        super().__init__(parent, calculator_core)
        
        # Display frame
        self.display_frame = ctk.CTkFrame(self)
        self.display_frame.pack(fill="x", padx=10, pady=5)
        
        self.display_var = tk.StringVar(value="0")
        self.display = ctk.CTkEntry(self.display_frame, textvariable=self.display_var,
                                  font=("Arial", 20), justify="right")
        self.display.pack(fill="x", padx=5, pady=5)
        
        # Precision control below display
        self.precision_frame = PrecisionFrame(self, calculator_core)
        self.precision_frame.pack(fill="x", padx=10, pady=5)
        
        # Main container for all buttons
        main_button_frame = ctk.CTkFrame(self)
        main_button_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Scientific functions (top rows)
        scientific_functions = [
            ['sin', 'cos', 'tan', 'π', 'e', 'W', 'log'],
            ['asin', 'acos', 'atan', '√', '^', '↑↑', 'ln'],
            ['sinh', 'cosh', 'tanh', 'abs', 'loga', '(', ')']
        ]
        
        # Basic calculator buttons (bottom rows)
        basic_buttons = [
            ['MC', 'MR', 'M-', 'M+', 'C', 'DEL', '±'],
            ['7', '8', '9', '/', 'EXP', '!', 'Ans'],
            ['4', '5', '6', '*', '×10ⁿ', '(', ')'],
            ['1', '2', '3', '-', 'π', 'e', 'W'],
            ['0', '.', '=', '+', 'log', 'ln', '√']
        ]

        # Add scientific function buttons
        for i, row in enumerate(scientific_functions):
            for j, text in enumerate(row):
                btn = ctk.CTkButton(main_button_frame, text=text, 
                                  width=60, height=30,
                                  command=lambda t=text: self.scientific_click(t))
                btn.grid(row=i, column=j, padx=2, pady=2, sticky='nsew')

        # Add basic calculator buttons
        for i, row in enumerate(basic_buttons):
            for j, text in enumerate(row):
                btn = ctk.CTkButton(main_button_frame, text=text,
                                  width=60, height=30,
                                  command=lambda t=text: self.button_click(t))
                btn.grid(row=i+3, column=j, padx=2, pady=2, sticky='nsew')

        # Configure grid
        for i in range(7):
            main_button_frame.grid_columnconfigure(i, weight=1)
        for i in range(8):
            main_button_frame.grid_rowconfigure(i, weight=1)

        # Initialize memory and answer storage
        self.memory = 0
        self.last_answer = 0

        # Bind Enter key to calculate
        self.display.bind('<Return>', lambda e: self.button_click('='))

    def button_click(self, text):
        if text == '=':
            try:
                result = self.calculator_core.evaluate_expression(self.display.get())
                self.last_answer = result
                self.display.delete(0, tk.END)
                self.display.insert(0, str(result))
            except:
                self.display.delete(0, tk.END)
                self.display.insert(0, "Error")
        elif text == 'C':
            self.display.delete(0, tk.END)
        elif text == 'DEL':
            self.display.delete(len(self.display.get())-1)
        elif text == 'M+':
            try:
                self.memory += float(self.calculator_core.evaluate_expression(self.display.get()))
            except:
                pass
        elif text == 'M-':
            try:
                self.memory -= float(self.calculator_core.evaluate_expression(self.display.get()))
            except:
                pass
        elif text == 'MR':
            self.display.insert(tk.END, str(self.memory))
        elif text == 'MC':
            self.memory = 0
        elif text == 'Ans':
            self.display.insert(tk.END, str(self.last_answer))
        elif text == '±':
            try:
                current = float(self.calculator_core.evaluate_expression(self.display.get()))
                self.display.delete(0, tk.END)
                self.display.insert(0, str(-current))
            except:
                pass
        elif text == '×10ⁿ':
            self.display.insert(tk.END, "*10**")
        elif text == 'EXP':
            self.display.insert(tk.END, "e**")
        elif text == '!':
            self.display.insert(tk.END, "factorial(")
        else:
            self.display.insert(tk.END, text)

    def scientific_click(self, text):
        if text == 'π':
            self.display.insert(tk.END, 'pi')
        elif text == '√':
            self.display.insert(tk.END, 'sqrt(')
        elif text == '^':
            self.display.insert(tk.END, '**')
        elif text == '↑↑':
            self.display.insert(tk.END, '↑↑')
        elif text == 'W':
            self.display.insert(tk.END, 'W(')
        elif text == 'log':
            self.display.insert(tk.END, 'log(')
        elif text == 'loga':
            self.display.insert(tk.END, 'loga(')
        else:
            self.display.insert(tk.END, text + '(')

class GraphingCalculatorFrame(CalculatorFrame):
    def __init__(self, parent, calculator_core):
        super().__init__(parent, calculator_core)
        
        # Add precision control at the top
        self.precision_frame = PrecisionFrame(self, calculator_core)
        self.precision_frame.pack(fill="x", padx=10, pady=5)
        
        # Function input
        input_frame = ctk.CTkFrame(self)
        input_frame.pack(fill='x', padx=5, pady=5)
        
        ctk.CTkLabel(input_frame, text="f(x) = ").pack(side='left', padx=5)
        self.function_entry = ctk.CTkEntry(input_frame, width=300)
        self.function_entry.pack(side='left', padx=5)
        
        # Common functions dropdown
        self.function_var = tk.StringVar()
        functions = [
            "Select a function...",
            "e^(x^2)",
            "sin(x)/x",
            "x^5 + 2*x^3 - 3*x",
            "sqrt(abs(x))",
            "1/(1 + x^2)",
            "e^((x^2+2)/sqrt(abs(x)))",
            "sin(x)*cos(x)",
            "tan(x)",
            "log(abs(x))",
            "x^2*e^(-x^2)"
        ]
        self.function_dropdown = ctk.CTkOptionMenu(
            input_frame,
            values=functions,
            command=self.function_selected
        )
        self.function_dropdown.pack(side='left', padx=5)
        self.function_dropdown.set("Select a function...")
        
        # Range inputs
        range_frame = ctk.CTkFrame(self)
        range_frame.pack(fill='x', padx=5, pady=5)
        
        ctk.CTkLabel(range_frame, text="x min:").pack(side='left', padx=5)
        self.x_min = ctk.CTkEntry(range_frame, width=60)
        self.x_min.pack(side='left', padx=5)
        self.x_min.insert(0, "-10")
        
        ctk.CTkLabel(range_frame, text="x max:").pack(side='left', padx=5)
        self.x_max = ctk.CTkEntry(range_frame, width=60)
        self.x_max.pack(side='left', padx=5)
        self.x_max.insert(0, "10")
        
        # Plot button
        self.plot_button = ctk.CTkButton(range_frame, text="Plot",
                                       command=self.plot_function,
                                       width=60, height=30)
        self.plot_button.pack(side='left', padx=20)
        
        # Clear button
        self.clear_button = ctk.CTkButton(range_frame, text="Clear",
                                        command=self.clear_plot,
                                        width=60, height=30)
        self.clear_button.pack(side='left', padx=5)
        
        # Create main container for plot and analysis
        self.main_container = ctk.CTkFrame(self)
        self.main_container.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Plot container (left side)
        self.plot_container = ctk.CTkFrame(self.main_container)
        self.plot_container.pack(side='left', fill='both', expand=True)
        
        # Matplotlib figure
        self.fig = Figure(figsize=(6, 4), dpi=100)
        self.plot = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_container)
        
        # Add toolbar for zoom, pan, etc.
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.plot_container)
        self.toolbar.update()
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Add coordinate display on hover
        self.coord_label = ctk.CTkLabel(self.plot_container, text="")
        self.coord_label.pack(pady=2)
        self.canvas.mpl_connect('motion_notify_event', self.update_coordinates)
        
        # Analysis container (right side)
        self.analysis_container = ctk.CTkFrame(self.main_container, width=300)
        self.analysis_container.pack(side='right', fill='y', padx=5)
        
        # Analysis text widget
        self.analysis_text = ctk.CTkTextbox(self.analysis_container, width=300, height=400)
        self.analysis_text.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Initialize the plot
        self.clear_plot()

    def function_selected(self, choice):
        """Handle function selection from dropdown."""
        if choice != "Select a function...":
            self.function_entry.delete(0, tk.END)
            self.function_entry.insert(0, choice)
            self.plot_function()  # Automatically plot when function is selected

    def clear_plot(self):
        """Clear the current plot."""
        self.plot.clear()
        self.plot.grid(True)
        self.plot.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        self.plot.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        self.canvas.draw()
        self.analysis_text.delete('1.0', tk.END)

    def update_coordinates(self, event):
        if event.inaxes:
            self.coord_label.configure(text=f'x: {event.xdata:.4f}, y: {event.ydata:.4f}')
        else:
            self.coord_label.configure(text="")

    def analyze_function(self, expr, x_min, x_max):
        try:
            # Calculate derivative
            derivative = self.calculator_core.calculate_derivative(expr)
            second_derivative = self.calculator_core.calculate_derivative(derivative)
            
            # Find critical points (where derivative = 0)
            x_vals = np.linspace(x_min, x_max, 1000)
            critical_points = []
            
            try:
                derivative_vals = []
                for x in x_vals:
                    try:
                        val = float(self.calculator_core.evaluate_function(derivative, x))
                        derivative_vals.append(val)
                    except:
                        derivative_vals.append(float('nan'))
                
                derivative_vals = np.array(derivative_vals)
                for i in range(len(x_vals)-1):
                    if not (np.isnan(derivative_vals[i]) or np.isnan(derivative_vals[i+1])):
                        if derivative_vals[i] * derivative_vals[i+1] <= 0:
                            x_crit = (x_vals[i] + x_vals[i+1]) / 2
                            try:
                                y_crit = float(self.calculator_core.evaluate_function(expr, x_crit))
                                critical_points.append((x_crit, y_crit))
                            except:
                                pass
            except Exception as e:
                print(f"Error finding critical points: {str(e)}")
            
            # Find inflection points (where second derivative = 0)
            inflection_points = []
            try:
                second_derivative_vals = []
                for x in x_vals:
                    try:
                        val = float(self.calculator_core.evaluate_function(second_derivative, x))
                        second_derivative_vals.append(val)
                    except:
                        second_derivative_vals.append(float('nan'))
                
                second_derivative_vals = np.array(second_derivative_vals)
                for i in range(len(x_vals)-1):
                    if not (np.isnan(second_derivative_vals[i]) or np.isnan(second_derivative_vals[i+1])):
                        if second_derivative_vals[i] * second_derivative_vals[i+1] <= 0:
                            x_infl = (x_vals[i] + x_vals[i+1]) / 2
                            try:
                                y_infl = float(self.calculator_core.evaluate_function(expr, x_infl))
                                inflection_points.append((x_infl, y_infl))
                            except:
                                pass
            except Exception as e:
                print(f"Error finding inflection points: {str(e)}")
            
            # Calculate integral
            integral = self.calculator_core.calculate_integral(expr)
            
            # Format analysis text
            analysis = f"Function Analysis:\n\n"
            analysis += f"Derivative: {derivative}\n\n"
            
            if critical_points:
                analysis += f"Critical Points:\n"
                for x, y in critical_points:
                    analysis += f"({x:.4f}, {y:.4f})\n"
            else:
                analysis += "No critical points found in range\n"
            
            if inflection_points:
                analysis += f"\nInflection Points:\n"
                for x, y in inflection_points:
                    analysis += f"({x:.4f}, {y:.4f})\n"
            else:
                analysis += "\nNo inflection points found in range\n"
            
            analysis += f"\nIndefinite Integral:\n{integral}\n"
            
            # Try to find asymptotes
            analysis += "\nAsymptotes:\n"
            asymptotes_found = False
            
            # Vertical asymptotes (look for denominators approaching zero)
            if '/' in expr:
                denom = expr.split('/')[1].strip('()')
                try:
                    zeros = self.calculator_core.solve_equation(denom)
                    for zero in zeros:
                        if zero.imag == 0 and x_min <= zero.real <= x_max:
                            analysis += f"Vertical asymptote at x = {zero.real:.4f}\n"
                            asymptotes_found = True
                except:
                    pass
            
            # Horizontal asymptotes (evaluate limits at infinity)
            try:
                limit_inf = float(self.calculator_core.evaluate_function(expr, 1e6))
                limit_neg_inf = float(self.calculator_core.evaluate_function(expr, -1e6))
                if abs(limit_inf) < 1e10:
                    analysis += f"Horizontal asymptote as x→∞: y = {limit_inf:.4f}\n"
                    asymptotes_found = True
                if abs(limit_neg_inf) < 1e10:
                    analysis += f"Horizontal asymptote as x→-∞: y = {limit_neg_inf:.4f}\n"
                    asymptotes_found = True
            except:
                pass
            
            if not asymptotes_found:
                analysis += "No asymptotes found in range\n"
            
            return analysis
        except Exception as e:
            return f"Error in analysis: {str(e)}"

    def plot_function(self):
        try:
            x_min = float(self.x_min.get())
            x_max = float(self.x_max.get())
            expr = self.function_entry.get()
            
            # Replace ^ with ** if not already done
            expr = expr.replace('^', '**')
            
            x_vals, y_vals = self.calculator_core.generate_plot_points(
                expr, x_min, x_max)
            
            if isinstance(y_vals, str) and y_vals.startswith("Error"):
                raise ValueError(y_vals)
            
            self.plot.clear()
            self.plot.plot(x_vals, y_vals, label='f(x)')
            
            # Plot derivative
            derivative = self.calculator_core.calculate_derivative(expr)
            x_vals_d, y_vals_d = self.calculator_core.generate_plot_points(
                derivative, x_min, x_max)
            if not isinstance(y_vals_d, str):
                self.plot.plot(x_vals_d, y_vals_d, '--', label="f'(x)")
            
            self.plot.grid(True)
            self.plot.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            self.plot.axvline(x=0, color='k', linestyle='-', alpha=0.3)
            self.plot.set_title(f"f(x) = {expr}")
            self.plot.legend()
            
            # Update analysis
            analysis = self.analyze_function(expr, x_min, x_max)
            self.analysis_text.delete('1.0', tk.END)
            self.analysis_text.insert('1.0', analysis)
            
            self.canvas.draw()
        except Exception as e:
            print(f"Error plotting function: {str(e)}")

class CalculusFrame(CalculatorFrame):
    def __init__(self, parent, calculator_core):
        super().__init__(parent, calculator_core)
        
        # Function input
        input_frame = ctk.CTkFrame(self)
        input_frame.pack(fill='x', padx=5, pady=5)
        
        ctk.CTkLabel(input_frame, text="f(x) = ").pack(side='left', padx=5)
        self.function_entry = ctk.CTkEntry(input_frame, width=300)
        self.function_entry.pack(side='left', padx=5)
        
        # Buttons frame
        buttons_frame = ctk.CTkFrame(self)
        buttons_frame.pack(fill='x', padx=5, pady=5)
        
        self.derivative_btn = ctk.CTkButton(buttons_frame, text="Calculate Derivative",
                                          command=self.calculate_derivative,
                                          width=150, height=30)
        self.derivative_btn.pack(side='left', padx=5)
        
        self.integral_btn = ctk.CTkButton(buttons_frame, text="Calculate Integral",
                                        command=self.calculate_integral,
                                        width=150, height=30)
        self.integral_btn.pack(side='left', padx=5)
        
        # Result display
        self.result_text = ctk.CTkTextbox(self, height=200)
        self.result_text.pack(fill='both', expand=True, padx=5, pady=5)

    def calculate_derivative(self):
        try:
            expr = self.function_entry.get()
            result = self.calculator_core.calculate_derivative(expr)
            self.result_text.delete('1.0', tk.END)
            self.result_text.insert('1.0', f"Derivative of {expr}:\n{result}")
        except Exception as e:
            self.result_text.delete('1.0', tk.END)
            self.result_text.insert('1.0', f"Error: {str(e)}")

    def calculate_integral(self):
        try:
            expr = self.function_entry.get()
            result = self.calculator_core.calculate_integral(expr)
            self.result_text.delete('1.0', tk.END)
            self.result_text.insert('1.0', f"Integral of {expr}:\n{result}")
        except Exception as e:
            self.result_text.delete('1.0', tk.END)
            self.result_text.insert('1.0', f"Error: {str(e)}")

class EquationSolverFrame(CalculatorFrame):
    def __init__(self, parent, calculator_core):
        super().__init__(parent, calculator_core)
        
        # Left side: Input and buttons
        left_frame = ctk.CTkFrame(self)
        left_frame.pack(side='left', fill='y', padx=5, pady=5)
        
        # Equation input
        input_frame = ctk.CTkFrame(left_frame)
        input_frame.pack(fill='x', padx=5, pady=5)
        
        ctk.CTkLabel(input_frame, text="Equation:").pack(side='left', padx=5)
        self.equation_entry = ctk.CTkEntry(input_frame, width=300)
        self.equation_entry.pack(side='left', padx=5)
        self.equation_entry.bind('<Return>', lambda e: self.solve_equation())
        
        # Help text
        help_text = """Examples:
• x**2 + 2*x + 1 = 0
• x**3 - 4*x = 9
• sin(x) = 0.5
• exp(x) = 2
• log(x) = 1
• x + y = 5, 2*x - y = 3 (system)"""
        
        help_label = ctk.CTkLabel(left_frame, text=help_text, justify='left')
        help_label.pack(pady=5)
        
        # Calculator buttons
        button_frame = ctk.CTkFrame(left_frame)
        button_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Scientific functions (top rows)
        scientific_functions = [
            ['sin', 'cos', 'tan', 'π', 'e', 'W', 'log'],
            ['asin', 'acos', 'atan', '√', '^', '=', 'ln'],
            ['sinh', 'cosh', 'tanh', 'abs', 'loga', '(', ')']
        ]
        
        # Basic calculator buttons (bottom rows)
        basic_buttons = [
            ['7', '8', '9', '/', 'x', 'y', 'z'],
            ['4', '5', '6', '*', 'DEL', 'C', ','],
            ['1', '2', '3', '-', '≤', '≥', '≠'],
            ['0', '.', '+', '-', '<', '>', '=']
        ]
        
        # Add buttons
        for i, row in enumerate(scientific_functions):
            for j, text in enumerate(row):
                btn = ctk.CTkButton(button_frame, text=text, 
                                  width=60, height=30,
                                  command=lambda t=text: self.button_click(t))
                btn.grid(row=i, column=j, padx=2, pady=2, sticky='nsew')
        
        for i, row in enumerate(basic_buttons):
            for j, text in enumerate(row):
                btn = ctk.CTkButton(button_frame, text=text,
                                  width=60, height=30,
                                  command=lambda t=text: self.button_click(t))
                btn.grid(row=i+3, column=j, padx=2, pady=2, sticky='nsew')
        
        # Configure grid
        for i in range(7):
            button_frame.grid_columnconfigure(i, weight=1)
        for i in range(7):
            button_frame.grid_rowconfigure(i, weight=1)
        
        # Solve button
        solve_btn = ctk.CTkButton(left_frame, text="Solve", 
                                command=self.solve_equation,
                                height=40)
        solve_btn.pack(fill='x', padx=5, pady=10)
        
        # Right side: Solution display
        right_frame = ctk.CTkFrame(self)
        right_frame.pack(side='right', fill='both', expand=True, padx=5, pady=5)
        
        # Solution display
        self.solution_text = ctk.CTkTextbox(right_frame, width=400)
        self.solution_text.pack(fill='both', expand=True, padx=5, pady=5)
    
    def button_click(self, text):
        if text == 'C':
            self.equation_entry.delete(0, tk.END)
        elif text == 'DEL':
            self.equation_entry.delete(len(self.equation_entry.get())-1)
        elif text == 'π':
            self.equation_entry.insert(tk.END, 'pi')
        elif text == '√':
            self.equation_entry.insert(tk.END, 'sqrt(')
        elif text == '^':
            self.equation_entry.insert(tk.END, '**')
        elif text in ['sin', 'cos', 'tan', 'asin', 'acos', 'atan', 
                     'sinh', 'cosh', 'tanh', 'log', 'ln', 'abs', 'W']:
            self.equation_entry.insert(tk.END, text + '(')
        else:
            self.equation_entry.insert(tk.END, text)
    
    def solve_equation(self):
        try:
            equation = self.equation_entry.get()
            self.solution_text.delete('1.0', tk.END)
            
            # Check if it's a system of equations
            if ',' in equation:
                self.solve_system(equation)
                return
            
            # Pre-process the equation
            equation = equation.replace('^', '**')
            
            # Standardize equation format
            if '=' in equation:
                left, right = equation.split('=')
                equation = f"{left}-({right})"
            
            # Parse with implicit multiplication
            transformations = standard_transformations + (implicit_multiplication_application,)
            expr = parse_expr(equation, transformations=transformations)
            
            # Show original equation
            if '=' not in self.equation_entry.get():
                self.solution_text.insert('1.0', f"Equation: {equation} = 0\n\n")
            else:
                self.solution_text.insert('1.0', f"Equation: {self.equation_entry.get()}\n\n")
            
            # Solve
            solution = solve(expr, 'x')
            
            # Display solutions
            self.solution_text.insert(tk.END, "Solutions:\n")
            if not solution:
                self.solution_text.insert(tk.END, "No solution exists\n")
            else:
                for i, sol in enumerate(solution, 1):
                    self.solution_text.insert(tk.END, f"x{i} = {sol}\n")
            
            # Verify solutions
            self.solution_text.insert(tk.END, "\nVerification:\n")
            for sol in solution:
                try:
                    verification = expr.subs('x', sol).evalf()
                    if abs(float(verification)) < 1e-10:
                        self.solution_text.insert(tk.END, f"x = {sol} ✓\n")
                except:
                    pass
            
        except Exception as e:
            self.solution_text.insert('1.0', f"Error: {str(e)}\nPlease check your equation format.\n\nExample formats:\n- x**2 + 6x + 9 = 0\n- x^2 + 6x + 9 = 0\n- sin(x) = 0.5\n- log(x) = 1\n- (3/4)^(x+2) = 2")
    
    def solve_system(self, system):
        try:
            # Split into individual equations
            equations = [eq.strip() for eq in system.split(',')]
            
            self.solution_text.insert('1.0', "System of Equations:\n")
            for eq in equations:
                self.solution_text.insert(tk.END, f"{eq}\n")
            
            # Parse equations
            parsed_eqs = []
            variables = set()
            for eq in equations:
                if '=' in eq:
                    left, right = eq.split('=')
                    eq = f"({left}) - ({right})"
                expr = parse_expr(eq)
                parsed_eqs.append(expr)
                variables.update(expr.free_symbols)
            
            # Solve system
            solution = solve(parsed_eqs, list(variables))
            
            # Display solutions
            self.solution_text.insert(tk.END, "\nSolutions:\n")
            if not solution:
                self.solution_text.insert(tk.END, "No solution exists\n")
            else:
                if isinstance(solution, dict):
                    for var, val in solution.items():
                        self.solution_text.insert(tk.END, f"{var} = {val}\n")
                else:
                    self.solution_text.insert(tk.END, str(solution))
            
        except Exception as e:
            self.solution_text.insert('1.0', f"Error: {str(e)}\n") 