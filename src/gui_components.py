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
        self.precision_frame.pack(fill="x", padx=10, pady=(5,0))
        
        # Mode selection with improved styling
        mode_frame = ctk.CTkFrame(self)
        mode_frame.pack(fill='x', padx=10, pady=5)
        
        mode_label = ctk.CTkLabel(mode_frame, text="Plotting Mode:", font=("Helvetica", 12, "bold"))
        mode_label.pack(side='left', padx=(5,10))
        
        self.mode_var = tk.StringVar(value="single")
        single_mode = ctk.CTkRadioButton(mode_frame, text="Single Function", 
                                       variable=self.mode_var, value="single",
                                       command=self.switch_mode,
                                       font=("Helvetica", 12))
        single_mode.pack(side='left', padx=5)
        
        multi_mode = ctk.CTkRadioButton(mode_frame, text="Multiple Functions", 
                                      variable=self.mode_var, value="multi",
                                      command=self.switch_mode,
                                      font=("Helvetica", 12))
        multi_mode.pack(side='left', padx=5)
        
        # Show derivative checkbox with improved styling
        self.show_derivative = tk.BooleanVar(value=True)
        derivative_check = ctk.CTkCheckBox(mode_frame, text="Show Derivatives",
                                         variable=self.show_derivative,
                                         command=self.plot_function,
                                         font=("Helvetica", 12))
        derivative_check.pack(side='right', padx=10)
        
        # Function input frames with improved styling
        self.single_input_frame = ctk.CTkFrame(self)
        self.multi_input_frame = ctk.CTkFrame(self)
        
        # Single function mode
        ctk.CTkLabel(self.single_input_frame, text="f(x) = ", 
                    font=("Helvetica", 14, "bold")).pack(side='left', padx=5)
        self.function_entry = ctk.CTkEntry(self.single_input_frame, width=400,
                                         font=("Helvetica", 12))
        self.function_entry.pack(side='left', padx=5)
        
        # Multiple functions mode with improved styling
        self.function_entries = []
        colors = ['blue', 'red', 'green']
        for i in range(3):
            func_frame = ctk.CTkFrame(self.multi_input_frame)
            func_frame.pack(fill='x', padx=5, pady=2)
            
            label = ctk.CTkLabel(func_frame, text=f"f{i+1}(x) = ",
                               font=("Helvetica", 14, "bold"),
                               text_color=colors[i])
            label.pack(side='left', padx=5)
            
            entry = ctk.CTkEntry(func_frame, width=400,
                               font=("Helvetica", 12))
            entry.pack(side='left', padx=5)
            self.function_entries.append(entry)
        
        # Show single function mode by default
        self.single_input_frame.pack(fill='x', padx=10, pady=5)
        
        # Range inputs and zoom control with improved styling
        range_frame = ctk.CTkFrame(self)
        range_frame.pack(fill='x', padx=10, pady=5)
        
        # Left side: Range inputs
        range_inputs = ctk.CTkFrame(range_frame)
        range_inputs.pack(side='left', padx=5)
        
        range_label = ctk.CTkLabel(range_inputs, text="Plot Range:",
                                 font=("Helvetica", 12, "bold"))
        range_label.pack(side='left', padx=(5,10))
        
        ctk.CTkLabel(range_inputs, text="x min:", 
                    font=("Helvetica", 12)).pack(side='left', padx=5)
        self.x_min = ctk.CTkEntry(range_inputs, width=70,
                                font=("Helvetica", 12))
        self.x_min.pack(side='left', padx=5)
        self.x_min.insert(0, "-10")
        
        ctk.CTkLabel(range_inputs, text="x max:",
                    font=("Helvetica", 12)).pack(side='left', padx=5)
        self.x_max = ctk.CTkEntry(range_inputs, width=70,
                                font=("Helvetica", 12))
        self.x_max.pack(side='left', padx=5)
        self.x_max.insert(0, "10")
        
        # Center: Plot and Clear buttons
        button_frame = ctk.CTkFrame(range_frame)
        button_frame.pack(side='left', padx=20)
        
        self.plot_button = ctk.CTkButton(button_frame, text="Plot",
                                       command=self.plot_function,
                                       width=80, height=32,
                                       font=("Helvetica", 12, "bold"))
        self.plot_button.pack(side='left', padx=5)
        
        self.clear_button = ctk.CTkButton(button_frame, text="Clear",
                                        command=self.clear_plot,
                                        width=80, height=32,
                                        font=("Helvetica", 12))
        self.clear_button.pack(side='left', padx=5)
        
        # Right side: Zoom control
        zoom_frame = ctk.CTkFrame(range_frame)
        zoom_frame.pack(side='right', padx=10)
        
        zoom_label = ctk.CTkLabel(zoom_frame, text="Zoom:",
                                font=("Helvetica", 12, "bold"))
        zoom_label.pack(side='left', padx=5)
        
        self.zoom_slider = ctk.CTkSlider(zoom_frame, from_=0.1, to=10.0,
                                       number_of_steps=99,
                                       command=self.update_zoom,
                                       width=150)
        self.zoom_slider.pack(side='left', padx=5)
        self.zoom_slider.set(1.0)
        
        self.zoom_label = ctk.CTkLabel(zoom_frame, text="1.0x",
                                     font=("Helvetica", 12))
        self.zoom_label.pack(side='left', padx=5)
        
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

    def switch_mode(self):
        if self.mode_var.get() == "single":
            self.multi_input_frame.pack_forget()
            self.single_input_frame.pack(fill='x', padx=5, pady=5)
        else:
            self.single_input_frame.pack_forget()
            self.multi_input_frame.pack(fill='x', padx=5, pady=5)
        self.clear_plot()
        
    def plot_function(self):
        try:
            x_min = float(self.x_min.get())
            x_max = float(self.x_max.get())
            
            self.plot.clear()
            
            if self.mode_var.get() == "single":
                expr = self.function_entry.get()
                self._plot_single_function(expr, x_min, x_max)
            else:
                self._plot_multiple_functions(x_min, x_max)
            
            self.plot.grid(True)
            self.plot.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            self.plot.axvline(x=0, color='k', linestyle='-', alpha=0.3)
            self.plot.legend()
            
            # Store current limits for zoom control
            self.current_xlim = self.plot.get_xlim()
            self.current_ylim = self.plot.get_ylim()
            
            # Apply current zoom level
            self.update_zoom(self.zoom_slider.get())
            
            self.canvas.draw()
        except Exception as e:
            print(f"Error plotting function: {str(e)}")
            
    def _plot_single_function(self, expr, x_min, x_max):
        # Expression is normalized in calculator_core
        x_vals, y_vals = self.calculator_core.generate_plot_points(
            expr, x_min, x_max)
        
        if isinstance(y_vals, str) and y_vals.startswith("Error"):
            raise ValueError(y_vals)
        
        self.plot.plot(x_vals, y_vals, label='f(x)')
        
        # Plot derivative if checkbox is checked
        if self.show_derivative.get():
            derivative = self.calculator_core.calculate_derivative(expr)
            x_vals_d, y_vals_d = self.calculator_core.generate_plot_points(
                derivative, x_min, x_max)
            if not isinstance(y_vals_d, str):
                self.plot.plot(x_vals_d, y_vals_d, '--', label="f'(x)")
        
        # Display expression with ^ for better readability
        display_expr = expr.replace('**', '^')
        self.plot.set_title(f"f(x) = {display_expr}")
        
        # Update analysis
        analysis = self.analyze_function(expr, x_min, x_max)
        self.analysis_text.delete('1.0', tk.END)
        self.analysis_text.insert('1.0', analysis)
            
    def _plot_multiple_functions(self, x_min, x_max):
        self.analysis_text.delete('1.0', tk.END)
        colors = ['b', 'r', 'g']  # Different colors for each function
        
        for i, entry in enumerate(self.function_entries):
            expr = entry.get().strip()
            if not expr:  # Skip empty entries
                continue
            
            try:
                x_vals, y_vals = self.calculator_core.generate_plot_points(
                    expr, x_min, x_max)
                
                if isinstance(y_vals, str) and y_vals.startswith("Error"):
                    continue
                
                self.plot.plot(x_vals, y_vals, color=colors[i], label=f'f{i+1}(x)')
                
                # Plot derivative if checkbox is checked
                if self.show_derivative.get():
                    derivative = self.calculator_core.calculate_derivative(expr)
                    x_vals_d, y_vals_d = self.calculator_core.generate_plot_points(
                        derivative, x_min, x_max)
                    if not isinstance(y_vals_d, str):
                        self.plot.plot(x_vals_d, y_vals_d, '--', color=colors[i], 
                                     label=f"f{i+1}'(x)")
                
                # Add analysis for this function
                analysis = self.analyze_function(expr, x_min, x_max)
                # Display expression with ^ for better readability
                display_expr = expr.replace('**', '^')
                self.analysis_text.insert(tk.END, f"\nFunction {i+1}: f(x) = {display_expr}\n{analysis}\n")
                
            except Exception as e:
                print(f"Error plotting function {i+1}: {str(e)}")
                continue
        
        self.plot.set_title("Multiple Functions")

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

    def update_zoom(self, value):
        """Update the zoom level and redraw the plot."""
        self.zoom_label.configure(text=f"{value:.1f}x")
        if hasattr(self, 'current_xlim') and hasattr(self, 'current_ylim'):
            # Calculate new limits based on zoom
            center_x = sum(self.current_xlim) / 2
            center_y = sum(self.current_ylim) / 2
            width_x = (self.current_xlim[1] - self.current_xlim[0]) / (2 * value)
            width_y = (self.current_ylim[1] - self.current_ylim[0]) / (2 * value)
            
            self.plot.set_xlim(center_x - width_x, center_x + width_x)
            self.plot.set_ylim(center_y - width_y, center_y + width_y)
            self.canvas.draw()

    def clear_plot(self):
        """Clear the current plot."""
        self.plot.clear()
        self.plot.grid(True)
        self.plot.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        self.plot.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        self.canvas.draw()
        self.analysis_text.delete('1.0', tk.END)
        # Reset zoom slider
        self.zoom_slider.set(1.0)
        self.zoom_label.configure(text="1.0x")

class CalculusFrame(CalculatorFrame):
    def __init__(self, parent, calculator_core):
        super().__init__(parent, calculator_core)
        
        # Add precision control at the top
        self.precision_frame = PrecisionFrame(self, calculator_core)
        self.precision_frame.pack(fill="x", padx=10, pady=(5,0))
        
        # Function input with improved styling
        input_frame = ctk.CTkFrame(self)
        input_frame.pack(fill='x', padx=10, pady=5)
        
        input_label = ctk.CTkLabel(input_frame, text="Enter Function:",
                                 font=("Helvetica", 12, "bold"))
        input_label.pack(side='left', padx=(5,10))
        
        ctk.CTkLabel(input_frame, text="f(x) = ",
                    font=("Helvetica", 14, "bold")).pack(side='left', padx=5)
        self.function_entry = ctk.CTkEntry(input_frame, width=400,
                                         font=("Helvetica", 12))
        self.function_entry.pack(side='left', padx=5)
        
        # Buttons frame with improved styling
        buttons_frame = ctk.CTkFrame(self)
        buttons_frame.pack(fill='x', padx=10, pady=5)
        
        # Left side: Operation buttons
        operations_frame = ctk.CTkFrame(buttons_frame)
        operations_frame.pack(side='left', padx=5)
        
        self.derivative_btn = ctk.CTkButton(operations_frame, 
                                          text="Calculate Derivative",
                                          command=self.calculate_derivative,
                                          width=180, height=32,
                                          font=("Helvetica", 12, "bold"))
        self.derivative_btn.pack(side='left', padx=5)
        
        self.integral_btn = ctk.CTkButton(operations_frame,
                                        text="Calculate Integral",
                                        command=self.calculate_integral,
                                        width=180, height=32,
                                        font=("Helvetica", 12, "bold"))
        self.integral_btn.pack(side='left', padx=5)
        
        # Right side: Definite integral inputs
        definite_frame = ctk.CTkFrame(buttons_frame)
        definite_frame.pack(side='right', padx=5)
        
        def_int_label = ctk.CTkLabel(definite_frame, text="Definite Integral:",
                                   font=("Helvetica", 12, "bold"))
        def_int_label.pack(side='left', padx=(5,10))
        
        ctk.CTkLabel(definite_frame, text="from:",
                    font=("Helvetica", 12)).pack(side='left', padx=5)
        self.lower_bound = ctk.CTkEntry(definite_frame, width=70,
                                      font=("Helvetica", 12))
        self.lower_bound.pack(side='left', padx=5)
        
        ctk.CTkLabel(definite_frame, text="to:",
                    font=("Helvetica", 12)).pack(side='left', padx=5)
        self.upper_bound = ctk.CTkEntry(definite_frame, width=70,
                                      font=("Helvetica", 12))
        self.upper_bound.pack(side='left', padx=5)
        
        self.def_integral_btn = ctk.CTkButton(definite_frame,
                                            text="Calculate",
                                            command=self.calculate_definite_integral,
                                            width=100, height=32,
                                            font=("Helvetica", 12))
        self.def_integral_btn.pack(side='left', padx=5)
        
        # Result display with improved styling
        result_label = ctk.CTkLabel(self, text="Result:",
                                  font=("Helvetica", 12, "bold"))
        result_label.pack(anchor='w', padx=15, pady=(5,0))
        
        self.result_text = ctk.CTkTextbox(self, height=300,
                                        font=("Helvetica", 12))
        self.result_text.pack(fill='both', expand=True, padx=10, pady=5)

    def calculate_derivative(self):
        try:
            expr = self.function_entry.get()
            result = self.calculator_core.calculate_derivative(expr)
            self.result_text.delete('1.0', tk.END)
            # Display expression with ^ for better readability
            display_expr = expr.replace('**', '^')
            self.result_text.insert('1.0', f"Derivative of f(x) = {display_expr}:\n\nf'(x) = {result}")
        except Exception as e:
            self.result_text.delete('1.0', tk.END)
            self.result_text.insert('1.0', f"Error: {str(e)}")

    def calculate_integral(self):
        try:
            expr = self.function_entry.get()
            result = self.calculator_core.calculate_integral(expr)
            self.result_text.delete('1.0', tk.END)
            # Display expression with ^ for better readability
            display_expr = expr.replace('**', '^')
            self.result_text.insert('1.0', f"Indefinite integral of f(x) = {display_expr}:\n\n∫f(x)dx = {result} + C")
        except Exception as e:
            self.result_text.delete('1.0', tk.END)
            self.result_text.insert('1.0', f"Error: {str(e)}")
            
    def calculate_definite_integral(self):
        try:
            expr = self.function_entry.get()
            lower = float(self.lower_bound.get())
            upper = float(self.upper_bound.get())
            result = self.calculator_core.definite_integral(expr, lower, upper)
            self.result_text.delete('1.0', tk.END)
            # Display expression with ^ for better readability
            display_expr = expr.replace('**', '^')
            self.result_text.insert('1.0', 
                f"Definite integral of f(x) = {display_expr}\n"
                f"from x = {lower} to x = {upper}:\n\n"
                f"∫({display_expr})dx = {result}")
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
            solutions = self.calculator_core.solve_equation(equation)
            
            self.result_text.delete('1.0', tk.END)
            # Display equation with ^ for better readability
            display_eq = equation.replace('**', '^')
            self.result_text.insert('1.0', f"Equation: {display_eq}\n\nSolutions:\n")
            
            if isinstance(solutions, str):
                self.result_text.insert(tk.END, solutions)
            else:
                for i, sol in enumerate(solutions, 1):
                    if abs(sol.imag) < 1e-10:
                        self.result_text.insert(tk.END, f"x{i} = {sol.real:.6g}\n")
                    else:
                        self.result_text.insert(tk.END, f"x{i} = {sol:.6g}\n")
        except Exception as e:
            self.result_text.delete('1.0', tk.END)
            self.result_text.insert('1.0', f"Error: {str(e)}") 