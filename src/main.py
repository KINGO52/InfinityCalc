import customtkinter as ctk
import tkinter as tk
from tkinter import ttk
import numpy as np
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr
from scipy.special import lambertw
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from calculator_core import CalculatorCore
from gui_components import (
    MainCalculatorFrame,
    GraphingCalculatorFrame,
    CalculusFrame,
    InfoFrame,
    EquationSolverFrame
)

class AdvancedCalculator(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Configure window
        self.title("InfinityCalc")
        self.geometry("1200x800")
        
        # Set theme
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # Initialize calculator core
        self.calculator_core = CalculatorCore()

        # Create notebook for different calculator modes
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)

        # Initialize different calculator modes
        self.main_frame = MainCalculatorFrame(self.notebook, self.calculator_core)
        self.scientific_frame = EquationSolverFrame(self.notebook, self.calculator_core)
        self.graphing_frame = GraphingCalculatorFrame(self.notebook, self.calculator_core)
        self.calculus_frame = CalculusFrame(self.notebook, self.calculator_core)
        self.info_frame = InfoFrame(self.notebook, self.calculator_core)

        # Add frames to notebook
        self.notebook.add(self.main_frame, text="Calculator")
        self.notebook.add(self.scientific_frame, text="Equation Solver")
        self.notebook.add(self.graphing_frame, text="Graphing")
        self.notebook.add(self.calculus_frame, text="Calculus")
        self.notebook.add(self.info_frame, text="Help")

if __name__ == "__main__":
    app = AdvancedCalculator()
    app.mainloop() 