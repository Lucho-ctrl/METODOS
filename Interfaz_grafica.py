"""Graphical User Interface Module for Fixed-Point Method.

This module provides the main GUI application for solving systems of
non-linear equations using the fixed-point iteration method. The interface
features a modern sidebar navigation with multiple sections for solver,
examples, visualizations, and configuration.

Main Components:
    - MainApplication: Main GUI class with sidebar navigation
    - Color scheme constants for consistent theming
    - Integration with numerical core from main_app module

Dependencies:
    - tkinter: GUI framework
    - numpy: Numerical computations
    - matplotlib: Visualization
"""

import tkinter as tk
from tkinter import messagebox, ttk
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Import numerical core from main_app
from main_app import (
    punto_fijo_sistema,
    build_G,
    EJEMPLOS
)

# Color scheme constants
BG = "#1e1e2e"
SURFACE = "#2a2a3e"
SURFACE2 = "#313149"
ACCENT = "#7c9eff"
ACCENT2 = "#a6e3a1"
TEXT = "#cdd6f4"
TEXT_MUTED = "#a6adc8"
RED_ERR = "#f38ba8"
YELLOW = "#f9e2af"
FONT_BODY = ("Consolas", 11)
FONT_HEAD = ("Segoe UI", 13, "bold")
FONT_MONO = ("Consolas", 10)


class MainApplication(tk.Tk):
    """Main GUI application with sidebar navigation.
    
    This class implements the primary graphical interface for the fixed-point
    method solver. It provides a modern, dark-themed interface with sidebar
    navigation containing multiple sections: home, solver, examples,
    visualizations, settings, and about.
    
    Attributes:
        current_section: StringVar tracking the current active section
        last_solution: NumPy array storing the last computed solution
        last_errors: List storing error history from last computation
        last_historial: List storing iteration history from last computation
        canvas_widget: Reference to matplotlib canvas widget
        content_frame: Frame container for dynamic content
    """

    def __init__(self):
        """Initialize the main application window.
        
        Sets up the window properties, initializes state variables, and
        builds the user interface with sidebar navigation.
        """
        super().__init__()
        self.title("Método del Punto Fijo — Universidad Distrital 2026-1")
        self.configure(bg=BG)
        self.resizable(True, True)
        
        # Full screen automatic
        self.state('zoomed')
        
        # Shared state
        self.current_section = tk.StringVar(value="home")
        self.last_solution = None
        self.last_errors = None
        self.last_historial = None
        self.canvas_widget = None
        
        self._build_ui()
    
    def _build_ui(self):
        """Build the main user interface with sidebar navigation.
        
        Creates the main frame, sidebar navigation, and content area.
        Initializes the home section as the default view.
        """
        main_frame = tk.Frame(self, bg=BG)
        main_frame.pack(fill="both", expand=True)
        
        # Sidebar
        self._build_sidebar(main_frame)
        
        # Content area
        self.content_frame = tk.Frame(main_frame, bg=BG)
        self.content_frame.pack(side="right", fill="both", expand=True)
        
        # Show initial section
        self._show_section("home")
    
    def _build_sidebar(self, parent):
        """Build the sidebar navigation panel.
        
        Args:
            parent: Parent frame for the sidebar
            
        Creates a dark-themed sidebar with navigation buttons for each
        section of the application.
        """
        sidebar = tk.Frame(parent, bg=SURFACE, width=200)
        sidebar.pack(side="left", fill="y", padx=0, pady=0)
        sidebar.pack_propagate(False)
        
        # Header
        hdr = tk.Frame(sidebar, bg="#11111b", height=80)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        tk.Label(hdr, text="⚙", bg="#11111b", fg=ACCENT,
                font=("Segoe UI", 24)).pack(pady=10)
        tk.Label(hdr, text="Punto Fijo", bg="#11111b", fg=ACCENT,
                font=("Segoe UI", 12, "bold")).pack()
        
        # Navigation buttons
        sections = [
            ("home", "🏠  Inicio"),
            ("solver", "🧮  Solucionador"),
            ("examples", "📚  Ejemplos"),
            ("visualizations", "📈  Visualizaciones"),
            ("settings", "⚙️  Configuración"),
            ("about", "ℹ️  Acerca de"),
        ]
        
        for sec_id, label in sections:
            btn = tk.Button(
                sidebar, text=label, bg=SURFACE, fg=TEXT_MUTED,
                font=("Segoe UI", 11), relief="flat", bd=0,
                padx=16, pady=12, anchor="w", cursor="hand2",
                command=lambda s=sec_id: self._show_section(s)
            )
            btn.pack(fill="x", padx=0, pady=2)
            btn.config(activebackground=SURFACE2, activeforeground=ACCENT)
    
    def _clear_content(self):
        """Clear all widgets from the content frame.
        
        Destroys all child widgets in the content area to prepare for
        displaying a new section.
        """
        for widget in self.content_frame.winfo_children():
            widget.destroy()
    
    def _show_section(self, section):
        """Display the specified section in the content area.
        
        Args:
            section: String identifier of the section to display
                    (home, solver, examples, visualizations, settings, about)
        """
        self._clear_content()
        self.current_section.set(section)
        
        if section == "home":
            self._section_home()
        elif section == "solver":
            self._section_solver()
        elif section == "examples":
            self._section_examples()
        elif section == "visualizations":
            self._section_visualizations()
        elif section == "settings":
            self._section_settings()
        elif section == "about":
            self._section_about()
    
    def _section_home(self):
        """Display the home section with welcome information.
        
        Shows an introduction to the fixed-point method and instructions
        for using the application.
        """
        frame = tk.Frame(self.content_frame, bg=BG)
        frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        tk.Label(frame, text="Bienvenido", bg=BG, fg=ACCENT,
                font=("Segoe UI", 20, "bold")).pack(pady=(0, 20))
        
        info_text = """
El Método del Punto Fijo es una técnica numérica para resolver sistemas 
de ecuaciones no lineales de la forma:
    
    x = G₁(x, y)
    y = G₂(x, y)

CARACTERÍSTICAS:
• Solucionador: Ingresa tus propias funciones G₁ y G₂
• Ejemplos: Tres casos de uso predefinidos
• Visualizaciones: Gráficas de convergencia y trayectoria
• Configuración: Ajusta parámetros numéricos
• Análisis: Verifica convergencia mediante norma jacobiana

CÓMO USAR:
1. Ve a "Solucionador" para ingresar tus funciones
2. O explora "Ejemplos" para ver casos predefinidos
3. Usa "Visualizaciones" para ver gráficas de convergencia
4. Ajusta parámetros en "Configuración" si es necesario
        """
        
        txt = tk.Label(frame, text=info_text, bg=BG, fg=TEXT,
                      font=("Segoe UI", 10), justify="left",
                      wraplength=800)
        txt.pack(anchor="nw", pady=20)
    
    def _section_solver(self):
        """Display the interactive solver section.
        
        Provides input fields for functions G₁ and G₂, initial values,
        and numerical parameters. Includes execution button and result display.
        """
        frame = tk.Frame(self.content_frame, bg=BG)
        frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        tk.Label(frame, text="Solucionador Interactivo", bg=BG, fg=ACCENT,
                font=FONT_HEAD).pack(anchor="w", pady=(0, 15))
        
        # Instructions
        info_frame = tk.Frame(frame, bg=SURFACE, bd=0, relief="flat")
        info_frame.pack(fill="x", pady=(0, 15))
        tk.Label(info_frame, text="ℹ  Ingresa G₁(x,y) y G₂(x,y)\n"
                                 "   Operadores: + - * / ** sqrt() exp() log() sin() cos() tan() abs()",
                bg=SURFACE, fg=TEXT_MUTED, font=("Segoe UI", 9), justify="left",
                wraplength=800).pack(padx=12, pady=8, anchor="w")
        
        # Input fields
        campo_frame = tk.Frame(frame, bg=BG)
        campo_frame.pack(fill="x", pady=(0, 15))
        
        def campo(parent, label, row, placeholder, var_name):
            tk.Label(parent, text=label, bg=BG, fg=TEXT, font=FONT_HEAD,
                    width=12, anchor="e").grid(row=row, column=0, padx=(0,10), pady=6, sticky="e")
            sv = tk.StringVar(value=placeholder)
            e = tk.Entry(parent, textvariable=sv, bg=SURFACE2, fg=ACCENT,
                        font=("Consolas", 11), insertbackground=ACCENT,
                        relief="flat", bd=6, width=50)
            e.grid(row=row, column=1, padx=4, pady=6, sticky="w")
            setattr(self, var_name, sv)
            return e
        
        campo(campo_frame, "G₁(x, y) =", 0, "sqrt(1 - y)", "sv_g1")
        campo(campo_frame, "G₂(x, y) =", 1, "sqrt(1 - x)", "sv_g2")
        campo(campo_frame, "x₀ =", 2, "0.5", "sv_x0")
        campo(campo_frame, "y₀ =", 3, "0.5", "sv_y0")
        
        # Parameters
        param_frame = tk.Frame(frame, bg=BG)
        param_frame.pack(fill="x", pady=(0, 20))
        
        def param(parent, label, col, default, var_name):
            tk.Label(parent, text=label, bg=BG, fg=TEXT_MUTED,
                    font=("Segoe UI", 10)).grid(row=0, column=col*2, padx=(0,8), sticky="e")
            sv = tk.StringVar(value=default)
            e = tk.Entry(parent, textvariable=sv, bg=SURFACE2, fg=YELLOW,
                        font=FONT_MONO, relief="flat", bd=4, width=12)
            e.grid(row=0, column=col*2+1, padx=(0,20))
            setattr(self, var_name, sv)
        
        param(param_frame, "Tolerancia:", 0, "1e-8", "sv_tol")
        param(param_frame, "Máx. iter:", 1, "500", "sv_maxiter")
        param(param_frame, "Omega (ω):", 2, "1.0", "sv_omega")
        
        # Execute button
        btn = tk.Button(frame, text="▶  EJECUTAR MÉTODO",
                       bg=ACCENT, fg="#1e1e2e", font=("Segoe UI", 12, "bold"),
                       relief="flat", bd=0, padx=20, pady=10,
                       activebackground="#5a7fee", cursor="hand2",
                       command=self._ejecutar_solucionador)
        btn.pack(pady=15)
        
        # Result frame with scrollbar
        result_container = tk.Frame(frame, bg=BG)
        result_container.pack(fill="both", expand=True, pady=(15, 0))
        
        result_scrollbar = ttk.Scrollbar(result_container)
        result_scrollbar.pack(side="right", fill="y")
        
        result_canvas = tk.Canvas(result_container, bg=BG, 
                                  yscrollcommand=result_scrollbar.set,
                                  highlightthickness=0)
        result_canvas.pack(side="left", fill="both", expand=True)
        
        result_scrollbar.config(command=result_canvas.yview)
        
        self.solver_result_frame = tk.Frame(result_canvas, bg=BG)
        result_canvas.create_window((0, 0), window=self.solver_result_frame, anchor="nw")
        
        # Configure canvas to update scroll region when frame changes
        self.solver_result_frame.bind("<Configure>", 
                                      lambda e: result_canvas.configure(scrollregion=result_canvas.bbox("all")))
        
        # Enable mouse wheel scrolling
        def _on_mousewheel(event):
            result_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        def _bind_mousewheel(event):
            result_canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        def _unbind_mousewheel(event):
            result_canvas.unbind_all("<MouseWheel>")
        
        result_canvas.bind("<Enter>", _bind_mousewheel)
        result_canvas.bind("<Leave>", _unbind_mousewheel)
    
    def _ejecutar_solucionador(self):
        """Execute the fixed-point method with user-provided parameters.
        
        Reads input values, validates them, constructs the G function,
        executes the numerical method, and displays results.
        """
        for widget in self.solver_result_frame.winfo_children():
            widget.destroy()
        
        try:
            expr_g1 = self.sv_g1.get().strip()
            expr_g2 = self.sv_g2.get().strip()
            x0_val = float(self.sv_x0.get())
            y0_val = float(self.sv_y0.get())
            tol = float(self.sv_tol.get())
            maxiter = int(self.sv_maxiter.get())
            omega = float(self.sv_omega.get())
        except ValueError as e:
            messagebox.showerror("Error", f"Parámetro inválido: {e}")
            return
        
        try:
            G = build_G(expr_g1, expr_g2)
            x0 = np.array([x0_val, y0_val])
            _ = G(x0)
        except Exception as e:
            messagebox.showerror("Error", f"Error en función G: {e}")
            return
        
        # Execute method
        try:
            sol, errores, historial, iters, convergio = punto_fijo_sistema(
                G, x0, tol=tol, max_iter=maxiter, omega=omega)
        except Exception as e:
            messagebox.showerror("Error", f"Error en iteración: {e}")
            return
        
        # Save results
        self.last_solution = sol
        self.last_errors = errores
        self.last_historial = historial
        
        # Display summary
        result_frame = tk.Frame(self.solver_result_frame, bg=SURFACE, relief="flat", bd=0)
        result_frame.pack(fill="x", padx=16, pady=10)
        
        estado = "✓ CONVERGIÓ" if convergio else "✗ No convergió"
        color = ACCENT2 if convergio else RED_ERR
        
        resumen = f"""
{estado} en {iters} iteraciones
Error final: {errores[-1]:.2e}

Solución:
  x* = {sol[0]:.10f}
  y* = {sol[1]:.10f}
        """.strip()
        
        tk.Label(result_frame, text=resumen, bg=SURFACE, fg=color,
                font=("Consolas", 10), justify="left",
                anchor="nw").pack(padx=12, pady=8, fill="x")
        
        # Iteration table
        self._mostrar_tabla_iteraciones(historial, errores)
        
        # View graph button
        btn_graph = tk.Button(self.solver_result_frame, text="📈  Ver Gráfica",
                             bg=ACCENT, fg="#1e1e2e", font=("Segoe UI", 10, "bold"),
                             relief="flat", bd=0, padx=15, pady=8,
                             activebackground="#5a7fee", cursor="hand2",
                             command=self._mostrar_grafica_solucionador)
        btn_graph.pack(pady=10)
    
    def _mostrar_tabla_iteraciones(self, historial, errores):
        """Display a table with iteration history.
        
        Creates a Treeview widget showing iteration number, x value,
        y value, and error for each iteration performed.
        
        Args:
            historial: List of numpy arrays with x, y values at each iteration
            errores: List of error values at each iteration
        """
        # Table frame
        table_frame = tk.Frame(self.solver_result_frame, bg=BG)
        table_frame.pack(fill="both", expand=True, padx=16, pady=(10, 0))
        
        # Title
        tk.Label(table_frame, text="Tabla de Iteraciones", bg=BG, fg=ACCENT,
                font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(0, 10))
        
        # Create Treeview with scrollbar
        tree_frame = tk.Frame(table_frame, bg=SURFACE)
        tree_frame.pack(fill="both", expand=True)
        
        scrollbar = ttk.Scrollbar(tree_frame)
        scrollbar.pack(side="right", fill="y")
        
        tree = ttk.Treeview(tree_frame, columns=("iter", "x", "y", "error"),
                           show="headings", yscrollcommand=scrollbar.set,
                           height=15)
        
        scrollbar.config(command=tree.yview)
        tree.pack(side="left", fill="both", expand=True)
        
        # Configure columns
        tree.heading("iter", text="Iteración")
        tree.heading("x", text="x")
        tree.heading("y", text="y")
        tree.heading("error", text="Error")
        
        tree.column("iter", width=80, anchor="center")
        tree.column("x", width=150, anchor="center")
        tree.column("y", width=150, anchor="center")
        tree.column("error", width=150, anchor="center")
        
        # Configure style
        style = ttk.Style()
        style.configure("Treeview",
                       background=SURFACE2,
                       foreground=TEXT,
                       fieldbackground=SURFACE2,
                       rowheight=25)
        style.configure("Treeview.Heading",
                       background=SURFACE,
                       foreground=ACCENT,
                       font=("Segoe UI", 10, "bold"))
        style.map("Treeview",
                 background=[("selected", ACCENT)],
                 foreground=[("selected", "#1e1e2e")])
        
        # Populate table with iteration data
        for i, (h, err) in enumerate(zip(historial, errores)):
            # Skip initial point (iteration 0) for error display
            if i == 0:
                err_val = "-"
            else:
                err_val = f"{err:.2e}"
            
            tree.insert("", "end", values=(
                i,
                f"{h[0]:.10f}",
                f"{h[1]:.10f}",
                err_val
            ))
    
    def _mostrar_grafica_solucionador(self):
        """Display convergence graph in the solver section.
        
        Shows error convergence plot and iterative trajectory plot.
        Includes button to expand graph in separate window.
        """
        if self.last_solution is None:
            messagebox.showwarning("Advertencia", "Ejecuta el método primero")
            return
        
        for widget in self.solver_result_frame.winfo_children():
            widget.destroy()
        
        # Button frame
        btn_frame = tk.Frame(self.solver_result_frame, bg=BG)
        btn_frame.pack(fill="x", pady=10)
        
        btn_expand = tk.Button(btn_frame, text="🔍 Ampliar Gráfica",
                              bg=ACCENT, fg="#1e1e2e", font=("Segoe UI", 10, "bold"),
                              relief="flat", bd=0, padx=15, pady=8,
                              activebackground="#5a7fee", cursor="hand2",
                              command=self._ampliar_grafica_solucionador)
        btn_expand.pack(side="left", padx=10)
        
        fig, axes = plt.subplots(1, 2, figsize=(10, 4),
                                facecolor="#11111b")
        fig.subplots_adjust(wspace=0.35)
        
        # Convergence plot
        ax1 = axes[0]
        ax1.set_facecolor("#1e1e2e")
        ax1.semilogy(range(1, len(self.last_errors)+1), self.last_errors,
                    color=ACCENT, linewidth=1.8, marker='o', markersize=2)
        ax1.set_title("Convergencia del Error", color=TEXT, fontsize=9)
        ax1.set_xlabel("Iteración", color=TEXT_MUTED, fontsize=8)
        ax1.set_ylabel("Error (norma ∞)", color=TEXT_MUTED, fontsize=8)
        ax1.tick_params(colors=TEXT_MUTED, labelsize=7)
        for spine in ax1.spines.values():
            spine.set_edgecolor("#313149")
        ax1.grid(True, alpha=0.2, color=TEXT_MUTED)
        
        # Trajectory plot
        ax2 = axes[1]
        ax2.set_facecolor("#1e1e2e")
        hist = np.array(self.last_historial)
        ax2.plot(hist[:, 0], hist[:, 1], 'o--',
                color=ACCENT, linewidth=1.2, markersize=3, alpha=0.7,
                label="Trayectoria")
        ax2.plot(self.last_solution[0], self.last_solution[1], '*',
                color=ACCENT2, markersize=14, label="Solución")
        ax2.set_title("Trayectoria Iterativa", color=TEXT, fontsize=9)
        ax2.set_xlabel("x", color=TEXT_MUTED, fontsize=8)
        ax2.set_ylabel("y", color=TEXT_MUTED, fontsize=8)
        ax2.tick_params(colors=TEXT_MUTED, labelsize=7)
        for spine in ax2.spines.values():
            spine.set_edgecolor("#313149")
        ax2.grid(True, alpha=0.2, color=TEXT_MUTED)
        ax2.legend(fontsize=7, facecolor=SURFACE, labelcolor=TEXT)
        
        if self.canvas_widget:
            self.canvas_widget.get_tk_widget().destroy()
        
        canvas = FigureCanvasTkAgg(fig, master=self.solver_result_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, pady=10)
        self.canvas_widget = canvas
        self.fig_cache = fig
    
    def _ampliar_grafica_solucionador(self):
        """Open expanded graph in a new window.
        
        Creates a separate toplevel window with larger plots for
        better visualization of convergence and trajectory.
        """
        if self.last_solution is None:
            messagebox.showwarning("Advertencia", "Ejecuta el método primero")
            return
        
        top = tk.Toplevel(self)
        top.title("Gráfica Ampliada")
        top.geometry("1200x600")
        top.configure(bg=BG)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6),
                                facecolor="#11111b")
        fig.subplots_adjust(wspace=0.35)
        
        # Convergence
        ax1 = axes[0]
        ax1.set_facecolor("#1e1e2e")
        ax1.semilogy(range(1, len(self.last_errors)+1), self.last_errors,
                    color=ACCENT, linewidth=2.5, marker='o', markersize=4)
        ax1.set_title("Convergencia del Error", color=TEXT, fontsize=14, fontweight='bold')
        ax1.set_xlabel("Iteración", color=TEXT_MUTED, fontsize=12)
        ax1.set_ylabel("Error (norma ∞)", color=TEXT_MUTED, fontsize=12)
        ax1.tick_params(colors=TEXT_MUTED, labelsize=10)
        for spine in ax1.spines.values():
            spine.set_edgecolor("#313149")
        ax1.grid(True, alpha=0.2, color=TEXT_MUTED)
        
        # Trajectory
        ax2 = axes[1]
        ax2.set_facecolor("#1e1e2e")
        hist = np.array(self.last_historial)
        ax2.plot(hist[:, 0], hist[:, 1], 'o--',
                color=ACCENT, linewidth=2, markersize=5, alpha=0.7,
                label="Trayectoria")
        ax2.plot(self.last_solution[0], self.last_solution[1], '*',
                color=ACCENT2, markersize=20, label="Solución")
        ax2.set_title("Trayectoria Iterativa", color=TEXT, fontsize=14, fontweight='bold')
        ax2.set_xlabel("x", color=TEXT_MUTED, fontsize=12)
        ax2.set_ylabel("y", color=TEXT_MUTED, fontsize=12)
        ax2.tick_params(colors=TEXT_MUTED, labelsize=10)
        for spine in ax2.spines.values():
            spine.set_edgecolor("#313149")
        ax2.grid(True, alpha=0.2, color=TEXT_MUTED)
        ax2.legend(fontsize=10, facecolor=SURFACE, labelcolor=TEXT)
        
        canvas = FigureCanvasTkAgg(fig, master=top)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
    
    def _section_examples(self):
        """Display the predefined examples section.
        
        Shows cards for each predefined example with description,
        formulas, and execution button.
        """
        frame = tk.Frame(self.content_frame, bg=BG)
        frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        tk.Label(frame, text="Ejemplos Predefinidos", bg=BG, fg=ACCENT,
                font=FONT_HEAD).pack(anchor="w", pady=(0, 20))
        
        for name, data in EJEMPLOS.items():
            self._crear_tarjeta_ejemplo(frame, name, data)
    
    def _crear_tarjeta_ejemplo(self, parent, name, data):
        """Create an example card with description and execution button.
        
        Args:
            parent: Parent frame for the card
            name: Name of the example
            data: Dictionary containing example data (desc, g1, g2, x0, y0, omega)
        """
        card = tk.Frame(parent, bg=SURFACE, relief="flat", bd=0)
        card.pack(fill="x", pady=10)
        
        # Header
        hdr = tk.Frame(card, bg=SURFACE2)
        hdr.pack(fill="x")
        tk.Label(hdr, text=name, bg=SURFACE2, fg=ACCENT,
                font=("Segoe UI", 11, "bold")).pack(anchor="w", padx=12, pady=8)
        
        # Description
        tk.Label(card, text=data["desc"], bg=SURFACE, fg=TEXT_MUTED,
                font=("Segoe UI", 10), justify="left").pack(anchor="w", padx=12, pady=4)
        
        # Formulas
        fmla = f"G₁(x,y) = {data['g1']}\nG₂(x,y) = {data['g2']}"
        tk.Label(card, text=fmla, bg=SURFACE, fg=ACCENT,
                font=("Consolas", 9), justify="left").pack(anchor="w", padx=12, pady=4)
        
        # Execute button
        btn = tk.Button(card, text="▶  Ejecutar este ejemplo",
                       bg=ACCENT, fg="#1e1e2e", font=("Segoe UI", 10),
                       relief="flat", bd=0, padx=15, pady=8,
                       activebackground="#5a7fee", cursor="hand2",
                       command=lambda: self._ejecutar_ejemplo(name, data))
        btn.pack(pady=8, padx=12)
    
    def _ejecutar_ejemplo(self, name, data):
        """Execute a predefined example.
        
        Args:
            name: Name of the example
            data: Dictionary containing example parameters
        """
        try:
            G = build_G(data["g1"], data["g2"])
            x0 = np.array([data["x0"], data["y0"]])
            sol, errores, historial, iters, convergio = punto_fijo_sistema(
                G, x0, tol=1e-8, max_iter=500, omega=data["omega"])
            
            self.last_solution = sol
            self.last_errors = errores
            self.last_historial = historial
            
            # Show result
            msg = f"✓ Convergió en {iters} iteraciones\n\n"
            msg += f"Solución:\n  x* = {sol[0]:.10f}\n  y* = {sol[1]:.10f}"
            messagebox.showinfo(name, msg)
            
            # Go to visualizations
            self._show_section("visualizations")
        except Exception as e:
            messagebox.showerror("Error", str(e))
    
    def _section_visualizations(self):
        """Display the visualizations section.
        
        Shows convergence and trajectory plots if a computation has been
        performed. Includes button to expand graphs.
        """
        frame = tk.Frame(self.content_frame, bg=BG)
        frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        tk.Label(frame, text="Visualizaciones", bg=BG, fg=ACCENT,
                font=FONT_HEAD).pack(anchor="w", pady=(0, 15))
        
        if self.last_solution is None:
            tk.Label(frame, text="Ejecuta el método primero para generar gráficas.",
                    bg=BG, fg=TEXT_MUTED, font=("Segoe UI", 10)).pack(pady=50)
            return
        
        # Button frame
        btn_frame = tk.Frame(frame, bg=BG)
        btn_frame.pack(fill="x", pady=(0, 15))
        
        btn_expand = tk.Button(btn_frame, text="🔍 Ampliar Gráfica",
                              bg=ACCENT, fg="#1e1e2e", font=("Segoe UI", 10, "bold"),
                              relief="flat", bd=0, padx=15, pady=8,
                              activebackground="#5a7fee", cursor="hand2",
                              command=self._ampliar_grafica_visualizaciones)
        btn_expand.pack(side="left", padx=10)
        
        # Graph
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5),
                                facecolor="#11111b")
        fig.subplots_adjust(wspace=0.35)
        
        # Convergence
        ax1 = axes[0]
        ax1.set_facecolor("#1e1e2e")
        ax1.semilogy(range(1, len(self.last_errors)+1), self.last_errors,
                    color=ACCENT, linewidth=1.8, marker='o', markersize=2)
        ax1.set_title("Convergencia del Error", color=TEXT, fontsize=10)
        ax1.set_xlabel("Iteración", color=TEXT_MUTED, fontsize=9)
        ax1.set_ylabel("Error (norma ∞)", color=TEXT_MUTED, fontsize=9)
        ax1.tick_params(colors=TEXT_MUTED, labelsize=8)
        for spine in ax1.spines.values():
            spine.set_edgecolor("#313149")
        ax1.grid(True, alpha=0.2, color=TEXT_MUTED)
        
        # Trajectory
        ax2 = axes[1]
        ax2.set_facecolor("#1e1e2e")
        hist = np.array(self.last_historial)
        ax2.plot(hist[:, 0], hist[:, 1], 'o--',
                color=ACCENT, linewidth=1.2, markersize=3, alpha=0.7,
                label="Trayectoria")
        ax2.plot(self.last_solution[0], self.last_solution[1], '*',
                color=ACCENT2, markersize=14, label="Solución")
        ax2.set_title("Trayectoria Iterativa", color=TEXT, fontsize=10)
        ax2.set_xlabel("x", color=TEXT_MUTED, fontsize=9)
        ax2.set_ylabel("y", color=TEXT_MUTED, fontsize=9)
        ax2.tick_params(colors=TEXT_MUTED, labelsize=8)
        for spine in ax2.spines.values():
            spine.set_edgecolor("#313149")
        ax2.grid(True, alpha=0.2, color=TEXT_MUTED)
        ax2.legend(fontsize=8, facecolor=SURFACE, labelcolor=TEXT)
        
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
    
    def _ampliar_grafica_visualizaciones(self):
        """Open expanded visualization graph in a new window.
        
        Creates a separate toplevel window with larger plots for
        better visualization of convergence and trajectory.
        """
        if self.last_solution is None:
            messagebox.showwarning("Advertencia", "Ejecuta el método primero")
            return
        
        top = tk.Toplevel(self)
        top.title("Gráfica Ampliada - Visualizaciones")
        top.geometry("1200x600")
        top.configure(bg=BG)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6),
                                facecolor="#11111b")
        fig.subplots_adjust(wspace=0.35)
        
        # Convergence
        ax1 = axes[0]
        ax1.set_facecolor("#1e1e2e")
        ax1.semilogy(range(1, len(self.last_errors)+1), self.last_errors,
                    color=ACCENT, linewidth=2.5, marker='o', markersize=4)
        ax1.set_title("Convergencia del Error", color=TEXT, fontsize=14, fontweight='bold')
        ax1.set_xlabel("Iteración", color=TEXT_MUTED, fontsize=12)
        ax1.set_ylabel("Error (norma ∞)", color=TEXT_MUTED, fontsize=12)
        ax1.tick_params(colors=TEXT_MUTED, labelsize=10)
        for spine in ax1.spines.values():
            spine.set_edgecolor("#313149")
        ax1.grid(True, alpha=0.2, color=TEXT_MUTED)
        
        # Trajectory
        ax2 = axes[1]
        ax2.set_facecolor("#1e1e2e")
        hist = np.array(self.last_historial)
        ax2.plot(hist[:, 0], hist[:, 1], 'o--',
                color=ACCENT, linewidth=2, markersize=5, alpha=0.7,
                label="Trayectoria")
        ax2.plot(self.last_solution[0], self.last_solution[1], '*',
                color=ACCENT2, markersize=20, label="Solución")
        ax2.set_title("Trayectoria Iterativa", color=TEXT, fontsize=14, fontweight='bold')
        ax2.set_xlabel("x", color=TEXT_MUTED, fontsize=12)
        ax2.set_ylabel("y", color=TEXT_MUTED, fontsize=12)
        ax2.tick_params(colors=TEXT_MUTED, labelsize=10)
        for spine in ax2.spines.values():
            spine.set_edgecolor("#313149")
        ax2.grid(True, alpha=0.2, color=TEXT_MUTED)
        ax2.legend(fontsize=10, facecolor=SURFACE, labelcolor=TEXT)
        
        canvas = FigureCanvasTkAgg(fig, master=top)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
    
    def _section_settings(self):
        """Display the configuration section.
        
        Shows information about numerical parameters, convergence criteria,
        and recommended parameter ranges.
        """
        frame = tk.Frame(self.content_frame, bg=BG)
        frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        tk.Label(frame, text="Configuración", bg=BG, fg=ACCENT,
                font=FONT_HEAD).pack(anchor="w", pady=(0, 20))
        
        settings_text = """
PARÁMETROS PREDETERMINADOS:

• Tolerancia (tol): Criterio de convergencia
  Valor por defecto: 1e-8
  Rango recomendado: 1e-12 a 1e-4

• Máximo de iteraciones: Límite de iteraciones
  Valor por defecto: 500
  Rango recomendado: 100 a 1000

• Omega (ω): Factor de relajación
  Valor por defecto: 1.0
  Rango: 0 < ω ≤ 1
  Si ‖J‖∞ ≥ 1, usar ω < 1 para mejorar convergencia

CRITERIO DE CONVERGENCIA:
  El método converge si ‖J_G‖∞ < 1, donde J_G es la Jacobiana.
  
  Si ‖J‖∞ ≥ 1:
  • Verificar la función G(x)
  • Usar factor de relajación ω < 1
  • Cambiar punto inicial x₀
        """
        
        txt = tk.Label(frame, text=settings_text, bg=BG, fg=TEXT,
                      font=("Consolas", 9), justify="left",
                      wraplength=900)
        txt.pack(anchor="nw", pady=20)
    
    def _section_about(self):
        """Display the about section with application information.
        
        Shows description, technology stack, and features of the application.
        """
        frame = tk.Frame(self.content_frame, bg=BG)
        frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        tk.Label(frame, text="Acerca de", bg=BG, fg=ACCENT,
                font=FONT_HEAD).pack(anchor="w", pady=(0, 20))
        
        about_text = """
MÉTODO DEL PUNTO FIJO

Institución: Universidad Distrital Francisco José de Caldas
Asignatura: Métodos Numéricos
Periodo: 2026-1

DESCRIPCIÓN:
Aplicación para resolver sistemas de ecuaciones no lineales 
mediante el método del punto fijo (fixed-point iteration).

TECNOLOGÍA:
• Python 3.x
• Tkinter (GUI)
• NumPy (Cálculos numéricos)
• Matplotlib (Visualización)

CARACTERÍSTICAS:
✓ Interfaz intuitiva y moderna
✓ Ejecución bajo demanda (sin gráficas automáticas)
✓ Análisis de convergencia mediante Jacobiana
✓ Ejemplos predefinidos
✓ Visualización de trayectorias e iteraciones
✓ Parámetros personalizables
        """
        
        txt = tk.Label(frame, text=about_text, bg=BG, fg=TEXT,
                      font=("Consolas", 9), justify="left",
                      wraplength=900, anchor="nw")
        txt.pack(pady=20, fill="both", expand=True)


def launch_interface():
    """Launch the graphical interface application.
    
    Returns:
        MainApplication instance
    """
    app = MainApplication()
    return app
