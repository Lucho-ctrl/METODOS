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
    norma_jacobiana,
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
        self.last_jacobian = None
        self.last_jacobian_norm = None
        self.last_var_names = None
        
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
        
        Provides input fields for functions G_i(x), initial values,
        and numerical parameters. Supports n-dimensional systems.
        """
        frame = tk.Frame(self.content_frame, bg=BG)
        frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        tk.Label(frame, text="Solucionador Interactivo", bg=BG, fg=ACCENT,
                font=FONT_HEAD).pack(anchor="w", pady=(0, 15))
        
        # Instructions
        info_frame = tk.Frame(frame, bg=SURFACE, bd=0, relief="flat")
        info_frame.pack(fill="x", pady=(0, 15))
        tk.Label(info_frame, text="ℹ  Ingresa el número de funciones y luego las expresiones G_i\n"
                                 "   Operadores: + - * / ** sqrt() exp() log() sin() cos() tan() abs()",
                bg=SURFACE, fg=TEXT_MUTED, font=("Segoe UI", 9), justify="left",
                wraplength=800).pack(padx=12, pady=8, anchor="w")
        
        # Number of functions input
        num_func_frame = tk.Frame(frame, bg=BG)
        num_func_frame.pack(fill="x", pady=(0, 15))
        
        tk.Label(num_func_frame, text="Número de funciones:", bg=BG, fg=TEXT,
                font=FONT_HEAD, width=15, anchor="e").pack(side="left", padx=(0, 10))
        
        self.sv_num_funcs = tk.StringVar(value="2")
        num_entry = tk.Entry(num_func_frame, textvariable=self.sv_num_funcs, 
                            bg=SURFACE2, fg=ACCENT, font=("Consolas", 11),
                            insertbackground=ACCENT, relief="flat", bd=6, width=10)
        num_entry.pack(side="left")
        
        btn_update = tk.Button(num_func_frame, text="Actualizar",
                              bg=ACCENT, fg="#1e1e2e", font=("Segoe UI", 9, "bold"),
                              relief="flat", bd=0, padx=10, pady=5,
                              activebackground="#5a7fee", cursor="hand2",
                              command=self._actualizar_campos_funciones)
        btn_update.pack(side="left", padx=(10, 0))
        
        # Dynamic input fields container
        self.func_input_frame = tk.Frame(frame, bg=BG)
        self.func_input_frame.pack(fill="x", pady=(0, 15))
        
        # Initialize with 2 functions
        self._actualizar_campos_funciones()
        
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
    
    def _actualizar_campos_funciones(self):
        """Update the input fields based on the number of functions.
        
        Dynamically creates input fields for each function G_i and initial value.
        """
        try:
            num_funcs = int(self.sv_num_funcs.get())
            if num_funcs < 1:
                num_funcs = 1
                self.sv_num_funcs.set("1")
        except ValueError:
            num_funcs = 2
            self.sv_num_funcs.set("2")
        
        # Clear existing fields
        for widget in self.func_input_frame.winfo_children():
            widget.destroy()
        
        # Generate variable names
        var_names = []
        for i in range(num_funcs):
            if i == 0:
                var_names.append('x')
            elif i == 1:
                var_names.append('y')
            elif i == 2:
                var_names.append('z')
            else:
                var_names.append(chr(ord('w') + (i - 3)))
        
        self.sv_funciones = []
        self.sv_iniciales = []
        
        # Create function input fields
        for i, var_name in enumerate(var_names):
            row_frame = tk.Frame(self.func_input_frame, bg=BG)
            row_frame.pack(fill="x", pady=4)
            
            # Function expression
            tk.Label(row_frame, text=f"G_{i+1}({', '.join(var_names)}) =", 
                    bg=BG, fg=TEXT, font=FONT_HEAD, width=20, anchor="e").pack(side="left", padx=(0, 10))
            
            sv_func = tk.StringVar(value=f"sqrt(1 - {var_names[(i+1) % len(var_names)]})" if i < 2 else "0")
            entry_func = tk.Entry(row_frame, textvariable=sv_func, bg=SURFACE2, fg=ACCENT,
                                font=("Consolas", 11), insertbackground=ACCENT,
                                relief="flat", bd=6, width=40)
            entry_func.pack(side="left", padx=(0, 20))
            self.sv_funciones.append(sv_func)
            
            # Initial value
            tk.Label(row_frame, text=f"{var_name}₀ =", 
                    bg=BG, fg=TEXT, font=FONT_HEAD, width=8, anchor="e").pack(side="left", padx=(0, 10))
            
            sv_init = tk.StringVar(value="0.5")
            entry_init = tk.Entry(row_frame, textvariable=sv_init, bg=SURFACE2, fg=ACCENT,
                                font=("Consolas", 11), insertbackground=ACCENT,
                                relief="flat", bd=6, width=12)
            entry_init.pack(side="left")
            self.sv_iniciales.append(sv_init)
    
    def _ejecutar_solucionador(self):
        """Execute the fixed-point method with user-provided parameters.
        
        Reads input values, validates them, constructs the G function,
        executes the numerical method, and displays results.
        """
        for widget in self.solver_result_frame.winfo_children():
            widget.destroy()
        
        try:
            # Get number of functions
            num_funcs = int(self.sv_num_funcs.get())
            
            # Get function expressions
            expr_list = [sv.get().strip() for sv in self.sv_funciones]
            
            # Get initial values
            x0_vals = [float(sv.get()) for sv in self.sv_iniciales]
            
            # Get parameters
            tol = float(self.sv_tol.get())
            maxiter = int(self.sv_maxiter.get())
            omega = float(self.sv_omega.get())
        except ValueError as e:
            messagebox.showerror("Error", f"Parámetro inválido: {e}")
            return
        
        try:
            # Generate variable names
            var_names = []
            for i in range(num_funcs):
                if i == 0:
                    var_names.append('x')
                elif i == 1:
                    var_names.append('y')
                elif i == 2:
                    var_names.append('z')
                else:
                    var_names.append(chr(ord('w') + (i - 3)))
            
            G = build_G(expr_list, var_names)
            x0 = np.array(x0_vals)
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
        
        # Calculate Jacobian matrix at solution
        try:
            norm_jac, jac_matrix = norma_jacobiana(G, sol)
        except Exception:
            norm_jac, jac_matrix = None, None
        
        # Save results
        self.last_solution = sol
        self.last_errors = errores
        self.last_historial = historial
        self.last_var_names = var_names
        self.last_jacobian = jac_matrix
        self.last_jacobian_norm = norm_jac
        
        # Display summary
        result_frame = tk.Frame(self.solver_result_frame, bg=SURFACE, relief="flat", bd=0)
        result_frame.pack(fill="x", padx=16, pady=10)
        
        estado = "✓ CONVERGIÓ" if convergio else "✗ No convergió"
        color = ACCENT2 if convergio else RED_ERR
        
        # Build solution string
        sol_str = "\n".join([f"  {var_names[i]}* = {sol[i]:.10f}" for i in range(len(sol))])
        
        resumen = f"""
{estado} en {iters} iteraciones
Error final: {errores[-1]:.2e}

Solución:
{sol_str}
        """.strip()
        
        tk.Label(result_frame, text=resumen, bg=SURFACE, fg=color,
                font=("Consolas", 10), justify="left",
                anchor="nw").pack(padx=12, pady=8, fill="x")
        
        # Display Jacobian matrix
        if jac_matrix is not None:
            jac_frame = tk.Frame(self.solver_result_frame, bg=SURFACE, relief="flat", bd=0)
            jac_frame.pack(fill="x", padx=16, pady=10)
            
            jac_color = ACCENT if norm_jac < 1 else RED_ERR
            jac_text = f"Matriz Jacobiana (‖J‖∞ = {norm_jac:.4f})\n"
            
            # Format Jacobian matrix as string
            for i in range(jac_matrix.shape[0]):
                row_str = "  ".join([f"{jac_matrix[i, j]:.6f}" for j in range(jac_matrix.shape[1])])
                jac_text += f"  {row_str}\n"
            
            tk.Label(jac_frame, text=jac_text, bg=SURFACE, fg=jac_color,
                    font=("Consolas", 9), justify="left",
                    anchor="nw").pack(padx=12, pady=8, fill="x")
        
        # Iteration table
        self._mostrar_tabla_iteraciones(historial, errores, var_names)
        
        # View graph button (only for 2D systems)
        if num_funcs == 2:
            btn_graph = tk.Button(self.solver_result_frame, text="📈  Ver Gráfica",
                                 bg=ACCENT, fg="#1e1e2e", font=("Segoe UI", 10, "bold"),
                                 relief="flat", bd=0, padx=15, pady=8,
                                 activebackground="#5a7fee", cursor="hand2",
                                 command=self._mostrar_grafica_solucionador)
            btn_graph.pack(pady=10)
    
    def _mostrar_tabla_iteraciones(self, historial, errores, var_names):
        """Display a table with iteration history.
        
        Creates a Treeview widget showing iteration number, variable values,
        and error for each iteration performed. Supports n-dimensional systems.
        
        Args:
            historial: List of numpy arrays with variable values at each iteration
            errores: List of error values at each iteration
            var_names: List of variable names (e.g., ['x', 'y', 'z'])
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
        
        # Create dynamic columns based on number of variables
        columns = ["iter"] + var_names + ["error"]
        tree = ttk.Treeview(tree_frame, columns=columns,
                           show="headings", yscrollcommand=scrollbar.set,
                           height=15)
        
        scrollbar.config(command=tree.yview)
        tree.pack(side="left", fill="both", expand=True)
        
        # Configure columns
        tree.heading("iter", text="Iteración")
        tree.column("iter", width=80, anchor="center")
        
        for var_name in var_names:
            tree.heading(var_name, text=var_name)
            tree.column(var_name, width=150, anchor="center")
        
        tree.heading("error", text="Error")
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
            
            # Build row values
            row_values = [i] + [f"{h[j]:.10f}" for j in range(len(h))] + [err_val]
            tree.insert("", "end", values=row_values)
    
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
            data: Dictionary containing example data (desc, expr_list, var_names, x0, omega)
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
        var_str = ", ".join(data["var_names"])
        fmla_lines = [f"G_{i+1}({var_str}) = {expr}" for i, expr in enumerate(data["expr_list"])]
        fmla = "\n".join(fmla_lines)
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
            data: Dictionary containing example data (expr_list, var_names, x0, omega)
        """
        try:
            G = build_G(data["expr_list"], data["var_names"])
            x0 = np.array(data["x0"])
            sol, errores, historial, iters, convergio = punto_fijo_sistema(
                G, x0, tol=1e-8, max_iter=500, omega=data["omega"])
            
            # Calculate Jacobian
            norm_jac, jac_matrix = norma_jacobiana(G, sol)
            
            self.last_solution = sol
            self.last_errors = errores
            self.last_historial = historial
            self.last_var_names = data["var_names"]
            self.last_jacobian = jac_matrix
            self.last_jacobian_norm = norm_jac
            
            # Show result
            sol_str = "\n".join([f"  {data['var_names'][i]}* = {sol[i]:.10f}" for i in range(len(sol))])
            msg = f"✓ Convergió en {iters} iteraciones\n\nSolución:\n{sol_str}"
            messagebox.showinfo(name, msg)
            
            # Go to visualizations
            self._show_section("visualizations")
        except Exception as e:
            messagebox.showerror("Error", str(e))
    
    def _section_visualizations(self):
        """Display the visualizations section.
        
        Shows convergence plot (for all dimensions) and trajectory plot
        (only for 2D systems). Includes button to expand graphs.
        """
        frame = tk.Frame(self.content_frame, bg=BG)
        frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        tk.Label(frame, text="Visualizaciones", bg=BG, fg=ACCENT,
                font=FONT_HEAD).pack(anchor="w", pady=(0, 15))
        
        if self.last_solution is None:
            tk.Label(frame, text="Ejecuta el método primero para generar gráficas.",
                    bg=BG, fg=TEXT_MUTED, font=("Segoe UI", 10)).pack(pady=50)
            return
        
        # Display Jacobian matrix
        if self.last_jacobian is not None:
            jac_frame = tk.Frame(frame, bg=SURFACE, relief="flat", bd=0)
            jac_frame.pack(fill="x", pady=(0, 15))
            
            jac_color = ACCENT if self.last_jacobian_norm < 1 else RED_ERR
            convergence_status = "CONVERGE" if self.last_jacobian_norm < 1 else "NO CONVERGE"
            
            jac_header = f"MATRIZ JACOBIANA (‖J‖∞ = {self.last_jacobian_norm:.6f}) - {convergence_status}"
            tk.Label(jac_frame, text=jac_header, bg=SURFACE, fg=jac_color,
                    font=("Segoe UI", 11, "bold")).pack(padx=12, pady=(8, 4), anchor="w")
            
            # Format Jacobian matrix as string
            jac_text = ""
            for i in range(self.last_jacobian.shape[0]):
                row_str = "  ".join([f"{self.last_jacobian[i, j]:.6f}" for j in range(self.last_jacobian.shape[1])])
                jac_text += f"  {row_str}\n"
            
            tk.Label(jac_frame, text=jac_text, bg=SURFACE, fg=TEXT,
                    font=("Consolas", 10), justify="left",
                    anchor="nw").pack(padx=12, pady=(0, 8), fill="x")
        
        # Check dimensionality
        num_dims = len(self.last_solution)
        
        # Button frame
        btn_frame = tk.Frame(frame, bg=BG)
        btn_frame.pack(fill="x", pady=(0, 15))
        
        btn_expand = tk.Button(btn_frame, text="🔍 Ampliar Gráfica",
                              bg=ACCENT, fg="#1e1e2e", font=("Segoe UI", 10, "bold"),
                              relief="flat", bd=0, padx=15, pady=8,
                              activebackground="#5a7fee", cursor="hand2",
                              command=self._ampliar_grafica_visualizaciones)
        btn_expand.pack(side="left", padx=10)
        
        # Graph - show convergence for all dimensions, trajectory only for 2D
        if num_dims == 2:
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
        else:
            # Only convergence plot for n > 2 dimensions
            fig, ax = plt.subplots(1, 1, figsize=(10, 4.5),
                                   facecolor="#11111b")
            ax.set_facecolor("#1e1e2e")
            ax.semilogy(range(1, len(self.last_errors)+1), self.last_errors,
                        color=ACCENT, linewidth=1.8, marker='o', markersize=2)
            ax.set_title("Convergencia del Error", color=TEXT, fontsize=10)
            ax.set_xlabel("Iteración", color=TEXT_MUTED, fontsize=9)
            ax.set_ylabel("Error (norma ∞)", color=TEXT_MUTED, fontsize=9)
            ax.tick_params(colors=TEXT_MUTED, labelsize=8)
            for spine in ax.spines.values():
                spine.set_edgecolor("#313149")
            ax.grid(True, alpha=0.2, color=TEXT_MUTED)
            
            # Add note about trajectory
            tk.Label(frame, text=f"Nota: La trayectoria solo se muestra para sistemas 2D. Sistema actual: {num_dims}D",
                    bg=BG, fg=TEXT_MUTED, font=("Segoe UI", 9)).pack(pady=5)
        
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
    
    def _ampliar_grafica_visualizaciones(self):
        """Open expanded visualization graph in a new window.
        
        Creates a separate toplevel window with larger plots for
        better visualization of convergence and trajectory (2D only).
        """
        if self.last_solution is None:
            messagebox.showwarning("Advertencia", "Ejecuta el método primero")
            return
        
        top = tk.Toplevel(self)
        top.title("Gráfica Ampliada - Visualizaciones")
        top.geometry("1200x600")
        top.configure(bg=BG)
        
        # Check dimensionality
        num_dims = len(self.last_solution)
        
        if num_dims == 2:
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
        else:
            # Only convergence plot for n > 2 dimensions
            fig, ax = plt.subplots(1, 1, figsize=(14, 6),
                                   facecolor="#11111b")
            ax.set_facecolor("#1e1e2e")
            ax.semilogy(range(1, len(self.last_errors)+1), self.last_errors,
                        color=ACCENT, linewidth=2.5, marker='o', markersize=4)
            ax.set_title("Convergencia del Error", color=TEXT, fontsize=14, fontweight='bold')
            ax.set_xlabel("Iteración", color=TEXT_MUTED, fontsize=12)
            ax.set_ylabel("Error (norma ∞)", color=TEXT_MUTED, fontsize=12)
            ax.tick_params(colors=TEXT_MUTED, labelsize=10)
            for spine in ax.spines.values():
                spine.set_edgecolor("#313149")
            ax.grid(True, alpha=0.2, color=TEXT_MUTED)
        
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
