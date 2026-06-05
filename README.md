# Solucionador del Método del Punto Fijo para Sistemas No Lineales

## Descripción

Aplicación GUI profesional para resolver sistemas de ecuaciones no lineales usando el método de iteración de punto fijo. El solucionador proporciona una interfaz intuitiva con visualizaciones bajo demanda, ejemplos predefinidos y análisis de convergencia en tiempo real.

## Características

- **Solucionador Interactivo**: Ingresa funciones personalizadas G₁(x,y) y G₂(x,y) con parámetros definidos por el usuario
- **Ejemplos Predefinidos**: Tres sistemas de ejemplo integrados (cuadrático, círculo/parábola, exponencial)
- **Visualizaciones Bajo Demanda**: Curvas de convergencia y gráficas de trayectoria iterativa con gráficas ampliables
- **Análisis Matemático**: Cálculo de jacobiana mediante diferencias finitas y verificación de criterios de convergencia
- **Parámetro de Relajación**: Parámetro Omega (ω) para mejorar la convergencia
- **Interfaz Moderna**: Ventana única con navegación de barra lateral, tema oscuro y modo pantalla completa automático

## Arquitectura del Proyecto

```
METODOS/
├── main_app.py              # Módulo de núcleo numérico
├── Interfaz_grafica.py      # Módulo de interfaz gráfica
├── README.md                # Este archivo
└── Graficas/                # Directorio de salida para visualizaciones guardadas
```

### Descripción General de la Arquitectura

**Componentes Principales:**

1. **Núcleo Numérico** (`main_app.py`)
   - `punto_fijo_sistema()`: Algoritmo de iteración de punto fijo con relajación
   - `norma_jacobiana()`: Cálculo de jacobiana mediante diferencias finitas
   - `build_G()`: Analizador seguro de expresiones para entrada de usuario
   - `EJEMPLOS`: Sistemas de ejemplo predefinidos

2. **Interfaz Gráfica** (`Interfaz_grafica.py`)
   - `MainApplication`: Clase GUI principal con navegación de barra lateral
   - Seis secciones: Inicio, Solucionador, Ejemplos, Visualizaciones, Configuración, Acerca de
   - Gestión de estado para persistencia de resultados entre secciones
   - Renderizado de gráficas bajo demanda usando Matplotlib

3. **Flujo de Datos**
   ```
   Entrada del Usuario → Analizador de Expresiones → Solucionador Numérico → Almacenamiento de Estado
                                                             ↓
                                       Análisis de Convergencia
                                                             ↓
                                    Visualización (Bajo Demanda)
   ```

## Instalación

### Requisitos

- Python 3.7+
- tkinter (generalmente incluido con Python)
- NumPy
- Matplotlib

### Configuración

```bash
# Navega al directorio del proyecto
cd METODOS

# Instala las dependencias
pip install numpy matplotlib

# Ejecuta la aplicación
python main_app.py
```

### Entorno Virtual (Recomendado)

```bash
python -m venv venv
# En Windows:
venv\Scripts\activate
# En Linux/Mac:
source venv/bin/activate

pip install numpy matplotlib
```

## Uso

### Lanzar la Aplicación

```bash
python main_app.py
```

La aplicación se abre automáticamente en modo pantalla completa.

### 1. Sección Inicio (🏠)
Guía de inicio rápido y descripción general de características. Comienza aquí si eres nuevo en la aplicación.

### 2. Sección Solucionador (🧮)
**Ingresa tus propias funciones:**

1. Ingresa las funciones de iteración G₁(x,y) y G₂(x,y)
   - Ejemplo: `G₁ = sqrt(1 - y)`, `G₂ = sqrt(1 - x)`
   
2. Especifica la aproximación inicial (x₀, y₀)
   - Ejemplo: `x₀ = 0.5, y₀ = 0.5`

3. Establece los parámetros:
   - **Tolerancia**: Criterio de convergencia (predeterminado: 1e-8)
   - **Máx. iteraciones**: Límite de seguridad (predeterminado: 500)
   - **Omega (ω)**: Parámetro de relajación (predeterminado: 1.0)

4. Haz clic en **▶ EJECUTAR MÉTODO**

5. Visualiza el resumen de resultados

6. Haz clic en **📈 Ver Gráfica** para ver las visualizaciones

### 3. Sección Ejemplos (📚)
Ejecuta ejemplos predefinidos con un clic:

- **Ejemplo 1**: x² + y = 1, x + y² = 1
- **Ejemplo 2**: Círculo (x²+y²=4) ∩ Parábola (y=x²-1)
- **Ejemplo 3**: Sistema exponencial (x·e^y=2, y·e^x=3)

### 4. Sección Visualizaciones (📈)
Ve las gráficas de convergencia y trayectorias. Haz clic en **🔍 Ampliar Gráfica** para vista más grande.

### 5. Sección Configuración (⚙️)
Guía de parámetros y explicación de criterios de convergencia.

### 6. Sección Acerca de (ℹ️)
Información de la aplicación y pila tecnológica.

## Ejemplos Matemáticos

### Ejemplo 1: Sistema Cuadrático Simple

**Sistema:**
```
x² + y = 1
x + y² = 1
```

**Funciones de Iteración:**
```
G₁(x,y) = sqrt(1 - y)
G₂(x,y) = sqrt(1 - x)
```

**Solución:**
```
x* ≈ 0.6823278...
y* ≈ 0.6823278...
```

**Convergencia:** ~15 iteraciones con tolerancia predeterminada

### Ejemplo 2: Intersección de Círculo y Parábola

**Sistema:**
```
x² + y² = 4         (círculo)
y = x² - 1          (parábola)
```

**Funciones de Iteración:**
```
G₁(x,y) = sqrt(y + 1)
G₂(x,y) = sqrt(4 - x²)
```

**Configuración:** ω = 0.5, x₀ = 1.5, y₀ = 1.0

### Ejemplo 3: Sistema Exponencial

**Sistema:**
```
x·e^y = 2
y·e^x = 3
```

**Funciones de Iteración:**
```
G₁(x,y) = 2/e^y
G₂(x,y) = 3/e^x
```

**Configuración:** ω = 0.5, máx. iteraciones = 500

## Trasfondo Matemático

### Método de Iteración de Punto Fijo

Para un sistema **x = G(x)**, la iteración es:
```
x^(k+1) = G(x^(k))
```

O con parámetro de relajación:
```
x^(k+1) = (1-ω)·x^(k) + ω·G(x^(k))
```

### Criterio de Convergencia

La convergencia local está garantizada si:
```
‖J_G(x*)‖∞ < 1
```

donde J_G es la matriz jacobiana de G.

### Medición del Error

El algoritmo usa la norma infinita (máximo componente absoluto):
```
error = ‖x^(k+1) - x^(k)‖∞ = max(|Δx_i|)
```

## Solución de Problemas

### "Error en Expresión"

- Verifica la sintaxis: usa solo operadores soportados (`+, -, *, /, **`)
- Funciones válidas: `sqrt(), exp(), log(), sin(), cos(), tan(), abs()`
- Las variables deben ser `x` e `y` (sensible a mayúsculas/minúsculas)
- Ejemplo correcto: `sqrt(1 - y)` ✓
- Ejemplo incorrecto: `sqrt{1 - y}` ✗

### "Norma Jacobiana ≥ 1"

La convergencia puede no ocurrir:
- Intenta una aproximación inicial diferente (x₀, y₀)
- Usa parámetro de relajación ω < 1 (recomendado: 0.5-0.8)
- Verifica la función de iteración G(x)
- Consulta la sección Configuración para más detalles

### "Sin Convergencia"

- Aumenta el número máximo de iteraciones
- Disminuye ligeramente la tolerancia
- Asegura que la aproximación inicial esté en el dominio de G
- Considera usar parámetro de relajación ω < 1

### Errores de Importación de Módulos

```bash
# Actualiza las dependencias
pip install --upgrade numpy matplotlib

# O reinstala desde cero
pip uninstall numpy matplotlib
pip install numpy matplotlib
```

## Referencias

- Numerical Recipes: The Art of Scientific Computing
- Kincaid & Cheney: Numerical Analysis
- Burden & Faires: Numerical Analysis

## Soporte

Para problemas o preguntas:
1. Consulta la sección "Solución de Problemas"
2. Verifica los ejemplos predefinidos
3. Revisa la documentación en cada sección de la aplicación
