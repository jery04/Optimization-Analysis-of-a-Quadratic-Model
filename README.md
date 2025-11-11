# Optimization Analysis of a Quadratic Model 🔍📐✨

A small, visual, and hands-on project to study unconstrained optimization of a quadratic function using two classic methods: Steepest Descent with optimal step size and Newton’s Method. We also generate 3D surfaces and contour plots to visualize the optimization paths. 🎯📉🗺️

---

## 🌟 Introduction
This repository explores the minimization of a convex quadratic function of the form

- f(x) = 1/2 xᵀ Q x + cᵀ x + r

In this project, the specific scalar function we analyze is:

$$
f(x, y) = (x + y - 7)^2 + (2x + y - 5)^2
$$

We implement and compare:
- Steepest Descent (with analytically optimal step size) 🏃‍♂️💨
- Newton’s Method (one-step solution for quadratics) ⚡🧠

You’ll see trajectories over contour plots and a 3D surface of the objective, making the behavior of both methods crystal clear. 📊🌈

---

## 📁 Project Structure

- `index.py` ✨
  - All-in-one script that defines the quadratic model, runs Steepest Descent and Newton’s Method, and renders 3D surface + contour trajectories. Great for a quick demo!
- `Scripts/` 🧩
  - `max_descenso.py` ➜ Steepest Descent with optimal step size (αₖ = (gᵀg)/(gᵀQg)); includes a simple CLI demo. 🏃‍♀️
  - `newton.py` ➜ Newton’s method specialized for quadratics (one iteration to optimal). ⚙️
  - `graficar.py` ➜ Plotting utilities (3D surface and contour trajectory) and an example that uses `max_descenso.py` and `newton.py`. 🎨🛰️
- `analisis_optimizacion.ipynb` 📓
  - Optional Jupyter notebook for interactive exploration and visualization.

Artifacts produced when running the demos include printed results and interactive plots. 🖨️📈

---

## 🧠 Problem Setup
The example problem used throughout is:

- Q = [[10, 6], [6, 4]]
- c = [-34, -24]
- r = 74

The unique minimizer is x* = -Q⁻¹c = (-2, 9), and the minimum value is f(x*) = 0. ✅

---

## 🚀 How to Run
You need Python 3.9+ with NumPy and Matplotlib. From PowerShell on Windows:

```powershell
# 1) Go to the project folder (note the quotes because of spaces/accents)
cd "D:\Modelo de Optimización"

# 2) (Optional) Activate your environment
# conda activate base

# 3) Quick all-in-one demo: runs methods + shows plots
python index.py
```

You should see console outputs with the optimal point and two plots:
- A 3D surface of f(x, y) 🌋
- Contour plots with the optimization paths (Steepest Descent and Newton) 🌀🧭

Alternative demo using the modular scripts:

```powershell
cd "D:\Modelo de Optimización\Scripts"

# Steepest Descent (prints results)
python .\max_descenso.py

# Newton’s method (prints results)
python .\newton.py

# Plotting demo using both methods
python .\graficar.py
```

---

## ✅ How to Test (Lightweight)
No formal test framework is required. You can verify correctness by checking that:
- Both methods return approximately x* = (-2, 9) 🧲
- The objective at the solution is f(x*) ≈ 0 🟢

Quick interactive check in Python:

```powershell
python - <<'PY'
import numpy as np
from Scripts.max_descenso import maximo_descenso_optimo
from Scripts.newton import newton_quadratic

Q = np.array([[10., 6.],[6., 4.]])
c = np.array([-34., -24.])
r = 74.0
x0 = np.array([0., 0.])

x_sd, info = maximo_descenso_optimo(Q, c, x0, tol=1e-9, max_iter=1000, verbose=False)
x_nt, xs = newton_quadratic(Q, c, x0)

f = lambda x: 0.5 * x @ (Q @ x) + c @ x + r
print("Steepest Descent x*:", x_sd, "f(x*):", f(x_sd))
print("Newton x*:", x_nt, "f(x*):", f(x_nt))
PY
```

Expected output (up to tiny rounding):
- Steepest Descent x* ≈ [-2.  9.], f(x*) ≈ 0.0
- Newton x* ≈ [-2.  9.], f(x*) ≈ 0.0

---

## 🔬 Methods Overview
- Steepest Descent with optimal step size 🏃‍♂️
  - Direction: -∇f(xₖ) = -(Qxₖ + c)
  - Step: αₖ = (gᵀg)/(gᵀQg)
  - Converges linearly on quadratics; the optimal αₖ formula ensures efficient progress.

- Newton’s Method 🧠
  - For quadratics, one iteration from any x₀ yields x* = -Q⁻¹c.
  - In practice, we solve Qp = -(Qx₀ + c) and take x₁ = x₀ + p.

---

## 🧩 Visualizations
- 3D surface of f(x, y) with a color map 🌈
- Contour plots with overlayed trajectories 🔁
- Customizable ranges, resolution, markers, and levels 🎛️

These make it easy to compare the behavior of both methods and see the path to the minimum. 👀📍

---

## 🧾 Conclusions
- Newton’s method hits the exact minimizer in a single iteration for quadratic objectives. ⚡
- Steepest Descent with the optimal step size converges reliably, illustrating a zig-zag path when level sets are elongated. 🔻➡️🔻
- Visualizations confirm both methods reach x* = (-2, 9) with f(x*) = 0. 🎉
- The project serves as a compact, didactic reference for quadratic optimization and method comparison. 📚

---

## 📦 Requirements
- Python 3.9+
- NumPy 🧮
- Matplotlib 📊

Install (optional, if needed):
```powershell
pip install numpy matplotlib
```

---

## 📝 Notes
- Matrices are symmetrized defensively in Steepest Descent to avoid numerical issues. 🛡️
- The plotting functions include sensible defaults and options for camera/view control. 🎥
- Paths with spaces/accents require quoting in Windows PowerShell (as shown). 🪟

Enjoy exploring optimization with math and visuals! 💫