# %% [markdown]
# # Reconstrucción del campo de flujo en una cavidad cuadrada
# ## TP N°3: Influencia de las estrategias de muestreo de puntos de colocación
# ### Redes Neuronales Informadas por Física - Maestria en Inteligencia Artificial
# #### Grupo N°4: Jorge Ceferino Valdez, Fabian Sarmiento y Trinidad Monreal.
# ---

# %% [markdown]
# Buscamos reconstruir el campo de flujo estacionario en una cavidad cuadrada usando una Red Neuronal Informada por Física (PINN). Se trata de resolver las ecuaciones de Navier-Stokes incomprensibles:
# 
# $$(\mathbf{u} \cdot \nabla)\mathbf{u} = -\nabla p + \frac{1}{Re} \nabla^2 \mathbf{u} \text{,  en Ω}$$
# 
# $$\nabla \cdot \mathbf{u} = 0 \text{,  en Ω}$$
# 
# con las siguientes condiciones de borde:
# 
# - _No-slip_ en las fronteras laterales e inferior ($\mathbf{u} = (0,0)$)
# - Velocidad constante en direccion $+x$ en la frontera superior ($\mathbf{u} = (1,0)$)
# 
# y $Ω = [0,1]⊗[0,1]$

# %%
import os
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.utils.data import Dataset, DataLoader, TensorDataset, ConcatDataset

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d
from scipy.io import loadmat
from scipy.interpolate import griddata

from IPython.display import clear_output
import time
from datetime import datetime, timedelta
import gc
from tqdm import trange

np.random.seed(42)

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    device_name = "CUDA GPU"
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    device_name = "Apple Silicon GPU (MPS)"
else:
    device = torch.device("cpu")
    device_name = "CPU"

# Configurar semillas para reproducibilidad
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
elif torch.backends.mps.is_available():
    # MPS usa la misma semilla que PyTorch general
    torch.manual_seed(42)

print(f"Usando dispositivo: {device} ({device_name})")

# %% [markdown]
# ### Clase PINN Navier-stokes

# %%
class PINN_Module(nn.Module):
    """
    Red Neuronal Informada por Física para resolver las ecuaciones de Navier-Stokes
    en una cavidad cuadrada con lid-driven flow
    """
    
    def __init__(self, model_parameters):
        super(PINN_Module, self).__init__()
        self.Device = model_parameters["Device"]
        self.LowerBounds = model_parameters["LowerBounds"]
        self.UpperBounds = model_parameters["UpperBounds"]
        self.Re = model_parameters["Re"]
        self.InputDimensions = model_parameters["InputDimensions"]
        self.OutputDimensions = model_parameters["OutputDimensions"]
        self.NumberOfNeurons = model_parameters["NumberOfNeurons"]
        self.NumberOfHiddenLayers = model_parameters["NumberOfHiddenLayers"]
        self.ActivationFunction = model_parameters["ActivationFunction"]
        
        # Definir arquitectura de la red
        self.InputLayer = nn.Linear(self.InputDimensions, self.NumberOfNeurons)
        self.HiddenLayers = nn.ModuleList(
            [nn.Linear(self.NumberOfNeurons, self.NumberOfNeurons) 
             for _ in range(self.NumberOfHiddenLayers - 1)])
        self.OutputLayer = nn.Linear(self.NumberOfNeurons, self.OutputDimensions)
        
        # Inicialización Xavier
        self.init_xavier()

    def forward(self, X):
        """
        Forward pass de la red neuronal
        Input: X tensor de forma (N, 2) con coordenadas (x, y)
        Output: tensor de forma (N, 3) con (u, v, p)
        """
        lb = self.LowerBounds
        ub = self.UpperBounds
        
        # Normalización de entradas a [-1, 1]
        X = 2 * (X - lb) / (ub - lb) - 1
        
        # Forward pass
        output = self.ActivationFunction(self.InputLayer(X))
        for k, l in enumerate(self.HiddenLayers):
            output = self.ActivationFunction(l(output))
        output = self.OutputLayer(output)
        
        return output

    def predict(self, X):
        """
        Predicción directa de u, v, p a partir de coordenadas (x, y).
        """
        uvp = self.forward(X)
        u = uvp[:, 0:1]
        v = uvp[:, 1:2]
        p = uvp[:, 2:3]
        return u, v, p

    def init_xavier(self):
        """Inicialización Xavier mejorada de los pesos"""
        def init_weights(m):
            if type(m) == nn.Linear and m.weight.requires_grad and m.bias.requires_grad:
                g = nn.init.calculate_gain('tanh')
                torch.nn.init.xavier_uniform_(m.weight, gain=g)
                # Bias no cero para romper simetría
                torch.nn.init.uniform_(m.bias, -0.1, 0.1)
        
        self.apply(init_weights)

    def navierstokesResidue(self, X, uvp):
        """
        Calcula los residuos de las ecuaciones de Navier-Stokes
        
        Ecuaciones:
        (u·∇)u = -∇p + (1/Re)∇²u  →  u*∂u/∂x + v*∂u/∂y + ∂p/∂x - (1/Re)(∂²u/∂x² + ∂²u/∂y²) = 0
        (u·∇)v = -∇p + (1/Re)∇²v  →  u*∂v/∂x + v*∂v/∂y + ∂p/∂y - (1/Re)(∂²v/∂x² + ∂²v/∂y²) = 0
        ∇·u = ∂u/∂x + ∂v/∂y = 0
        """
        u = uvp[:, 0:1]  # velocidad en x
        v = uvp[:, 1:2]  # velocidad en y
        p = uvp[:, 2:3]  # presión
        
        Re = self.Re
        
        # Derivadas de primer orden usando autodiferenciación
        diff_u = torch.autograd.grad(u, X, create_graph=True, grad_outputs=torch.ones_like(u))[0]
        u_x = diff_u[:, 0:1]
        u_y = diff_u[:, 1:2]
        
        diff_v = torch.autograd.grad(v, X, create_graph=True, grad_outputs=torch.ones_like(v))[0]
        v_x = diff_v[:, 0:1]
        v_y = diff_v[:, 1:2]
        
        diff_p = torch.autograd.grad(p, X, create_graph=True, grad_outputs=torch.ones_like(p))[0]
        p_x = diff_p[:, 0:1]
        p_y = diff_p[:, 1:2]
        
        # Derivadas de segundo orden
        u_xx = torch.autograd.grad(u_x, X, create_graph=True, grad_outputs=torch.ones_like(u_x))[0][:, 0:1]
        u_yy = torch.autograd.grad(u_y, X, create_graph=True, grad_outputs=torch.ones_like(u_y))[0][:, 1:2]
        
        v_xx = torch.autograd.grad(v_x, X, create_graph=True, grad_outputs=torch.ones_like(v_x))[0][:, 0:1]
        v_yy = torch.autograd.grad(v_y, X, create_graph=True, grad_outputs=torch.ones_like(v_y))[0][:, 1:2]
        
        # Residuos de las ecuaciones de momentum
        residue_u = u * u_x + v * u_y + p_x - (1/Re) * (u_xx + u_yy)
        residue_v = u * v_x + v * v_y + p_y - (1/Re) * (v_xx + v_yy)
        
        # Residuo de continuidad
        residue_continuity = u_x + v_y
        
        return residue_u, residue_v, residue_continuity
    
    def compute_residual_norm(self, X):
        uvp = self.forward(X)
        r_u, r_v, _ = self.navierstokesResidue(X, uvp)
        residual_norm = (r_u ** 2 + r_v ** 2).squeeze()
        return residual_norm

# %% [markdown]
# #### Configuración del dominio y condiciones de borde

# # %%
# # Parámetros del problema
# Re = 100.0  # Número de Reynolds
# xi, xf = 0.0, 1.0  # Límites en x
# yi, yf = 0.0, 1.0  # Límites en y

# # Límites del dominio
# lb = torch.tensor([xi, yi], device=device)  # Lower bounds
# ub = torch.tensor([xf, yf], device=device)  # Upper bounds

# # Condiciones de borde
# u_wall = 0.0    # Velocidad en las paredes (no-slip)
# v_wall = 0.0    # Velocidad normal en las paredes
# u_lid = 1.0     # Velocidad de la tapa superior
# v_lid = 0.0     # Velocidad normal en la tapa

# print(f"Número de Reynolds: {Re}")
# print(f"Dominio: [{xi}, {xf}] x [{yi}, {yf}]")
# print(f"Condiciones de borde:")
# print(f"  - Paredes laterales e inferior: u=v=0 (no-slip)")
# print(f"  - Tapa superior: u={u_lid}, v={v_lid}")

# %% [markdown]
# #### Carga de datos de ground-truth

# %%
# Cargar datos de los .mat
pressure_mat = loadmat('Re-100/pressure.mat')
velocity_mat = loadmat('Re-100/velocity.mat')

x = pressure_mat['x'].squeeze()
y = pressure_mat['y'].squeeze()
p = pressure_mat['p'].squeeze()
u = velocity_mat['u'].squeeze()
v = velocity_mat['v'].squeeze()

# Reconstruir grilla regular
x_unique = np.linspace(x.min(), x.max(), 201)
y_unique = np.linspace(y.min(), y.max(), 201)
X_grid, Y_grid = np.meshgrid(x_unique, y_unique)

# Interpolar campos sobre la grilla
U_grid = griddata((x, y), u, (X_grid, Y_grid), method='cubic')
V_grid = griddata((x, y), v, (X_grid, Y_grid), method='cubic')
P_grid = griddata((x, y), p, (X_grid, Y_grid), method='cubic')

# Preparar puntos para evaluación del modelo
X_eval = torch.tensor(np.stack([X_grid.flatten(), Y_grid.flatten()], axis=1), dtype=torch.float32, device=device)

# %%
# fig, axs = plt.subplots(1, 3, figsize=(18, 5))

# # Campo u
# im0 = axs[0].imshow(U_grid, extent=[x.min(), x.max(), y.min(), y.max()],
#                     origin='lower', cmap='RdBu_r', aspect='equal')
# axs[0].set_title("Velocidad u - Interpolada")
# axs[0].set_xlabel("x")
# axs[0].set_ylabel("y")
# plt.colorbar(im0, ax=axs[0])

# # Campo v
# im1 = axs[1].imshow(V_grid, extent=[x.min(), x.max(), y.min(), y.max()],
#                     origin='lower', cmap='RdBu_r', aspect='equal')
# axs[1].set_title("Velocidad v - Interpolada")
# axs[1].set_xlabel("x")
# axs[1].set_ylabel("y")
# plt.colorbar(im1, ax=axs[1])

# # Campo p
# im2 = axs[2].imshow(P_grid, extent=[x.min(), x.max(), y.min(), y.max()],
#                     origin='lower', cmap='Spectral', aspect='equal')
# axs[2].set_title("Presión p - Interpolada")
# axs[2].set_xlabel("x")
# axs[2].set_ylabel("y")
# plt.colorbar(im2, ax=axs[2])

# plt.tight_layout()
#plt.show()

# %% [markdown]
# ### 1. Estrategias de muestreo de puntos de colocación
# 
# *Para el modelo PINN desarrollado en el TP N°2, implementar las siguientes estrategias de muestreo para la construcción de los subconjuntos de puntos de colocación: muestreo aleatorio uniforme, muestreo por el método de hipercubo latino (LHS), y por muestreo adaptativo basado en residuos (estrategia RAD). En este último caso, utilice los residuos de las dos ecuaciones de balance de cantidad de movimiento como métrica para llevar a cabo la estrategia.*

# %% [markdown]
# #### (a) Muestreo aleatorio uniforme

# %%
def uniform_random_sampling(N_pde, N_bc, device):
    """
    Genera puntos de colocación para PINNs usando muestreo aleatorio uniforme.
    - Puntos interiores (PDE)
    - Puntos de borde (BC): bottom, top, left, right
    """
    # PDE points en el dominio [0,1] x [0,1]
    pde_points = torch.rand(N_pde, 2, device=device)

    # Puntos de borde
    N_each = N_bc // 4
    assert N_bc % 4 == 0, "N_bc debe ser divisible por 4 para distribuirlo en las 4 fronteras."
    rand = torch.rand(N_each, device=device)

    bottom = torch.stack([rand, torch.zeros_like(rand)], dim=1)
    top    = torch.stack([rand, torch.ones_like(rand)], dim=1)
    left   = torch.stack([torch.zeros_like(rand), rand], dim=1)
    right  = torch.stack([torch.ones_like(rand), rand], dim=1)

    bc_points = torch.cat([bottom, top, left, right], dim=0)

    return pde_points, bottom, top, left, right, bc_points

# %% [markdown]
# #### (b) Muestreo por Hipercubo Latino (LHS)

# %%
def lhs_1d(N, device):
    """
    Realiza muestreo por hipercubo latino en 1D.
    Divide el intervalo [0,1] en N subintervalos y toma un punto aleatorio en cada uno,
    luego los mezcla aleatoriamente.
    """
    intervals = torch.linspace(0, 1, N + 1, device=device)
    lower_bounds = intervals[:-1]
    upper_bounds = intervals[1:]
    points = lower_bounds + (upper_bounds - lower_bounds) * torch.rand(N, device=device)
    return points[torch.randperm(N)]

def latin_hypercube_sampling(N_pde, N_bc, device):
    """
    Genera puntos de colocación para PINNs usando muestreo por hipercubo latino (LHS).
    - Puntos interiores (PDE)
    - Puntos de borde (BC): bottom, top, left, right
    """
    x_pde = lhs_1d(N_pde, device)
    y_pde = lhs_1d(N_pde, device)
    pde_points = torch.stack([x_pde, y_pde], dim=1)

    N_each = N_bc // 4
    rand = lhs_1d(N_each, device)

    bottom = torch.stack([rand, torch.zeros_like(rand)], dim=1)
    top    = torch.stack([rand, torch.ones_like(rand)], dim=1)
    left   = torch.stack([torch.zeros_like(rand), rand], dim=1)
    right  = torch.stack([torch.ones_like(rand), rand], dim=1)

    bc_points = torch.cat([bottom, top, left, right], dim=0)

    return pde_points, bottom, top, left, right, bc_points

# %% [markdown]
# #### (c) Muestreo adaptativo basado en residuos (RAD)

# %%
def residual_adaptive_sampling(model, N_pde, device, grid_N=None):
    """
    Genera puntos de colocación para PINNs usando muestreo adaptativo basado en residuos.
    Se asegura que haya suficientes puntos candidatos para seleccionar.
    """
    if grid_N is None:
        grid_N = max(2 * N_pde, 10000)  # mínimo 2*N_pde para garantizar topk

    test_points = torch.rand(grid_N, 2, device=device).requires_grad_(True)
    residual_norm = model.compute_residual_norm(test_points)

    # Asegurar que haya suficientes puntos para seleccionar
    if residual_norm.shape[0] < N_pde:
        raise ValueError(f"Intentando seleccionar {N_pde} puntos pero sólo hay {residual_norm.shape[0]} disponibles.")

    topk = torch.topk(residual_norm, N_pde)
    selected_points = test_points[topk.indices].detach()

    return selected_points

# %% [markdown]
# ### 2. Para cada estrategia, preparar tres datasets con las siguientes cantidades de puntos.
# 
# - *Primer dataset con Npde = 1000 y Nbc = 100.*
# - *Segundo dataset con Npde = 10000 y Nbc = 1000.*
# - *Tercer dataset con Npde = 100000 y Nbc = 10000.*
# 
# Para resolver el punto 2, se implementa una función `generate_collocation_points()` que permite generar datasets de entrenamiento para las tres estrategias de muestreo requeridas: uniforme, hipercubo latino (LHS) y adaptativo por residuos (RAD), variando la cantidad de puntos de colocación según el enunciado. Se generan los datasets en loop dentro del entrenamiento mas adelante.

# %%
def generate_collocation_points(strategy, N_pde, N_bc, device, model=None):
    """
    Genera puntos de colocación para entrenamiento PINN según la estrategia indicada.
    
    Parámetros:
    - strategy: "uniform", "lhs" o "rad"
    - N_pde: cantidad de puntos interiores
    - N_bc: cantidad de puntos de borde
    - device: 'cpu' o 'cuda'
    - model: requerido solo si strategy="rad"

    Retorna:
    - pde_points, bottom, top, left, right, bc_points
    """
    if strategy == "uniform":
        return uniform_random_sampling(N_pde, N_bc, device)

    elif strategy == "lhs":
        return latin_hypercube_sampling(N_pde, N_bc, device)

    elif strategy == "rad":
        assert model is not None, "Se requiere un modelo entrenado para RAD."
        pde_points = residual_adaptive_sampling(model, N_pde, device)
        _, bottom, top, left, right, bc_points = uniform_random_sampling(0, N_bc, device)
        return pde_points, bottom, top, left, right, bc_points

    else:
        raise ValueError(f"Estrategia de muestreo desconocida: {strategy}")

# %%
# strategies = ["uniform", "lhs", "rad"]
# point_configs = [
#     {"Npde": 1000, "Nbc": 100},
#     {"Npde": 10000, "Nbc": 1000},
#     {"Npde": 100000, "Nbc": 10000},
# ]

# %% [markdown]
# ### 3. Entrenar el modelo PINN para las nueve configuraciones de datasets preparados en el item 2. 
# 
# *Con el fin de realizar una comparación “justa”, en todos los casos deberá utilizar la misma configuración de hiperparámetros (tamaño de red, optimizador, cantidad de epochs, etc.), y defina el criterio de selección de dichos valores. Utilice un valor 𝜆bc = 10 para el peso de la componente de la función de pérdida asociada al residuo de las condiciones de borde de velocidad. Calcular en cada caso la norma-2 del error de la misma manera que se realizó en el item 3 del TP N°2*

# %% [markdown]
# #### Configuración del modelo

# %%
# Parámetros del modelo: mantenemos igual a TP2
# model_parameters = {
#     "Device": device,
#     "LowerBounds": lb.to(device),
#     "UpperBounds": ub.to(device),
#     "Re": Re,
#     "InputDimensions": 2,      # (x, y)
#     "OutputDimensions": 3,     # (u, v, p)
#     "NumberOfNeurons": 64,    # Era 200 - REDUCIDO por CPU
#     "NumberOfHiddenLayers": 5, # Era 12 - REDUCIDO por CPU
#     "ActivationFunction": nn.Tanh()
# }

# %%
# Crear modelo
# torch.manual_seed(10)
# model = PINN_Module(model_parameters).to(device)

# print("Arquitectura del modelo:")
# print(f"  - Entradas: {model_parameters['InputDimensions']} (x, y)")
# print(f"  - Salidas: {model_parameters['OutputDimensions']} (u, v, p)")
# print(f"  - Capas ocultas: {model_parameters['NumberOfHiddenLayers']}")
# print(f"  - Neuronas por capa: {model_parameters['NumberOfNeurons']}")
# print(f"  - Función de activación: {model_parameters['ActivationFunction']}")

# # Contar parámetros
# total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print(f"  - Total de parámetros entrenables: {total_params}")

# %% [markdown]
# ### Entrenamiento

# %%
def compute_bc_loss(uvp, u_target, v_target, loss_fn):
    loss_u = loss_fn(uvp[:, 0:1], u_target)
    loss_v = loss_fn(uvp[:, 1:2], v_target)
    return loss_u + loss_v

# Función para calcular pérdida de presión de referencia
def compute_pressure_reference_loss(model, loss_fn):
    """
    Calcula la pérdida para forzar que p(0,0) = 0
    Esto elimina la indeterminación de la constante en el campo de presión
    """
    # Punto de referencia (0,0)
    reference_point = torch.tensor([[0.0, 0.0]], device=device, requires_grad=True)
    
    # Evaluar el modelo en el punto de referencia
    uvp_ref = model(reference_point)
    p_ref = uvp_ref[:, 2:3]  # Extraer presión
    
    # La presión en (0,0) debe ser cero
    target_pressure = torch.zeros_like(p_ref)
    
    return loss_fn(p_ref, target_pressure)

# Función de entrenamiento
def train_pinn(model, pde_points, top, bottom, left, right, 
               epochs, optimizer, scheduler,
               weight_pde, lambda_bc, weight_pressure_ref,
               strategy=None, Npde=None, Nbc=None,
               loss_fn=nn.MSELoss()):
    """
    Entrena un modelo PINN para el problema de la cavidad cuadrada.
    Guarda el modelo con nombre basado en la estrategia y el tamaño del dataset.
    """
    # Historial de pérdidas
    loss_train = []
    loss_train_momentum_u = []
    loss_train_momentum_v = []
    loss_train_continuity = []
    loss_train_bc = []
    loss_train_pressure_ref = []

    print("Iniciando entrenamiento ...")
    t0 = datetime.now()

    for epoch in trange(epochs, desc="Entrenando modelo PINN"):
        model.train()
        optimizer.zero_grad()

        # === PDE ===
        pde_points_epoch = pde_points.detach().clone().requires_grad_(True)
        uvp_pde = model(pde_points_epoch)
        res_u, res_v, res_cont = model.navierstokesResidue(pde_points_epoch, uvp_pde)

        # Pérdidas PDE
        loss_u = loss_fn(res_u, torch.zeros_like(res_u))
        loss_v = loss_fn(res_v, torch.zeros_like(res_v))
        loss_cont = loss_fn(res_cont, torch.zeros_like(res_cont))
        loss_pde_total = loss_u + loss_v + loss_cont

        # === BC ===
        uvp_top = model(top)
        uvp_bottom = model(bottom)
        uvp_left = model(left)
        uvp_right = model(right)

        loss_bc_top = compute_bc_loss(uvp_top,
                                      torch.ones_like(uvp_top[:, 0:1]),
                                      torch.zeros_like(uvp_top[:, 1:2]),
                                      loss_fn)
        loss_bc_bottom = compute_bc_loss(uvp_bottom,
                                         torch.zeros_like(uvp_bottom[:, 0:1]),
                                         torch.zeros_like(uvp_bottom[:, 1:2]),
                                         loss_fn)
        loss_bc_left = compute_bc_loss(uvp_left,
                                       torch.zeros_like(uvp_left[:, 0:1]),
                                       torch.zeros_like(uvp_left[:, 1:2]),
                                       loss_fn)
        loss_bc_right = compute_bc_loss(uvp_right,
                                        torch.zeros_like(uvp_right[:, 0:1]),
                                        torch.zeros_like(uvp_right[:, 1:2]),
                                        loss_fn)
        loss_bc_total = loss_bc_top + loss_bc_bottom + loss_bc_left + loss_bc_right

        # === Presión de referencia ===
        loss_pressure_ref = compute_pressure_reference_loss(model, loss_fn)

        # === Pérdida total ===
        loss_total = (weight_pde * loss_pde_total +
                      lambda_bc * loss_bc_total +
                      weight_pressure_ref * loss_pressure_ref)

        loss_total.backward()
        optimizer.step()
        scheduler.step()

        # Guardar pérdidas
        loss_train.append(loss_total.item())
        loss_train_momentum_u.append(loss_u.item())
        loss_train_momentum_v.append(loss_v.item())
        loss_train_continuity.append(loss_cont.item())
        loss_train_bc.append(loss_bc_total.item())
        loss_train_pressure_ref.append(loss_pressure_ref.item())

        # Logs
        if epoch % 500 == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                u_top_mean = model(top)[:, 0].mean().item()
                ref_point = torch.tensor([[0.0, 0.0]], device=model.Device)
                p_ref_value = model(ref_point)[0, 2].item()

            print(f"Epoch {epoch:4d} | Total: {loss_total.item():.2e} | PDE: {loss_pde_total.item():.2e} | "
                  f"BC: {loss_bc_total.item():.2e} | P_ref: {loss_pressure_ref.item():.2e} | "
                  f"p(0,0): {p_ref_value:.4f} | u_top: {u_top_mean:.3f}")

    elapsed = datetime.now() - t0
    print(f"\nTiempo total: {elapsed.total_seconds():.1f} segundos")

    # Guardar modelo con nombre descriptivo
    models_dir = "models"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    time_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"pinn_{strategy}_Npde{Npde}_Nbc{Nbc}_{time_str}.pt" if strategy and Npde and Nbc else f"pinn_trained_{time_str}.pt"
    torch.save(model.state_dict(), os.path.join(models_dir, filename))
    print(f"Modelo guardado como: models/{filename}")

    return {
        "total": loss_train,
        "u_momentum": loss_train_momentum_u,
        "v_momentum": loss_train_momentum_v,
        "continuity": loss_train_continuity,
        "bc": loss_train_bc,
        "pressure_ref": loss_train_pressure_ref
    }


# %% [markdown]
# #### Gráficos de evolución de pérdidas

# %%
# Función para graficar pérdidas con nueva pérdida
def plot_losses(loss_dict):
    loss_list = [
        loss_dict["total"],
        loss_dict["u_momentum"],
        loss_dict["v_momentum"],
        loss_dict["continuity"],
        loss_dict["bc"],
        loss_dict["pressure_ref"]  # NUEVO
    ]
    colors = ['black', 'red', 'green', 'blue', 'magenta', 'orange']  # NUEVO color
    titles = ['Overall Loss', 'Momentum U', 'Momentum V', 'Continuity', 'BC', 'Pressure Ref']  # NUEVO título

    _, ax = plt.subplots(1, len(loss_list), figsize=(25, 4))
    for i, loss in enumerate(loss_list):
        ax[i].loglog(np.arange(len(loss)), loss, color=colors[i])
        ax[i].set_xlabel("Epoch")
        ax[i].set_title(titles[i])
        ax[i].grid(True)
    plt.tight_layout()
    plt.show()

# %%
def plot_comparacion_uvp(u_pred, v_pred, p_pred, strategy, Npde, Nbc,
                          U_grid, V_grid, P_grid, x, y):
    """
    Plotea comparación de u, v, p entre solución de referencia (MEF) y predicción (PINN),
    incluyendo errores absolutos. 
    
    MODIFICACIÓN: Escalas de colores sincronizadas para comparación visual precisa.
    
    Parámetros:
    - u_pred, v_pred, p_pred: resultados del modelo PINN (2D arrays)
    - strategy: nombre de la estrategia ('uniform', 'lhs', 'rad')
    - Npde, Nbc: tamaños del dataset
    - U_grid, V_grid, P_grid: campos de referencia (MEF)
    - x, y: arrays con coordenadas para los ejes
    """
    fig, axs = plt.subplots(3, 3, figsize=(18, 13))
    
    # Extensión para todas las gráficas
    extent = [x.min(), x.max(), y.min(), y.max()]
    
    # --- Velocidad u ---
    im0 = axs[0, 0].imshow(U_grid, extent=extent,
                           origin='lower', cmap='RdBu_r', aspect='equal')
    axs[0, 0].set_title(r"$u_{MEF}$")
    axs[0, 0].set_ylabel("y")
    plt.colorbar(im0, ax=axs[0, 0])
    
    # PREDICCIÓN PINN u - ESCALA SINCRONIZADA
    im1 = axs[0, 1].imshow(u_pred, extent=extent,
                           origin='lower', cmap='RdBu_r', aspect='equal',
                           vmin=U_grid.min(), vmax=U_grid.max())  # ← SINCRONIZACIÓN
    axs[0, 1].set_title(r"$u_{PINN}$")
    plt.colorbar(im1, ax=axs[0, 1])
    
    im2 = axs[0, 2].imshow(np.abs(u_pred - U_grid), extent=extent,
                           origin='lower', cmap='Reds', aspect='equal')
    axs[0, 2].set_title(r"$|u_{PINN} - u_{MEF}|$")
    plt.colorbar(im2, ax=axs[0, 2])
    
    # --- Velocidad v ---
    im3 = axs[1, 0].imshow(V_grid, extent=extent,
                           origin='lower', cmap='RdBu_r', aspect='equal')
    axs[1, 0].set_title(r"$v_{MEF}$")
    axs[1, 0].set_ylabel("y")
    plt.colorbar(im3, ax=axs[1, 0])
    
    # PREDICCIÓN PINN v - ESCALA SINCRONIZADA
    im4 = axs[1, 1].imshow(v_pred, extent=extent,
                           origin='lower', cmap='RdBu_r', aspect='equal',
                           vmin=V_grid.min(), vmax=V_grid.max())  # ← SINCRONIZACIÓN
    axs[1, 1].set_title(r"$v_{PINN}$")
    plt.colorbar(im4, ax=axs[1, 1])
    
    im5 = axs[1, 2].imshow(np.abs(v_pred - V_grid), extent=extent,
                           origin='lower', cmap='Reds', aspect='equal')
    axs[1, 2].set_title(r"$|v_{PINN} - v_{MEF}|$")
    plt.colorbar(im5, ax=axs[1, 2])
    
    # --- Presión p ---
    im6 = axs[2, 0].imshow(P_grid, extent=extent,
                           origin='lower', cmap='Spectral', aspect='equal')
    axs[2, 0].set_title(r"$p_{MEF}$")
    axs[2, 0].set_ylabel("y")
    axs[2, 0].set_xlabel("x")
    plt.colorbar(im6, ax=axs[2, 0])
    
    # PREDICCIÓN PINN p - ESCALA SINCRONIZADA
    im7 = axs[2, 1].imshow(p_pred, extent=extent,
                           origin='lower', cmap='Spectral', aspect='equal',
                           vmin=P_grid.min(), vmax=P_grid.max())  # ← SINCRONIZACIÓN
    axs[2, 1].set_title(r"$p_{PINN}$")
    axs[2, 1].set_xlabel("x")
    plt.colorbar(im7, ax=axs[2, 1])
    
    im8 = axs[2, 2].imshow(np.abs(p_pred - P_grid), extent=extent,
                           origin='lower', cmap='Reds', aspect='equal')
    axs[2, 2].set_title(r"$|p_{PINN} - p_{MEF}|$")
    axs[2, 2].set_xlabel("x")
    plt.colorbar(im8, ax=axs[2, 2])
    
    plt.suptitle(f"Comparación de campos: {strategy.upper()} - Npde={Npde}, Nbc={Nbc}", 
                 fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # espacio para el título
    plt.show()

# # %%
# results = {}

# error_metrics = {
#     "strategy": [],
#     "Npde": [],
#     "Nbc": [],
#     "error_u": [],
#     "error_v": [],
#     "error_p": [],
# }

# for strategy in strategies:
#     for config in point_configs:
#         Npde = config["Npde"]
#         Nbc = config["Nbc"]
#         key = f"{strategy}_Npde{Npde}_Nbc{Nbc}"

#         print(f"\n=== Entrenando modelo: {key} ===")
        
#         # Entrenar primero un modelo base para RAD
#         if strategy == "rad" and f"uniform_Npde1000_Nbc100" not in results:
#             print("Esperando modelo base uniform para usar en RAD...")
#             continue  # saltar por ahora

#         # Usar modelo base entrenado para RAD
#         base_key = f"uniform_Npde{Npde}_Nbc{Nbc}"
#         base_model = results.get(base_key, {}).get("model") if strategy == "rad" else None

#         # Generar datos
#         pde_points, bottom, top, left, right, bc_points = generate_collocation_points(
#             strategy=strategy,
#             N_pde=Npde,
#             N_bc=Nbc,
#             device=device,
#             model=base_model
#         )

#         # Crear nuevo modelo
#         torch.manual_seed(10)
#         model = PINN_Module(model_parameters).to(device)

#         # Optim y scheduler
#         optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
#         scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9995)

#         # Entrenamiento
#         losses = train_pinn(
#             model, pde_points, top, bottom, left, right,
#             epochs=1,#0000, #TODO: change to 10k! just testing code functioning
#             optimizer=optimizer,
#             scheduler=scheduler,
#             weight_pde=100.0,
#             lambda_bc=10.0,                     # requerido por TP3
#             weight_pressure_ref=10000.0,       # alto peso para condición de referencia
#             strategy=strategy,
#             Npde=Npde,
#             Nbc=Nbc
#         )

#         # Guardar resultados
#         results[key] = {
#             "model": model,
#             "losses": losses,
#         }

#         # Graficar pérdidas
#         print(f"Plotting training losses for {key}")
#         plot_losses(losses)

#         # Evaluación de errores en malla de referencia 
#         model.eval()
#         with torch.no_grad():
#             uvp_pred = model(X_eval)
#             u_pred = uvp_pred[:, 0].cpu().numpy().reshape(U_grid.shape)
#             v_pred = uvp_pred[:, 1].cpu().numpy().reshape(V_grid.shape)
#             p_pred = uvp_pred[:, 2].cpu().numpy().reshape(P_grid.shape)

#         # Errores absolutos
#         error_u = np.abs(u_pred - U_grid)
#         error_v = np.abs(v_pred - V_grid)
#         error_p = np.abs(p_pred - P_grid)

#         # Normas-2 absolutas
#         norm2_u = np.linalg.norm(error_u)
#         norm2_v = np.linalg.norm(error_v)
#         norm2_p = np.linalg.norm(error_p)

#         plot_comparacion_uvp(
#             u_pred, v_pred, p_pred,
#             strategy, Npde, Nbc,
#             U_grid, V_grid, P_grid,
#             x, y
#         )

#         # Guardar en lista
#         error_metrics["strategy"].append(strategy)
#         error_metrics["Npde"].append(Npde)
#         error_metrics["Nbc"].append(Nbc)
#         error_metrics["error_u"].append(norm2_u)
#         error_metrics["error_v"].append(norm2_v)
#         error_metrics["error_p"].append(norm2_p)

# # %% [markdown]
# # ### 4. Convergencia en cantidad de puntos de colocación. 
# # 
# # *Grafique los valores previamente calculados de la norma-2 del error en función del tamaño del dataset, en una gráfica doble logarítmica (loglog). ¿Observa un patrón de convergencia? ¿Es posible reportar una tasa de convergencia estable? ¿Cuál de todas las configuraciones alcanzó el mejor desempeño? ¿Es consistente esta conclusión con lo que usted esperaba? ¿Porqué si o porqué no?*

# # %%
# df_errors = pd.DataFrame(error_metrics)
# df_errors.to_csv("error_metrics.csv", index=False)  # opcional
# display(df_errors)

# # %%
# variables = ["error_u", "error_v", "error_p"]
# labels = ["u", "v", "p"]

# for var, label in zip(variables, labels):
#     plt.figure(figsize=(6, 4))
#     for strategy in df_errors["strategy"].unique():
#         subset = df_errors[df_errors["strategy"] == strategy]
#         subset = subset.sort_values("Npde")
#         plt.loglog(subset["Npde"], subset[var], marker='o', label=strategy)

#     plt.xlabel("Npde (log)")
#     plt.ylabel(f"Error L2 de {label} (log)")
#     plt.title(f"Convergencia de error L2 - {label}")
#     plt.grid(True, which='both')
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(f"./graficas/convergencia_loglog_{label}.png")
#     plt.show()

# # %% [markdown]
# # ### Analizando las gráficas de convergencia y el dataframe de errores, podemos decir que:
# # 
# # 1. ¿Observa un patrón de convergencia?
# # 
# #     Si, pero depende significativamente según la estrategia de muestreo. Por ejemplo:
# #       - Estrategia "RAD": color verde: muestra un patrón de convergencia claro en las tres variables (p, u y v). El error disminuye consistentemente al aumentar el número de nodos, en especial en la variable p donde el error baja de -40 a -19.
# #       - Estrategia "Uniforme" (color azul) y "LHS" (color naranja): muestran patrones de convergencia muy débiles o nulos. Sus errores se mantienen relativamente constantes a tráves del rango de Npde, con ligeras variaciones que no indican convergencia sistemática.
# # 
# # 2. ¿Es posible reportar una tasa de convergencia estable?
# #     Solo para la estrategia "RAD". En gráficas loglog, la tasa de convergencia se estima por la pendiente de la línea.
# #       - Para "RAD": Se puede estimar una tasa de convergencia aproximada de $$O(N^{-0.3}) \text{ a } O(N^{-0.4})$$ basándose en las pendientes observadas.
# #       - Para "Uniforme" y "LHS": No es posible reportar tasas de convergencia estables debido a la ausencia de tendencias decrecientes consistentes.
# # 
# # 3. ¿Cuál de todas las configuraciones alcanzó el mejor desempeño?
# #     La estrategia "LHS" (Latin Hypercube Sampling) logró el mejor desempeño final en Npde=100_000:
# #       - Variable p: LHS (19.16) < Uniforme (19.21) < RAD (19.41)
# #       - Variable u: LHS (26.23) < Uniforme (26.89) < RAD (28.82)
# #       - Variable v: LHS (27.59) < Uniforme (27.65) < RAD (28.63)
# # 
# #     LHS supera consistentemente a las otras estrategias en las tres variables.
# # 
# # 4. ¿Es consistente esta conclusión con lo que usted esperaba?
# # 
# #     Es parcialmente consistente, con algunos aspectos a tener en cuenta:
# #     Consistente:
# #       - El aspecto de que LHS supere a la Uniforme es esperado, ya que LHS es una técnica de muestreo más sofisticada que garantiza mejor cobertura del espacio de muestreo.
# # 
# #     Sorprende:
# #       - "RAD" muestre convergencia clara pero termine con errores ligeramente mayores.
# #       - La ausencia de convergencia en uniforme y LHS es inesperada para problemas numéricos típicos.
# # 
# # 5. ¿Por qué sí o por qué no?
# # Las posibles razones que explican los resultados:
# # 
# #     - LHS vs Uniforme: LHS proporciona una distribución más uniforme en el espacio de parámetros, reduciendo la varianza del estimador y logrando mejor precisión con el mismo número de puntos.
# #     - Comportamiento de "RAD": La convergencia clara sugiere que esta estrategia es más sensible al número de puntos pero requiere más nodos para alcanzar su potencial. Posiblemente sea una estrategia adaptativa que mejora su distribución con más datos.
# #     - Falta de convergencia en uniforme/LHS: Esto podría indicar que:
# #         * El problema ha alcanzado una saturación de precisión limitada por otros factores (discretización espacial, condiciones de contorno, etc.)
# #         * El rango de Npde evaluado no es suficiente para observar convergencia
# #         * Existe ruido numérico que enmascara la convergencia

# # %% [markdown]
# # 


