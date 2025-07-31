#!/usr/bin/env python3
"""
optuna.py - Módulo de optimización de hiperparámetros para PINNs usando Optuna

Este módulo contiene todas las funciones necesarias para optimizar hiperparámetros
de redes neuronales informadas por física (PINNs) usando la librería Optuna.

Autor: Generado para TP3 - Redes Neuronales Informadas por Física
"""

import os
import torch
import torch.nn as nn
import optuna
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import pickle
import gc
from typing import Dict, Tuple, List, Optional, Any
import warnings
import traceback

# Suprimir warnings de Optuna para output más limpio
optuna.logging.set_verbosity(optuna.logging.WARNING)


class OptunaOptimizer:
    """
    Clase principal para optimización de hiperparámetros de PINNs usando Optuna.
    """
    
    def __init__(self, 
                 base_model_params: Dict[str, Any],
                 device: torch.device,
                 validation_data: Tuple[torch.Tensor, np.ndarray, np.ndarray, np.ndarray],
                 results_dir: str = "optuna_results"):
        """
        Inicializa el optimizador Optuna.
        
        Args:
            base_model_params: Parámetros base del modelo PINN
            device: Dispositivo de computación (CPU/GPU)
            validation_data: Tupla (X_eval, U_grid, V_grid, P_grid) para validación
            results_dir: Directorio donde guardar resultados
        """
        self.base_model_params = base_model_params
        self.device = device
        self.validation_data = validation_data
        self.results_dir = results_dir
        
        # Crear directorio de resultados
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Mapeo de funciones de activación
        self.activation_functions = {
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'swish': nn.SiLU(),
            'elu': nn.ELU()
        }
        
        print(f"OptunaOptimizer inicializado")
        print(f"Dispositivo: {device}")
        print(f"Directorio de resultados: {results_dir}")
    
    def suggest_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Define el espacio de búsqueda de hiperparámetros.
        
        Args:
            trial: Trial de Optuna
            
        Returns:
            Diccionario con hiperparámetros sugeridos
        """
        hyperparams = {
            # Arquitectura de la red - RANGOS MÁS ESTRECHOS
            'n_neurons': trial.suggest_int('n_neurons', 48, 96, step=16),      # Era 32-128
            'n_hidden_layers': trial.suggest_int('n_hidden_layers', 4, 6),     # Era 3-8
            'activation': trial.suggest_categorical('activation', 
                                                 ['tanh', 'gelu']),            # Solo las mejores
            
            # Optimización - MÁS ENFOCADO
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 5e-3, log=True),  # Rango más estrecho
            'weight_decay': trial.suggest_float('weight_decay', 1e-7, 1e-5, log=True),    # Rango más estrecho
            'scheduler_gamma': trial.suggest_float('scheduler_gamma', 0.9995, 0.9998),    # Rango más estrecho
            
            # Pesos de pérdidas - ENFOCADOS
            'weight_pde': trial.suggest_float('weight_pde', 25.0, 100.0),                 # Alrededor del valor óptimo encontrado
            'weight_pressure_ref': trial.suggest_float('weight_pressure_ref', 15000.0, 30000.0),  # Alrededor del valor óptimo
            'lambda_bc': 10.0,  # ✅ FIJO - pero incluido en hyperparams
            
            # Entrenamiento
            'epochs': trial.suggest_int('epochs', 5000, 8000, step=500),       # Rango más estrecho
            
            # Datos rotulados (opcional)
            'weight_data': 0.0  # Deshabilitado por ahora
        }
        
        return hyperparams
    
    def create_optimized_model(self, hyperparams: Dict[str, Any]):
        """
        Crea un modelo PINN con hiperparámetros optimizados.
        
        Args:
            hyperparams: Diccionario con hiperparámetros
            
        Returns:
            Modelo PINN configurado
        """
        # Actualizar parámetros del modelo
        optimized_params = self.base_model_params.copy()
        optimized_params.update({
            "NumberOfNeurons": hyperparams['n_neurons'],
            "NumberOfHiddenLayers": hyperparams['n_hidden_layers'],
            "ActivationFunction": self.activation_functions[hyperparams['activation']]
        })
        
        # ✅ IMPORT DENTRO DE LA FUNCIÓN para evitar problemas de importación circular
        try:
            import sys
            import os
            
            # Agregar directorio actual al path si no está
            current_dir = os.path.dirname(os.path.abspath(__file__))
            if current_dir not in sys.path:
                sys.path.insert(0, current_dir)
            
            from tp3 import PINN_Module
            return PINN_Module(optimized_params).to(self.device)
            
        except ImportError as e:
            print(f"Error importando PINN_Module: {e}")
            print("Asegúrate de que TP3.py esté en el mismo directorio")
            raise
    
    def create_optimizer_and_scheduler(self, model: nn.Module, hyperparams: Dict[str, Any]):
        """
        Crea optimizador y scheduler con hiperparámetros optimizados.
        
        Args:
            model: Modelo PyTorch
            hyperparams: Diccionario con hiperparámetros
            
        Returns:
            Tupla (optimizer, scheduler)
        """
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=hyperparams['learning_rate'], 
            weight_decay=hyperparams['weight_decay']
        )
        
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, 
            gamma=hyperparams['scheduler_gamma']
        )
        
        return optimizer, scheduler
    
    def generate_collocation_points(self, strategy: str, N_pde: int, N_bc: int, model=None):
        """
        Genera puntos de colocación según la estrategia especificada.
        
        Args:
            strategy: Estrategia de muestreo ('uniform', 'lhs', 'rad')
            N_pde: Número de puntos interiores
            N_bc: Número de puntos de borde
            model: Modelo (necesario solo para RAD)
            
        Returns:
            Tupla con puntos generados
        """
        try:
            import sys
            import os
            
            # Agregar directorio actual al path si no está
            current_dir = os.path.dirname(os.path.abspath(__file__))
            if current_dir not in sys.path:
                sys.path.insert(0, current_dir)
            
            from tp3 import generate_collocation_points
            return generate_collocation_points(strategy, N_pde, N_bc, self.device, model)
            
        except ImportError as e:
            print(f"Error importando generate_collocation_points: {e}")
            raise
    
    def compute_bc_loss(self, uvp, u_target, v_target, loss_fn):
        """Calcula pérdida de condiciones de borde."""
        loss_u = loss_fn(uvp[:, 0:1], u_target)
        loss_v = loss_fn(uvp[:, 1:2], v_target)
        return loss_u + loss_v
    
    def compute_pressure_reference_loss(self, model, loss_fn):
        """Calcula pérdida para forzar p(0,0) = 0."""
        reference_point = torch.tensor([[0.0, 0.0]], device=self.device, requires_grad=True)
        uvp_ref = model(reference_point)
        p_ref = uvp_ref[:, 2:3]
        target_pressure = torch.zeros_like(p_ref)
        return loss_fn(p_ref, target_pressure)
    
    def train_single_trial(self, 
                          model: nn.Module, 
                          hyperparams: Dict[str, Any],
                          strategy: str,
                          N_pde: int,
                          N_bc: int,
                          trial: optuna.Trial = None) -> float:
        """
        Entrena un modelo para un trial específico.
        
        Args:
            model: Modelo PINN
            hyperparams: Hiperparámetros del trial
            strategy: Estrategia de muestreo
            N_pde: Número de puntos PDE
            N_bc: Número de puntos BC
            trial: Trial de Optuna (para pruning)
            
        Returns:
            Error promedio final
        """
        try:
            # Extraer datos de validación
            X_eval, U_grid, V_grid, P_grid = self.validation_data
            
            # Crear optimizador y scheduler
            optimizer, scheduler = self.create_optimizer_and_scheduler(model, hyperparams)
            
            # Generar puntos de colocación
            pde_points, bottom, top, left, right, _ = self.generate_collocation_points(
                strategy, N_pde, N_bc
            )
            
            # Configuración de entrenamiento
            epochs = hyperparams['epochs']
            loss_fn = nn.MSELoss()
            
            # Entrenamiento
            model.train()
            for epoch in range(epochs):
                optimizer.zero_grad()
                
                # === PDE Loss ===
                pde_points_epoch = pde_points.detach().clone().requires_grad_(True)
                uvp_pde = model(pde_points_epoch)
                res_u, res_v, res_cont = model.navierstokesResidue(pde_points_epoch, uvp_pde)
                
                loss_u = loss_fn(res_u, torch.zeros_like(res_u))
                loss_v = loss_fn(res_v, torch.zeros_like(res_v))
                loss_cont = loss_fn(res_cont, torch.zeros_like(res_cont))
                loss_pde_total = loss_u + loss_v + loss_cont
                
                # === BC Loss ===
                uvp_top = model(top)
                uvp_bottom = model(bottom)
                uvp_left = model(left)
                uvp_right = model(right)
                
                loss_bc_top = self.compute_bc_loss(uvp_top,
                                                  torch.ones_like(uvp_top[:, 0:1]),
                                                  torch.zeros_like(uvp_top[:, 1:2]),
                                                  loss_fn)
                loss_bc_bottom = self.compute_bc_loss(uvp_bottom,
                                                     torch.zeros_like(uvp_bottom[:, 0:1]),
                                                     torch.zeros_like(uvp_bottom[:, 1:2]),
                                                     loss_fn)
                loss_bc_left = self.compute_bc_loss(uvp_left,
                                                   torch.zeros_like(uvp_left[:, 0:1]),
                                                   torch.zeros_like(uvp_left[:, 1:2]),
                                                   loss_fn)
                loss_bc_right = self.compute_bc_loss(uvp_right,
                                                    torch.zeros_like(uvp_right[:, 0:1]),
                                                    torch.zeros_like(uvp_right[:, 1:2]),
                                                    loss_fn)
                loss_bc_total = loss_bc_top + loss_bc_bottom + loss_bc_left + loss_bc_right
                
                # === Pressure Reference Loss ===
                loss_pressure_ref = self.compute_pressure_reference_loss(model, loss_fn)
                
                # === Total Loss ===
                loss_total = (hyperparams['weight_pde'] * loss_pde_total +
                             hyperparams['lambda_bc'] * loss_bc_total +
                             hyperparams['weight_pressure_ref'] * loss_pressure_ref)
                
                loss_total.backward()
                optimizer.step()
                scheduler.step()
                
                # Pruning intermedio para acelerar optimización
                if trial and epoch % 500 == 0 and epoch > 0:
                    model.eval()
                    with torch.no_grad():
                        uvp_pred = model(X_eval)
                        u_pred = uvp_pred[:, 0].cpu().numpy().reshape(U_grid.shape)
                        v_pred = uvp_pred[:, 1].cpu().numpy().reshape(V_grid.shape)
                        p_pred = uvp_pred[:, 2].cpu().numpy().reshape(P_grid.shape)
                        
                        error_u = np.linalg.norm(u_pred - U_grid)
                        error_v = np.linalg.norm(v_pred - V_grid)
                        error_p = np.linalg.norm(p_pred - P_grid)
                        avg_error = (error_u + error_v + error_p) / 3
                    
                    model.train()
                    trial.report(avg_error, epoch)
                    
                    if trial.should_prune():
                        raise optuna.exceptions.TrialPruned()
            
            # Evaluación final
            model.eval()
            with torch.no_grad():
                uvp_pred = model(X_eval)
                u_pred = uvp_pred[:, 0].cpu().numpy().reshape(U_grid.shape)
                v_pred = uvp_pred[:, 1].cpu().numpy().reshape(V_grid.shape)
                p_pred = uvp_pred[:, 2].cpu().numpy().reshape(P_grid.shape)
                
                error_u = np.linalg.norm(u_pred - U_grid)
                error_v = np.linalg.norm(v_pred - V_grid)
                error_p = np.linalg.norm(p_pred - P_grid)
                avg_error = (error_u + error_v + error_p) / 3
            
            # Limpiar memoria
            del model, optimizer, scheduler
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            gc.collect()
            
            return avg_error
            
        except optuna.exceptions.TrialPruned:
            # Manejo específico para TrialPruned
            print(f"❌ El trial {trial.number} fue podado porque no mostró un rendimiento prometedor.")
            return float('inf')
        
        except Exception as e:
            print(f"❌ Error en trial {trial.number if trial else 'N/A'}: {str(e)}")
            print(f"   Tipo de error: {type(e).__name__}")
            import traceback
            print(f"   Detalle: {traceback.format_exc()}")
            return float('inf')
    
    def objective_function(self, trial: optuna.Trial, strategy: str, N_pde: int, N_bc: int) -> float:
        """
        Función objetivo para Optuna.
        
        Args:
            trial: Trial de Optuna
            strategy: Estrategia de muestreo
            N_pde: Número de puntos PDE
            N_bc: Número de puntos BC
            
        Returns:
            Valor a minimizar (error promedio)
        """
        # Obtener hiperparámetros sugeridos
        hyperparams = self.suggest_hyperparameters(trial)
        
        # Crear modelo
        model = self.create_optimized_model(hyperparams)
        
        # Entrenar y evaluar
        avg_error = self.train_single_trial(model, hyperparams, strategy, N_pde, N_bc, trial)
        
        return avg_error
    
    def run_optimization(self, 
                        strategy: str,
                        N_pde: int,
                        N_bc: int,
                        n_trials: int = 50,
                        study_name: str = "pinn_optimization") -> optuna.Study:
        """
        Ejecuta la optimización de hiperparámetros.
        
        Args:
            strategy: Estrategia de muestreo
            N_pde: Número de puntos PDE
            N_bc: Número de puntos BC
            n_trials: Número de trials
            study_name: Nombre del estudio
            
        Returns:
            Estudio de Optuna completado
        """
        print(f"=== INICIANDO OPTIMIZACIÓN ===")
        print(f"Estrategia: {strategy}")
        print(f"Puntos PDE: {N_pde}, Puntos BC: {N_bc}")
        print(f"Número de trials: {n_trials}")
        print(f"Nombre del estudio: {study_name}")
        
        # Crear estudio
        study = optuna.create_study(
            direction='minimize',
            study_name=study_name,
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        )
        
        # Función objetivo parcial
        objective_partial = lambda trial: self.objective_function(trial, strategy, N_pde, N_bc)
        
        # Ejecutar optimización
        start_time = datetime.now()
        study.optimize(objective_partial, n_trials=n_trials, timeout=None)
        end_time = datetime.now()
        
        elapsed = end_time - start_time
        print(f"\nOptimización completada en {elapsed}")
        print(f"Trials completados: {len(study.trials)}")
        print(f"Mejor valor: {study.best_value:.6f}")
        
        # Guardar resultados
        self.save_optimization_results(study, strategy, N_pde, N_bc)
        
        return study
    
    def save_optimization_results(self, 
                                 study: optuna.Study, 
                                 strategy: str, 
                                 N_pde: int, 
                                 N_bc: int):
        """
        Guarda los resultados de optimización.
        
        Args:
            study: Estudio completado
            strategy: Estrategia utilizada
            N_pde: Número de puntos PDE
            N_bc: Número de puntos BC
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_name = f"{strategy}_Npde{N_pde}_Nbc{N_bc}_{timestamp}"
        
        # Guardar estudio completo
        study_file = os.path.join(self.results_dir, f"study_{base_name}.pkl")
        with open(study_file, 'wb') as f:
            pickle.dump(study, f)
        
        # Guardar mejores parámetros
        params_file = os.path.join(self.results_dir, f"best_params_{base_name}.txt")
        with open(params_file, 'w') as f:
            f.write(f"Optimización - {strategy.upper()}\n")
            f.write(f"Configuración: Npde={N_pde}, Nbc={N_bc}\n")
            f.write(f"Mejor valor objetivo: {study.best_value:.6f}\n")
            f.write(f"Número de trials: {len(study.trials)}\n\n")
            f.write("Mejores hiperparámetros:\n")
            for key, value in study.best_params.items():
                f.write(f"  {key}: {value}\n")
        
        # Guardar historial de trials
        trials_data = []
        for trial in study.trials:
            trial_data = {
                'number': trial.number,
                'value': trial.value,
                'state': trial.state.name,
                **trial.params
            }
            trials_data.append(trial_data)
        
        trials_df = pd.DataFrame(trials_data)
        trials_file = os.path.join(self.results_dir, f"trials_{base_name}.csv")
        trials_df.to_csv(trials_file, index=False)
        
        print(f"Resultados guardados:")
        print(f"  - Estudio: {study_file}")
        print(f"  - Parámetros: {params_file}")
        print(f"  - Trials: {trials_file}")
    
    def analyze_results(self, study: optuna.Study) -> Dict[str, Any]:
        """
        Analiza y visualiza resultados de optimización.
        
        Args:
            study: Estudio completado
            
        Returns:
            Diccionario con análisis de resultados
        """
        print("\n=== ANÁLISIS DE RESULTADOS ===")
        print(f"Mejor valor: {study.best_value:.6f}")
        print(f"Mejores parámetros:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")
        
        # Importancia de parámetros
        try:
            importance = optuna.importance.get_param_importances(study)
            print(f"\nImportancia de parámetros:")
            for param, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True):
                print(f"  {param}: {imp:.4f}")
        except Exception as e:
            print(f"No se pudo calcular importancia: {e}")
            importance = {}
        
        # Crear visualizaciones
        self.create_optimization_plots(study)
        
        # Estadísticas básicas
        values = [trial.value for trial in study.trials if trial.value is not None]
        stats = {
            'best_value': study.best_value,
            'best_params': study.best_params,
            'n_trials': len(study.trials),
            'n_completed': len(values),
            'mean_value': np.mean(values) if values else None,
            'std_value': np.std(values) if values else None,
            'parameter_importance': importance
        }
        
        return stats
    
    def create_optimization_plots(self, study: optuna.Study):
        """
        Crea gráficos de análisis de optimización CORREGIDOS.
        
        Args:
            study: Estudio completado
        """
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # ✅ CORREGIR: Usar números reales de trial y mostrar todos los estados
            completed_trials = [(trial.number, trial.value) for trial in study.trials 
                               if trial.value is not None and np.isfinite(trial.value)]
            pruned_trials = [trial.number for trial in study.trials 
                           if trial.state == optuna.trial.TrialState.PRUNED]
            failed_trials = [trial.number for trial in study.trials 
                           if trial.state == optuna.trial.TrialState.FAIL]
            
            if len(completed_trials) == 0:
                print("⚠️  No hay trials completados para graficar")
                return
            
            trial_numbers, trial_values = zip(*completed_trials)
            
            print(f"📊 Graficando: {len(completed_trials)} completados, {len(pruned_trials)} pruned, {len(failed_trials)} fallidos")
            print(f"🔍 Trials completados: {sorted(trial_numbers)}")
            print(f"🔍 Trials pruned: {sorted(pruned_trials)}")
            
            # ✅ Historia de optimización - EJES ENTEROS
            axes[0,0].plot(trial_numbers, trial_values, 'o-', alpha=0.7, label='Completados', markersize=8)
            
            # Marcar trials pruned y fallidos
            if pruned_trials:
                axes[0,0].scatter(pruned_trials, [max(trial_values) * 1.02] * len(pruned_trials), 
                                color='orange', marker='x', s=100, label='Pruned', alpha=0.7)
            if failed_trials:
                axes[0,0].scatter(failed_trials, [max(trial_values) * 1.04] * len(failed_trials), 
                                color='red', marker='x', s=100, label='Fallidos', alpha=0.7)
            
            axes[0,0].set_title(f'Historia de Optimización (Total: {len(study.trials)} trials)')
            axes[0,0].set_xlabel('Número de Trial')
            axes[0,0].set_ylabel('Valor Objetivo')
            
            # ✅ FORZAR EJES ENTEROS
            all_trial_numbers = sorted(set(trial_numbers) | set(pruned_trials) | set(failed_trials))
            axes[0,0].set_xticks(all_trial_numbers)
            axes[0,0].set_xticklabels([str(int(x)) for x in all_trial_numbers])
            
            axes[0,0].grid(True)
            axes[0,0].legend()
            
            # Distribución de valores (solo completados)
            if len(trial_values) > 1:
                axes[0,1].hist(trial_values, bins=min(10, len(trial_values)), alpha=0.7, color='orange')
            axes[0,1].set_title('Distribución de Valores Objetivo')
            axes[0,1].set_xlabel('Valor Objetivo')
            axes[0,1].set_ylabel('Frecuencia')
            axes[0,1].grid(True)
            
            # Top trials con números reales
            sorted_completed = sorted(completed_trials, key=lambda x: x[1])
            top_trials = sorted_completed[:min(10, len(sorted_completed))]
            
            if top_trials:
                top_numbers, top_values = zip(*top_trials)
                bars = axes[1,0].bar(range(len(top_values)), top_values, color='green', alpha=0.7)
                axes[1,0].set_title(f'Top {len(top_values)} Mejores Trials')
                axes[1,0].set_xlabel('Ranking')
                axes[1,0].set_ylabel('Valor Objetivo')
                axes[1,0].set_xticks(range(len(top_values)))
                axes[1,0].set_xticklabels([f'T{int(n)}' for n in top_numbers], rotation=45)
                axes[1,0].grid(True)
                
                # Agregar valores encima de las barras
                for i, (bar, val) in enumerate(zip(bars, top_values)):
                    axes[1,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                                 f'{val:.1f}', ha='center', va='bottom', fontsize=8)
            
            # ✅ Convergencia CORREGIDA - mostrar TODOS los trials completados
            all_trials_sorted = sorted(study.trials, key=lambda x: x.number)
            convergence_x = []
            convergence_y = []
            current_best = float('inf')
            
            print(f"🔍 DEBUG - Analizando convergencia:")
            
            for trial in all_trials_sorted:
                trial_num = trial.number
                trial_val = trial.value
                trial_state = trial.state.name
                
                print(f"  Trial {trial_num}: valor={trial_val}, estado={trial_state}")
                
                if trial.value is not None and np.isfinite(trial.value):
                    old_best = current_best
                    current_best = min(current_best, trial.value)
                    convergence_x.append(trial.number)
                    convergence_y.append(current_best)
                    
                    if current_best < old_best:
                        print(f"    ✅ NUEVO MEJOR: {current_best:.4f} (mejora de {old_best:.4f})")
                    else:
                        print(f"    ➡️  Sin mejora: mejor sigue siendo {current_best:.4f}")
            
            print(f"🔍 Convergencia X: {convergence_x}")
            print(f"🔍 Convergencia Y: {[round(y, 3) for y in convergence_y]}")
            
            if convergence_x:
                axes[1,1].plot(convergence_x, convergence_y, color='red', linewidth=2, marker='o', markersize=6)
                axes[1,1].set_title('Convergencia (Mejor Valor)')
                axes[1,1].set_xlabel('Número de Trial')
                axes[1,1].set_ylabel('Mejor Valor Hasta Ahora')
                
                # ✅ FORZAR EJES ENTEROS EN CONVERGENCIA
                axes[1,1].set_xticks(convergence_x)
                axes[1,1].set_xticklabels([str(int(x)) for x in convergence_x])
                
                axes[1,1].grid(True)
                
                # ✅ MEJOR VISUALIZACIÓN: Marcar cada punto de mejora
                improvement_count = 0
                for i, (x, y) in enumerate(zip(convergence_x, convergence_y)):
                    if i == 0 or y < convergence_y[i-1]:  # Primer punto o mejora
                        improvement_count += 1
                        axes[1,1].annotate(f'T{int(x)}\n{y:.2f}',
                                         xy=(x, y),
                                         xytext=(0, 15), textcoords='offset points',
                                         bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7),
                                         arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'),
                                         ha='center')
                
                # Marcar el mejor trial final
                best_idx = convergence_y.index(min(convergence_y))
                axes[1,1].annotate(f'MEJOR FINAL\nT{int(convergence_x[best_idx])}: {min(convergence_y):.3f}',
                                 xy=(convergence_x[best_idx], min(convergence_y)),
                                 xytext=(20, -20), textcoords='offset points',
                                 bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8),
                                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2'),
                                 fontsize=10, fontweight='bold')
                
                # ✅ AGREGAR: Información sobre la convergencia
                total_improvement = max(convergence_y) - min(convergence_y) if len(convergence_y) > 1 else 0
                trials_without_improvement = len(convergence_y) - improvement_count
                axes[1,1].text(0.02, 0.98, 
                             f'Mejora total: {total_improvement:.3f}\nTrials sin mejora: {trials_without_improvement}',
                             transform=axes[1,1].transAxes, 
                             verticalalignment='top',
                             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            
            print(f"📈 Convergencia: {len(convergence_x)} puntos graficados")
            
            plt.tight_layout()
            
            # Guardar gráfico
            plot_file = os.path.join(self.results_dir, f'optimization_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.show()
            
            print(f"Gráficos corregidos guardados en: {plot_file}")
            
            # ✅ Mostrar resumen detallado CORREGIDO
            print(f"\n📈 RESUMEN DE TRIALS:")
            print(f"  Total trials ejecutados: {len(study.trials)}")
            print(f"  Completados exitosamente: {len(completed_trials)}")
            print(f"  Pruned (eliminados temprano): {len(pruned_trials)}")
            print(f"  Fallidos: {len(failed_trials)}")
            print(f"  Trials completados: {sorted(trial_numbers)}")
            print(f"  Trials pruned: {sorted(pruned_trials)}")
            if completed_trials:
                best_trial_num, best_value = min(completed_trials, key=lambda x: x[1])
                print(f"  Mejor trial: #{int(best_trial_num)} con valor {best_value:.4f}")
            
        except Exception as e:
            print(f"Error creando gráficos: {e}")
            import traceback
            print(f"Detalle del error: {traceback.format_exc()}")
    
    def load_study(self, study_file: str) -> optuna.Study:
        """
        Carga un estudio guardado previamente.
        
        Args:
            study_file: Archivo del estudio
            
        Returns:
            Estudio cargado
        """
        with open(study_file, 'rb') as f:
            study = pickle.load(f)
        print(f"Estudio cargado desde: {study_file}")
        return study


def create_best_model_from_study(study: optuna.Study, 
                                base_model_params: Dict[str, Any], 
                                device: torch.device) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Crea un modelo con los mejores hiperparámetros del estudio.
    
    Args:
        study: Estudio completado
        base_model_params: Parámetros base del modelo
        device: Dispositivo
        
    Returns:
        Tupla (modelo, hiperparámetros)
    """
    best_params = study.best_params
    
    # Mapeo de funciones de activación
    activation_functions = {
        'tanh': nn.Tanh(),
        'sigmoid': nn.Sigmoid(),
        'relu': nn.ReLU(),
        'gelu': nn.GELU(),
        'swish': nn.SiLU(),
        'elu': nn.ELU()
    }
    
    # ✅ ASEGURAR lambda_bc está presente
    if 'lambda_bc' not in best_params:
        best_params['lambda_bc'] = 10.0
        print("⚠️  lambda_bc agregado con valor por defecto: 10.0")
    
    # Crear parámetros optimizados
    optimized_params = base_model_params.copy()
    optimized_params.update({
        "NumberOfNeurons": best_params['n_neurons'],
        "NumberOfHiddenLayers": best_params['n_hidden_layers'],
        "ActivationFunction": activation_functions[best_params['activation']]
    })
    
    # Importar y crear modelo
    try:
        import sys
        import os
        
        # Agregar directorio actual al path si no está
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
            
        from tp3 import PINN_Module
        model = PINN_Module(optimized_params).to(device)
        
        return model, best_params
        
    except ImportError as e:
        print(f"Error importando PINN_Module: {e}")
        raise


if __name__ == "__main__":
    print("Módulo optuna.py cargado correctamente")
    print("Funciones disponibles:")
    print("  - OptunaOptimizer: Clase principal")
    print("  - create_best_model_from_study: Crear modelo optimizado")
