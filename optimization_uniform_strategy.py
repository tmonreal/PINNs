#!/usr/bin/env python3
"""
optimizacion_estrategia_uniforme.py

Script principal para optimización de hiperparámetros de PINNs usando estrategia uniforme.
Este script es escalable y modular, permitiendo fácil configuración y extensión.

Uso:
    python optimizacion_estrategia_uniforme.py

Autor: Generado para TP3 - Redes Neuronales Informadas por Física
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import json
import argparse
from typing import Dict, List, Tuple, Any
import warnings


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


# Importar módulo de optimización
from optuna_module import OptunaOptimizer, create_best_model_from_study

# Importar funciones del TP3 (ajustar según tu estructura de archivos)
try:
    from tp3 import (
        PINN_Module, 
        generate_collocation_points,
        train_pinn,
        compute_bc_loss,
        compute_pressure_reference_loss,
        plot_losses,
        plot_comparacion_uvp
    )
except ImportError as e:
    print(f"Error importando funciones de TP3: {e}")
    print("Asegúrate de que TP3.py esté en el mismo directorio o en el PYTHONPATH")
    sys.exit(1)

# Configuración
warnings.filterwarnings('ignore')
plt.style.use('default')


class OptimizationConfig:
    """
    Clase de configuración para la optimización.
    Centraliza todos los parámetros para fácil modificación.
    """
    
    def __init__(self):
        # === CONFIGURACIÓN DEL PROBLEMA ===
        self.Re = 100.0  # Número de Reynolds
        self.domain_bounds = {
            'xi': 0.0, 'xf': 1.0,
            'yi': 0.0, 'yf': 1.0
        }
        
        # === CONFIGURACIÓN DEL DISPOSITIVO ===
        self.device = self._setup_device()
        
        # === CONFIGURACIÓN DE DATOS ===
        self.data_files = {
            'pressure': 'Re-100/pressure.mat',
            'velocity': 'Re-100/velocity.mat'
        }
        self.grid_resolution = 201  # Resolución de la grilla interpolada
        
        # === CONFIGURACIÓN DEL MODELO BASE ===
        self.base_model_params = {
            "Device": self.device,
            "Re": self.Re,
            "InputDimensions": 2,      # (x, y)
            "OutputDimensions": 3,     # (u, v, p)
            "NumberOfNeurons": 64,     # Base (se optimizará)
            "NumberOfHiddenLayers": 5, # Base (se optimizará)
            "ActivationFunction": nn.Tanh()  # Base (se optimizará)
        }
        
        # === CONFIGURACIÓN DE OPTIMIZACIÓN ===
        self.optimization_config = {
            'strategy': 'uniform',      # Estrategia fija para este script
            'n_trials': 100, #10, 50,            # Número de trials de Optuna
            'study_name': 'pinn_uniform_optimization',
            'results_dir': 'optuna_results_uniform'
        }
        
        # === CONFIGURACIONES DE DATASET ===
        self.dataset_configs = [
            {'Npde': 1000, 'Nbc': 100, 'description': 'Dataset pequeño'},
            # {'Npde': 10000, 'Nbc': 1000, 'description': 'Dataset medio'},
            # {'Npde': 100000, 'Nbc': 10000, 'description': 'Dataset grande'}
        ]
        
        # === CONFIGURACIÓN DE ENTRENAMIENTO FINAL ===
        self.final_training_config = {
            'epochs': 10000, #5000,
            'early_stopping_patience': 1000,
            'save_models': True,
            'generate_plots': True
        }
        
        # === CONFIGURACIÓN DE ANÁLISIS ===
        self.analysis_config = {
            'compare_with_baseline': True,
            'generate_comparison_plots': True,
            'save_detailed_results': True
        }
    
    def _setup_device(self) -> torch.device:
        """Configura el dispositivo de computación."""
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            device_name = "CUDA GPU"
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            device_name = "Apple Silicon GPU (MPS)"
        else:
            device = torch.device("cpu")
            device_name = "CPU"
        
        print(f"Dispositivo configurado: {device} ({device_name})")
        return device
    
    def update_bounds(self):
        """Actualiza los bounds del modelo."""
        lb = torch.tensor([self.domain_bounds['xi'], self.domain_bounds['yi']], device=self.device)
        ub = torch.tensor([self.domain_bounds['xf'], self.domain_bounds['yf']], device=self.device)
        self.base_model_params['LowerBounds'] = lb
        self.base_model_params['UpperBounds'] = ub
    
    def save_config(self, filepath: str):
        """Guarda la configuración en un archivo JSON."""
        config_dict = {
            'Re': self.Re,
            'domain_bounds': self.domain_bounds,
            'device': str(self.device),
            'optimization_config': self.optimization_config,
            'dataset_configs': self.dataset_configs,
            'final_training_config': self.final_training_config,
            'analysis_config': self.analysis_config,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        print(f"Configuración guardada en: {filepath}")


class DataLoader:
    """
    Clase para cargar y procesar datos de ground-truth.
    """
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.device = config.device
        
    def load_ground_truth_data(self) -> Tuple[torch.Tensor, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Carga y procesa los datos de ground-truth.
        
        Returns:
            Tupla (X_eval, U_grid, V_grid, P_grid, x_coords, y_coords)
        """
        try:
            from scipy.io import loadmat
            from scipy.interpolate import griddata
            
            print("Cargando datos de ground-truth...")
            
            # Cargar datos de archivos .mat
            pressure_mat = loadmat(self.config.data_files['pressure'])
            velocity_mat = loadmat(self.config.data_files['velocity'])
            
            x = pressure_mat['x'].squeeze()
            y = pressure_mat['y'].squeeze()
            p = pressure_mat['p'].squeeze()
            u = velocity_mat['u'].squeeze()
            v = velocity_mat['v'].squeeze()
            
            print(f"Datos cargados: {len(x)} puntos")
            
            # Crear grilla regular
            x_unique = np.linspace(x.min(), x.max(), self.config.grid_resolution)
            y_unique = np.linspace(y.min(), y.max(), self.config.grid_resolution)
            X_grid, Y_grid = np.meshgrid(x_unique, y_unique)
            
            # Interpolar campos sobre la grilla
            U_grid = griddata((x, y), u, (X_grid, Y_grid), method='cubic')
            V_grid = griddata((x, y), v, (X_grid, Y_grid), method='cubic')
            P_grid = griddata((x, y), p, (X_grid, Y_grid), method='cubic')
            
            # Preparar puntos para evaluación del modelo
            X_eval = torch.tensor(
                np.stack([X_grid.flatten(), Y_grid.flatten()], axis=1), 
                dtype=torch.float32, 
                device=self.device
            )
            
            print(f"Grilla interpolada: {X_grid.shape}")
            print(f"Puntos de evaluación: {X_eval.shape}")
            
            return X_eval, U_grid, V_grid, P_grid, x, y
            
        except Exception as e:
            print(f"Error cargando datos: {e}")
            raise
    
    def visualize_ground_truth(self, U_grid: np.ndarray, V_grid: np.ndarray, P_grid: np.ndarray, 
                              x: np.ndarray, y: np.ndarray):
        """Visualiza los campos de ground-truth."""
        fig, axs = plt.subplots(1, 3, figsize=(18, 5))
        extent = [x.min(), x.max(), y.min(), y.max()]
        
        # Campo u
        im0 = axs[0].imshow(U_grid, extent=extent, origin='lower', cmap='RdBu_r', aspect='equal')
        axs[0].set_title("Velocidad u - Ground Truth")
        axs[0].set_xlabel("x")
        axs[0].set_ylabel("y")
        plt.colorbar(im0, ax=axs[0])
        
        # Campo v
        im1 = axs[1].imshow(V_grid, extent=extent, origin='lower', cmap='RdBu_r', aspect='equal')
        axs[1].set_title("Velocidad v - Ground Truth")
        axs[1].set_xlabel("x")
        axs[1].set_ylabel("y")
        plt.colorbar(im1, ax=axs[1])
        
        # Campo p
        im2 = axs[2].imshow(P_grid, extent=extent, origin='lower', cmap='Spectral', aspect='equal')
        axs[2].set_title("Presión p - Ground Truth")
        axs[2].set_xlabel("x")
        axs[2].set_ylabel("y")
        plt.colorbar(im2, ax=axs[2])
        
        plt.tight_layout()
        
        # Guardar gráfico
        os.makedirs(self.config.optimization_config['results_dir'], exist_ok=True)
        plt.savefig(
            os.path.join(self.config.optimization_config['results_dir'], 'ground_truth_fields.png'),
            dpi=300, bbox_inches='tight'
        )
        plt.show()


class OptimizedTrainer:
    """
    Clase para entrenar modelos con hiperparámetros optimizados.
    """
    
    def __init__(self, config: OptimizationConfig, validation_data: tuple):
        self.config = config
        self.validation_data = validation_data
        self.device = config.device
        
    def train_optimized_model(self, 
                             best_hyperparams: Dict[str, Any], 
                             Npde: int, 
                             Nbc: int,
                             description: str = "") -> Dict[str, Any]:
        """
        Entrena un modelo con hiperparámetros optimizados.
        
        Args:
            best_hyperparams: Mejores hiperparámetros del estudio
            Npde: Número de puntos PDE
            Nbc: Número de puntos BC
            description: Descripción del experimento
            
        Returns:
            Diccionario con resultados del entrenamiento
        """
        print(f"\n=== ENTRENANDO MODELO OPTIMIZADO ===")
        print(f"Configuración: Npde={Npde}, Nbc={Nbc}")
        print(f"Descripción: {description}")
        
        # Crear modelo con hiperparámetros optimizados
        model, _ = create_best_model_from_study(
            type('Study', (), {'best_params': best_hyperparams})(),
            self.config.base_model_params,
            self.device
        )
        
        # Configurar optimizador y scheduler
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=best_hyperparams['learning_rate'],
            weight_decay=best_hyperparams['weight_decay']
        )
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=best_hyperparams['scheduler_gamma']
        )
        
        # Generar puntos de colocación
        pde_points, bottom, top, left, right, _ = generate_collocation_points(
            strategy=self.config.optimization_config['strategy'],
            N_pde=Npde,
            N_bc=Nbc,
            device=self.device
        )
        
        # Entrenar modelo
        final_epochs = self.config.final_training_config['epochs']
        print(f"Entrenando modelo final con {final_epochs} épocas...")
        # Asegurar que se use el valor de epochs optimizado
        print(f"🎯 DEBUG: Configuración epochs = {final_epochs}")
        print(f"🎯 DEBUG: Llamando train_pinn con epochs = {final_epochs}")
        
        losses = train_pinn(
            model=model,
            pde_points=pde_points,
            top=top,
            bottom=bottom,
            left=left,
            right=right,
            epochs=final_epochs,  # ← Asegurar que se use este valor
            optimizer=optimizer,
            scheduler=scheduler,
            weight_pde=best_hyperparams['weight_pde'],
            lambda_bc=best_hyperparams['lambda_bc'],
            weight_pressure_ref=best_hyperparams['weight_pressure_ref'],
            strategy=f"optimized_{self.config.optimization_config['strategy']}",
            Npde=Npde,
            Nbc=Nbc
        )
        
        # Evaluar modelo
        errors = self._evaluate_model(model)
        
        # Generar visualizaciones si está configurado
        if self.config.final_training_config['generate_plots']:
            self._generate_model_plots(model, losses, Npde, Nbc, description)
        
        return {
            'model': model,
            'losses': losses,
            'errors': errors,
            'hyperparams': best_hyperparams,
            'config': {'Npde': Npde, 'Nbc': Nbc, 'description': description}
        }
    
    def _evaluate_model(self, model: nn.Module) -> Dict[str, float]:
        """Evalúa el modelo y calcula errores."""
        X_eval, U_grid, V_grid, P_grid = self.validation_data
        
        model.eval()
        with torch.no_grad():
            uvp_pred = model(X_eval)
            u_pred = uvp_pred[:, 0].cpu().numpy().reshape(U_grid.shape)
            v_pred = uvp_pred[:, 1].cpu().numpy().reshape(V_grid.shape)
            p_pred = uvp_pred[:, 2].cpu().numpy().reshape(P_grid.shape)
        
        # Calcular errores L2
        error_u = np.linalg.norm(u_pred - U_grid)
        error_v = np.linalg.norm(v_pred - V_grid)
        error_p = np.linalg.norm(p_pred - P_grid)
        avg_error = (error_u + error_v + error_p) / 3
        
        errors = {
            'error_u': error_u,
            'error_v': error_v,
            'error_p': error_p,
            'avg_error': avg_error
        }
        
        print(f"Errores del modelo optimizado:")
        print(f"  Error u: {error_u:.6f}")
        print(f"  Error v: {error_v:.6f}")
        print(f"  Error p: {error_p:.6f}")
        print(f"  Error promedio: {avg_error:.6f}")
        
        return errors
    
    def _generate_model_plots(self, model: nn.Module, losses: Dict, Npde: int, Nbc: int, description: str):
        """Genera gráficos del modelo entrenado."""
        X_eval, U_grid, V_grid, P_grid, x, y = self.validation_data + (None, None)  # Ajustar según tu estructura
        
        # Gráfico de pérdidas
        plot_losses(losses)
        
        # Predicciones del modelo
        model.eval()
        with torch.no_grad():
            uvp_pred = model(X_eval)
            u_pred = uvp_pred[:, 0].cpu().numpy().reshape(U_grid.shape)
            v_pred = uvp_pred[:, 1].cpu().numpy().reshape(V_grid.shape)
            p_pred = uvp_pred[:, 2].cpu().numpy().reshape(P_grid.shape)
        
        # Gráfico de comparación
        # plot_comparacion_uvp(
        #     u_pred, v_pred, p_pred,
        #     f"Optimized_{self.config.optimization_config['strategy']}",
        #     Npde, Nbc,
        #     U_grid, V_grid, P_grid, x, y
        # )


class ResultsAnalyzer:
    """
    Clase para analizar y comparar resultados.
    """
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.results_dir = config.optimization_config['results_dir']
    
    def create_optimization_summary(self, 
                                   optimization_results: List[Dict],
                                   study_results: List[Dict]) -> pd.DataFrame:
        """
        Crea un resumen de todos los resultados de optimización.
        
        Args:
            optimization_results: Resultados de entrenamiento optimizado
            study_results: Resultados de estudios de Optuna
            
        Returns:
            DataFrame con resumen completo
        """
        summary_data = []
        
        for opt_result, study_result in zip(optimization_results, study_results):
            config = opt_result['config']
            errors = opt_result['errors']
            hyperparams = opt_result['hyperparams']
            study = study_result['study']
            
            summary_row = {
                'Npde': config['Npde'],
                'Nbc': config['Nbc'],
                'description': config['description'],
                'strategy': self.config.optimization_config['strategy'],
                'n_trials': len(study.trials),
                'best_objective_value': study.best_value,
                'final_error_u': errors['error_u'],
                'final_error_v': errors['error_v'],
                'final_error_p': errors['error_p'],
                'final_avg_error': errors['avg_error'],
                'best_n_neurons': hyperparams['n_neurons'],
                'best_n_layers': hyperparams['n_hidden_layers'],
                'best_activation': hyperparams['activation'],
                'best_lr': hyperparams['learning_rate'],
                'best_weight_decay': hyperparams['weight_decay'],
                'best_weight_pde': hyperparams['weight_pde'],
                'best_weight_pressure_ref': hyperparams['weight_pressure_ref']
            }
            
            summary_data.append(summary_row)
        
        df_summary = pd.DataFrame(summary_data)
        
        # Guardar resumen
        summary_file = os.path.join(self.results_dir, 'optimization_summary.csv')
        df_summary.to_csv(summary_file, index=False)
        
        print(f"\nResumen de optimización guardado en: {summary_file}")
        return df_summary
    
    def create_comparison_plots(self, df_summary: pd.DataFrame):
        """
        Crea gráficos de comparación de resultados.
        
        Args:
            df_summary: DataFrame con resumen de resultados
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Error vs tamaño de dataset
        axes[0, 0].loglog(df_summary['Npde'], df_summary['final_avg_error'], 'o-', markersize=8)
        axes[0, 0].set_xlabel('Número de puntos PDE')
        axes[0, 0].set_ylabel('Error promedio final')
        axes[0, 0].set_title('Convergencia con Optimización')
        axes[0, 0].grid(True)
        
        # Distribución de hiperparámetros
        axes[0, 1].bar(df_summary['description'], df_summary['best_n_neurons'], alpha=0.7)
        axes[0, 1].set_title('Mejores Número de Neuronas')
        axes[0, 1].set_ylabel('Neuronas')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        axes[1, 0].bar(df_summary['description'], df_summary['best_n_layers'], alpha=0.7, color='orange')
        axes[1, 0].set_title('Mejores Número de Capas')
        axes[1, 0].set_ylabel('Capas Ocultas')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Learning rates
        axes[1, 1].semilogy(df_summary['description'], df_summary['best_lr'], 'o-', color='red')
        axes[1, 1].set_title('Mejores Learning Rates')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Guardar gráfico
        plot_file = os.path.join(self.results_dir, 'optimization_comparison.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Gráficos de comparación guardados en: {plot_file}")


def main():
    """Función principal del script."""
    
    print("=" * 70)
    print("OPTIMIZACIÓN DE HIPERPARÁMETROS - ESTRATEGIA UNIFORME")
    print("=" * 70)
    
    # === CONFIGURACIÓN ===
    config = OptimizationConfig()
    config.update_bounds()
    
    # Crear directorio de resultados
    os.makedirs(config.optimization_config['results_dir'], exist_ok=True)
    
    # Guardar configuración
    config_file = os.path.join(config.optimization_config['results_dir'], 'config.json')
    config.save_config(config_file)
    
    # === CARGA DE DATOS ===
    data_loader = DataLoader(config)
    try:
        X_eval, U_grid, V_grid, P_grid, x, y = data_loader.load_ground_truth_data()
        validation_data = (X_eval, U_grid, V_grid, P_grid)
        
        # Visualizar datos de ground-truth
        data_loader.visualize_ground_truth(U_grid, V_grid, P_grid, x, y)
        
    except Exception as e:
        print(f"Error en carga de datos: {e}")
        return 1
    
    # === OPTIMIZACIÓN ===
    print(f"\n{'='*50}")
    print("INICIANDO PROCESO DE OPTIMIZACIÓN")
    print(f"{'='*50}")
    
    optimizer = OptunaOptimizer(
        base_model_params=config.base_model_params,
        device=config.device,
        validation_data=validation_data,
        results_dir=config.optimization_config['results_dir']
    )
    
    study_results = []
    optimization_results = []
    
    # Ejecutar optimización para cada configuración de dataset
    for dataset_config in config.dataset_configs:
        Npde = dataset_config['Npde']
        Nbc = dataset_config['Nbc']
        description = dataset_config['description']
        
        print(f"\n--- Optimizando para {description} (Npde={Npde}, Nbc={Nbc}) ---")
        
        # Ejecutar optimización
        study = optimizer.run_optimization(
            strategy=config.optimization_config['strategy'],
            N_pde=Npde,
            N_bc=Nbc,
            n_trials=config.optimization_config['n_trials'],
            study_name=f"{config.optimization_config['study_name']}_Npde{Npde}_Nbc{Nbc}"
        )
        
        # Analizar resultados del estudio
        study_analysis = optimizer.analyze_results(study)
        study_results.append({
            'study': study,
            'analysis': study_analysis,
            'config': dataset_config
        })
        
        # === ENTRENAMIENTO CON HIPERPARÁMETROS OPTIMIZADOS ===
        print(f"\n--- Entrenando modelo final para {description} ---")
        
        trainer = OptimizedTrainer(config, validation_data)
        optimized_result = trainer.train_optimized_model(
            best_hyperparams=study.best_params,
            Npde=Npde,
            Nbc=Nbc,
            description=description
        )
        
        optimization_results.append(optimized_result)
    
    # === ANÁLISIS DE RESULTADOS ===
    print(f"\n{'='*50}")
    print("ANÁLISIS FINAL DE RESULTADOS")
    print(f"{'='*50}")
    
    analyzer = ResultsAnalyzer(config)
    
    # Crear resumen completo
    df_summary = analyzer.create_optimization_summary(optimization_results, study_results)
    print("\nResumen de optimización:")
    print(df_summary.round(6))
    
    # Crear gráficos de comparación
    analyzer.create_comparison_plots(df_summary)
    
    # === GUARDAR RESULTADOS FINALES ===
    final_results = {
        'config': config_file,
        'optimization_results': optimization_results,
        'study_results': study_results,
        'summary_dataframe': df_summary,
        'timestamp': datetime.now().isoformat()
    }
    
    results_file = os.path.join(config.optimization_config['results_dir'], 'final_results.pkl')
    import pickle
    with open(results_file, 'wb') as f:
        pickle.dump(final_results, f)
    
    print(f"\n{'='*70}")
    print("OPTIMIZACIÓN COMPLETADA EXITOSAMENTE")
    print(f"{'='*70}")
    print(f"Resultados guardados en: {config.optimization_config['results_dir']}")
    print(f"Resumen: {df_summary.shape[0]} configuraciones optimizadas")
    print(f"Mejor configuración: Npde={df_summary.loc[df_summary['final_avg_error'].idxmin(), 'Npde']}")
    print(f"Mejor error promedio: {df_summary['final_avg_error'].min():.6f}")
    
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nOptimización interrumpida por el usuario")
        sys.exit(1)
    except Exception as e:
        print(f"\nError durante la optimización: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
