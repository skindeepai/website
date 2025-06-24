"""
PLGL (Preference Learning in Generative Latent Spaces)
Core Implementation Example

This is a complete, working implementation of the PLGL framework
that can be adapted for any generative model with a latent space.

Originally pioneered in SkinDeep.ai (2018-2019)
Open sourced for community benefit
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class PreferenceSample:
    """Single preference sample containing latent code, content, and rating"""
    latent: torch.Tensor
    content: torch.Tensor
    rating: float  # 0-1 normalized rating
    metadata: Optional[Dict] = None


class GenerativeModel(ABC):
    """Abstract base class for generative models"""
    
    @abstractmethod
    def generate(self, z: torch.Tensor) -> torch.Tensor:
        """Generate content from latent code"""
        pass
    
    @abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode content to feature representation"""
        pass


class PreferenceModel(nn.Module):
    """Neural network for learning user preferences"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [512, 256, 128]):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class PLGLCore:
    """
    Core PLGL Implementation
    
    This class provides the fundamental functionality for:
    1. Collecting user preferences
    2. Training preference models
    3. Navigating latent spaces to generate personalized content
    """
    
    def __init__(
        self, 
        generator: GenerativeModel,
        latent_dim: int,
        feature_dim: int,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.generator = generator
        self.latent_dim = latent_dim
        self.feature_dim = feature_dim
        self.device = device
        
        self.preference_model = None
        self.preference_samples = []
        self.training_history = []
    
    def sample_latent(self, n_samples: int = 1, diversity: float = 1.0) -> torch.Tensor:
        """
        Sample from latent space with controllable diversity
        
        Args:
            n_samples: Number of samples to generate
            diversity: Standard deviation multiplier (higher = more diverse)
        """
        return torch.randn(n_samples, self.latent_dim, device=self.device) * diversity
    
    def collect_preferences(
        self, 
        n_samples: int = 50,
        sampling_strategy: str = 'random',
        existing_samples: Optional[List[PreferenceSample]] = None
    ) -> List[PreferenceSample]:
        """
        Collect user preferences through rating interface
        
        Args:
            n_samples: Number of samples to collect
            sampling_strategy: 'random', 'diverse', or 'uncertainty'
            existing_samples: Previously collected samples for active learning
        """
        samples = []
        
        if sampling_strategy == 'diverse':
            # Use furthest point sampling for diversity
            latents = self._diverse_sampling(n_samples)
        elif sampling_strategy == 'uncertainty' and self.preference_model is not None:
            # Sample where model is most uncertain
            latents = self._uncertainty_sampling(n_samples)
        else:
            # Random sampling
            latents = self.sample_latent(n_samples)
        
        for i in range(n_samples):
            z = latents[i:i+1]
            
            # Generate content
            with torch.no_grad():
                content = self.generator.generate(z)
                features = self.generator.encode(content)
            
            # Get user rating (this would be replaced with actual UI)
            rating = self._simulate_user_rating(features)
            
            sample = PreferenceSample(
                latent=z,
                content=features,
                rating=rating
            )
            
            samples.append(sample)
            self.preference_samples.append(sample)
        
        return samples
    
    def train_preference_model(
        self,
        samples: Optional[List[PreferenceSample]] = None,
        epochs: int = 200,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        validation_split: float = 0.2
    ):
        """Train the preference model on collected samples"""
        
        if samples is None:
            samples = self.preference_samples
        
        if len(samples) < 10:
            raise ValueError("Need at least 10 samples to train preference model")
        
        # Prepare data
        X = torch.cat([s.content for s in samples])
        y = torch.tensor([s.rating for s in samples], device=self.device)
        
        # Train/validation split
        n_train = int(len(samples) * (1 - validation_split))
        indices = torch.randperm(len(samples))
        
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]
        
        X_train, y_train = X[train_indices], y[train_indices]
        X_val, y_val = X[val_indices], y[val_indices]
        
        # Initialize model
        self.preference_model = PreferenceModel(
            input_dim=self.feature_dim,
            hidden_dims=[512, 256, 128]
        ).to(self.device)
        
        optimizer = Adam(self.preference_model.parameters(), lr=learning_rate)
        criterion = nn.BCELoss()
        
        # Training loop
        for epoch in range(epochs):
            # Train
            self.preference_model.train()
            train_loss = 0
            
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i+batch_size]
                batch_y = y_train[i:i+batch_size]
                
                optimizer.zero_grad()
                pred = self.preference_model(batch_X).squeeze()
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validate
            self.preference_model.eval()
            with torch.no_grad():
                val_pred = self.preference_model(X_val).squeeze()
                val_loss = criterion(val_pred, y_val).item()
            
            self.training_history.append({
                'epoch': epoch,
                'train_loss': train_loss / (len(X_train) // batch_size),
                'val_loss': val_loss
            })
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
    
    def optimize_latent(
        self,
        initial_z: Optional[torch.Tensor] = None,
        steps: int = 1000,
        learning_rate: float = 0.01,
        momentum: float = 0.9,
        return_trajectory: bool = False
    ) -> torch.Tensor:
        """
        Optimize latent vector to maximize preference score
        
        Uses gradient ascent to find optimal point in latent space
        """
        if self.preference_model is None:
            raise ValueError("Must train preference model first")
        
        # Initialize
        if initial_z is None:
            z = self.sample_latent(1)
        else:
            z = initial_z.clone()
        
        z.requires_grad_(True)
        optimizer = torch.optim.SGD([z], lr=learning_rate, momentum=momentum)
        
        trajectory = []
        
        for step in range(steps):
            optimizer.zero_grad()
            
            # Generate and score
            content = self.generator.generate(z)
            features = self.generator.encode(content)
            score = self.preference_model(features)
            
            # Maximize score (minimize negative score)
            loss = -score
            loss.backward()
            optimizer.step()
            
            # Optional: Constrain to reasonable latent space region
            with torch.no_grad():
                z.clamp_(-3, 3)
            
            if return_trajectory:
                trajectory.append(z.clone().detach())
        
        if return_trajectory:
            return z.detach(), trajectory
        return z.detach()
    
    def generate_distribution(
        self,
        n_samples: int = 20,
        optimization_steps: int = 500,
        diversity_weight: float = 0.3,
        min_score_threshold: float = 0.7
    ) -> List[torch.Tensor]:
        """
        Generate a diverse distribution of high-preference samples
        
        Balances between preference score and diversity
        """
        samples = []
        
        while len(samples) < n_samples:
            # Start from random point
            z_init = self.sample_latent(1)
            
            # Optimize
            z_opt = self.optimize_latent(z_init, steps=optimization_steps)
            
            # Check score
            with torch.no_grad():
                content = self.generator.generate(z_opt)
                features = self.generator.encode(content)
                score = self.preference_model(features).item()
            
            # Accept if score is high enough
            if score >= min_score_threshold:
                # Check diversity (distance from existing samples)
                if samples:
                    distances = [torch.norm(z_opt - s) for s in samples]
                    min_distance = min(distances)
                    
                    # Accept based on diversity criterion
                    if min_distance > diversity_weight:
                        samples.append(z_opt)
                else:
                    samples.append(z_opt)
        
        return samples
    
    def _diverse_sampling(self, n_samples: int) -> torch.Tensor:
        """Furthest point sampling for diverse latent codes"""
        samples = [self.sample_latent(1)]
        
        for _ in range(1, n_samples):
            # Generate candidates
            candidates = self.sample_latent(100)
            
            # Find furthest from existing samples
            min_distances = []
            for candidate in candidates:
                distances = [torch.norm(candidate - s) for s in samples]
                min_distances.append(min(distances))
            
            # Select furthest candidate
            best_idx = np.argmax(min_distances)
            samples.append(candidates[best_idx:best_idx+1])
        
        return torch.cat(samples)
    
    def _uncertainty_sampling(self, n_samples: int) -> torch.Tensor:
        """Sample where preference model is most uncertain"""
        candidates = []
        uncertainties = []
        
        # Generate candidate pool
        for _ in range(n_samples * 20):
            z = self.sample_latent(1)
            
            with torch.no_grad():
                content = self.generator.generate(z)
                features = self.generator.encode(content)
                score = self.preference_model(features).item()
            
            # Uncertainty = distance from decision boundary (0.5)
            uncertainty = abs(score - 0.5)
            
            candidates.append(z)
            uncertainties.append(uncertainty)
        
        # Select most uncertain
        indices = np.argsort(uncertainties)[:n_samples]
        return torch.cat([candidates[i] for i in indices])
    
    def _simulate_user_rating(self, features: torch.Tensor) -> float:
        """Simulate user rating for demonstration (replace with actual UI)"""
        # This would be replaced with actual user interface
        # For now, simulate with a random preference function
        
        # Example: prefer certain feature patterns
        score = torch.sigmoid(features.mean() + 0.1 * torch.randn(1)).item()
        return score


# Example usage with a mock generative model
if __name__ == "__main__":
    
    class MockGenerator(GenerativeModel):
        """Mock generator for demonstration"""
        
        def __init__(self, latent_dim: int, output_dim: int):
            self.latent_dim = latent_dim
            self.output_dim = output_dim
            
            # Simple MLP generator
            self.generator = nn.Sequential(
                nn.Linear(latent_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.Linear(512, output_dim),
                nn.Tanh()
            )
            
            # Simple encoder
            self.encoder = nn.Sequential(
                nn.Linear(output_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 128)
            )
        
        def generate(self, z: torch.Tensor) -> torch.Tensor:
            return self.generator(z)
        
        def encode(self, x: torch.Tensor) -> torch.Tensor:
            return self.encoder(x)
    
    # Initialize
    generator = MockGenerator(latent_dim=100, output_dim=512)
    plgl = PLGLCore(
        generator=generator,
        latent_dim=100,
        feature_dim=128
    )
    
    # Collect preferences
    print("Collecting user preferences...")
    preferences = plgl.collect_preferences(n_samples=50, sampling_strategy='diverse')
    
    # Train preference model
    print("\nTraining preference model...")
    plgl.train_preference_model(epochs=100)
    
    # Generate optimal content
    print("\nGenerating personalized content...")
    optimal_z = plgl.optimize_latent(steps=1000)
    optimal_content = generator.generate(optimal_z)
    
    # Generate distribution
    print("\nGenerating preference distribution...")
    distribution = plgl.generate_distribution(n_samples=10)
    
    print(f"\nGenerated {len(distribution)} high-preference samples!")
    print("Ready for deployment in your application.")