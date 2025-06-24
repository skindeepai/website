"""
Core PLGL implementation with clean, production-ready API
"""

import torch
import numpy as np
from typing import List, Optional, Union, Dict, Any, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import warnings
from .models import DeepPreferenceModel
from .sampling import DiversitySampler
from .optimization import GradientOptimizer


@dataclass
class PLGLConfig:
    """Configuration for PLGL system"""
    device: str = field(default_factory=lambda: 'cuda' if torch.cuda.is_available() else 'cpu')
    latent_dim: int = 512
    feature_extractor: Optional[Callable] = None
    preference_model_config: Dict[str, Any] = field(default_factory=lambda: {
        'hidden_dims': [512, 256, 128],
        'dropout': 0.2
    })
    optimization_config: Dict[str, Any] = field(default_factory=lambda: {
        'steps': 1000,
        'learning_rate': 0.01,
        'momentum': 0.9
    })
    sampling_config: Dict[str, Any] = field(default_factory=lambda: {
        'method': 'diverse',
        'diversity': 0.8
    })


@dataclass
class PreferenceSample:
    """Container for preference data"""
    latent: torch.Tensor
    content: Any  # Can be image, audio, text, etc.
    features: Optional[torch.Tensor] = None
    rating: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class GenerativeModel(ABC):
    """Abstract interface for generative models"""
    
    @abstractmethod
    def generate(self, z: torch.Tensor) -> Any:
        """Generate content from latent codes"""
        pass
    
    def encode(self, content: Any) -> torch.Tensor:
        """Optional: Extract features from content"""
        return content
    
    @property
    def latent_dim(self) -> int:
        """Dimension of latent space"""
        raise NotImplementedError


class PreferenceLearner:
    """
    Main PLGL interface for preference learning and generation
    
    This is the primary class users interact with for:
    - Collecting preferences
    - Training preference models
    - Generating personalized content
    """
    
    def __init__(
        self,
        generator: GenerativeModel,
        config: Optional[PLGLConfig] = None,
        rating_interface: Optional[Callable] = None
    ):
        """
        Initialize PLGL system
        
        Args:
            generator: Any generative model implementing GenerativeModel interface
            config: Configuration object (uses defaults if not provided)
            rating_interface: Optional custom rating collection function
        """
        self.generator = generator
        self.config = config or PLGLConfig(latent_dim=generator.latent_dim)
        self.rating_interface = rating_interface
        
        # Initialize components
        self.preference_model = None
        self.samples: List[PreferenceSample] = []
        self.sampler = DiversitySampler(self.config.latent_dim)
        self.optimizer = GradientOptimizer(self.config.optimization_config)
        
        # Move to device
        self.device = self.config.device
        
    def collect_preferences(
        self,
        n_samples: int = 20,
        sampling_strategy: str = 'diverse',
        batch_size: int = 1,
        show_progress: bool = True
    ) -> List[PreferenceSample]:
        """
        Collect user preferences through ratings
        
        Args:
            n_samples: Number of samples to collect
            sampling_strategy: 'random', 'diverse', 'grid', or 'active'
            batch_size: Number of samples to show at once
            show_progress: Show progress bar
            
        Returns:
            List of PreferenceSample objects with ratings
        """
        samples = []
        
        # Select sampling strategy
        if sampling_strategy == 'active' and self.preference_model is None:
            warnings.warn("Active sampling requires trained model, using diverse sampling")
            sampling_strategy = 'diverse'
        
        # Generate latent codes
        if sampling_strategy == 'diverse':
            latents = self.sampler.sample_diverse(n_samples)
        elif sampling_strategy == 'grid':
            latents = self.sampler.sample_grid(n_samples)
        elif sampling_strategy == 'active':
            latents = self._active_sampling(n_samples)
        else:  # random
            latents = torch.randn(n_samples, self.config.latent_dim, device=self.device)
        
        # Generate content and collect ratings
        for i in range(0, n_samples, batch_size):
            batch_latents = latents[i:i+batch_size]
            
            # Generate content
            with torch.no_grad():
                content_batch = [self.generator.generate(z.unsqueeze(0)) 
                               for z in batch_latents]
            
            # Extract features if needed
            if self.config.feature_extractor:
                features_batch = [self.config.feature_extractor(c) for c in content_batch]
            else:
                features_batch = [self.generator.encode(c) for c in content_batch]
            
            # Get ratings
            if self.rating_interface:
                ratings = self.rating_interface(content_batch)
            else:
                ratings = self._default_rating_interface(content_batch)
            
            # Create samples
            for j, (z, content, features, rating) in enumerate(
                zip(batch_latents, content_batch, features_batch, ratings)
            ):
                sample = PreferenceSample(
                    latent=z,
                    content=content,
                    features=features,
                    rating=rating
                )
                samples.append(sample)
                self.samples.append(sample)
        
        return samples
    
    def train(
        self,
        samples: Optional[List[PreferenceSample]] = None,
        model: Optional[torch.nn.Module] = None,
        epochs: int = 100,
        batch_size: int = 32,
        validation_split: float = 0.2,
        early_stopping: bool = True,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Train preference model on collected samples
        
        Args:
            samples: Training samples (uses all collected if None)
            model: Custom preference model (uses default if None)
            epochs: Training epochs
            batch_size: Batch size for training
            validation_split: Fraction for validation
            early_stopping: Enable early stopping
            verbose: Print training progress
            
        Returns:
            Training history dictionary
        """
        if samples is None:
            samples = [s for s in self.samples if s.rating is not None]
        
        if len(samples) < 10:
            raise ValueError(f"Need at least 10 samples, got {len(samples)}")
        
        # Initialize model if needed
        if model is None:
            feature_dim = samples[0].features.shape[-1]
            self.preference_model = DeepPreferenceModel(
                input_dim=feature_dim,
                **self.config.preference_model_config
            ).to(self.device)
        else:
            self.preference_model = model.to(self.device)
        
        # Prepare data
        X = torch.stack([s.features for s in samples]).to(self.device)
        y = torch.tensor([s.rating for s in samples], device=self.device)
        
        # Train model
        history = self.preference_model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            early_stopping=early_stopping,
            verbose=verbose
        )
        
        return history
    
    def generate_optimal(
        self,
        n_attempts: int = 1,
        initial_z: Optional[torch.Tensor] = None,
        return_all: bool = False,
        **optimization_kwargs
    ) -> Union[Any, List[Any]]:
        """
        Generate content optimized for user preferences
        
        Args:
            n_attempts: Number of optimization attempts
            initial_z: Starting point(s) in latent space
            return_all: Return all attempts or just best
            **optimization_kwargs: Override optimization parameters
            
        Returns:
            Generated content (or list if return_all=True)
        """
        if self.preference_model is None:
            raise ValueError("Must train preference model first")
        
        # Merge optimization kwargs
        opt_config = {**self.config.optimization_config, **optimization_kwargs}
        
        results = []
        scores = []
        
        for i in range(n_attempts):
            # Initialize latent code
            if initial_z is not None and i == 0:
                z = initial_z.to(self.device)
            else:
                z = torch.randn(1, self.config.latent_dim, device=self.device)
            
            # Optimize
            z_opt = self.optimizer.optimize(
                z, 
                self.generator,
                self.preference_model,
                **opt_config
            )
            
            # Generate content
            with torch.no_grad():
                content = self.generator.generate(z_opt)
                features = self.generator.encode(content)
                score = self.preference_model(features).item()
            
            results.append(content)
            scores.append(score)
        
        if return_all:
            return results
        else:
            # Return best result
            best_idx = np.argmax(scores)
            return results[best_idx]
    
    def generate_distribution(
        self,
        n_samples: int = 10,
        min_score: float = 0.7,
        diversity: float = 0.3,
        max_attempts: int = None
    ) -> List[Any]:
        """
        Generate diverse distribution of high-preference content
        
        Args:
            n_samples: Number of samples to generate
            min_score: Minimum acceptable preference score
            diversity: Minimum distance between samples (0-1)
            max_attempts: Maximum generation attempts
            
        Returns:
            List of generated content
        """
        if self.preference_model is None:
            raise ValueError("Must train preference model first")
        
        if max_attempts is None:
            max_attempts = n_samples * 10
        
        results = []
        latents = []
        attempts = 0
        
        while len(results) < n_samples and attempts < max_attempts:
            attempts += 1
            
            # Generate candidate
            z = torch.randn(1, self.config.latent_dim, device=self.device)
            z_opt = self.optimizer.optimize(
                z,
                self.generator,
                self.preference_model,
                steps=self.config.optimization_config['steps'] // 2  # Faster
            )
            
            # Check score
            with torch.no_grad():
                content = self.generator.generate(z_opt)
                features = self.generator.encode(content)
                score = self.preference_model(features).item()
            
            # Check acceptance criteria
            if score >= min_score:
                # Check diversity
                if latents:
                    distances = [torch.norm(z_opt - z_prev).item() 
                               for z_prev in latents]
                    min_dist = min(distances)
                    
                    if min_dist >= diversity * torch.norm(z_opt).item():
                        results.append(content)
                        latents.append(z_opt)
                else:
                    results.append(content)
                    latents.append(z_opt)
        
        if len(results) < n_samples:
            warnings.warn(f"Only generated {len(results)}/{n_samples} samples meeting criteria")
        
        return results
    
    def update_preferences(self, new_samples: List[PreferenceSample]):
        """Add new preference samples for continuous learning"""
        self.samples.extend(new_samples)
    
    def save(self, path: str):
        """Save the preference model and configuration"""
        torch.save({
            'preference_model': self.preference_model.state_dict(),
            'config': self.config,
            'samples': self.samples
        }, path)
    
    def load(self, path: str):
        """Load a saved preference model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.config = checkpoint['config']
        self.samples = checkpoint['samples']
        
        # Reconstruct preference model
        if self.samples:
            feature_dim = self.samples[0].features.shape[-1]
            self.preference_model = DeepPreferenceModel(
                input_dim=feature_dim,
                **self.config.preference_model_config
            ).to(self.device)
            self.preference_model.load_state_dict(checkpoint['preference_model'])
    
    def _default_rating_interface(self, content_batch: List[Any]) -> List[float]:
        """Default rating interface (for testing/demo)"""
        print(f"Please rate {len(content_batch)} samples (0-1):")
        ratings = []
        for i, content in enumerate(content_batch):
            while True:
                try:
                    rating = float(input(f"Sample {i+1}: "))
                    if 0 <= rating <= 1:
                        ratings.append(rating)
                        break
                    else:
                        print("Rating must be between 0 and 1")
                except ValueError:
                    print("Invalid input, please enter a number")
        return ratings
    
    def _active_sampling(self, n_samples: int) -> torch.Tensor:
        """Active learning sampling based on model uncertainty"""
        from .active import UncertaintySampler
        active_sampler = UncertaintySampler(
            self.preference_model,
            self.generator,
            self.config.latent_dim
        )
        return active_sampler.select_samples(n_samples)