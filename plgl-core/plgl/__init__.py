"""
PLGL - Preference Learning in Generative Latent Spaces

Transform user preferences into personalized AI-generated content.
"""

__version__ = "1.0.0"

from .core import (
    PreferenceLearner,
    PreferenceSample,
    GenerativeModel,
    PLGLConfig
)

from .models import (
    LinearPreferenceModel,
    DeepPreferenceModel,
    EnsemblePreferenceModel
)

from .sampling import (
    RandomSampler,
    DiversitySampler,
    GridSampler
)

from .optimization import (
    GradientOptimizer,
    EvolutionaryOptimizer,
    BayesianOptimizer
)

from .active import (
    UncertaintySampler,
    ExpectedImprovementSampler,
    BatchActiveSampler
)

from .evaluation import (
    PreferenceEvaluator,
    DiversityMetrics,
    QualityMetrics
)

__all__ = [
    # Core
    "PreferenceLearner",
    "PreferenceSample",
    "GenerativeModel",
    "PLGLConfig",
    
    # Models
    "LinearPreferenceModel",
    "DeepPreferenceModel",
    "EnsemblePreferenceModel",
    
    # Sampling
    "RandomSampler",
    "DiversitySampler", 
    "GridSampler",
    
    # Optimization
    "GradientOptimizer",
    "EvolutionaryOptimizer",
    "BayesianOptimizer",
    
    # Active Learning
    "UncertaintySampler",
    "ExpectedImprovementSampler",
    "BatchActiveSampler",
    
    # Evaluation
    "PreferenceEvaluator",
    "DiversityMetrics",
    "QualityMetrics",
]