# PLGL - Preference Learning in Generative Latent Spaces

[![PyPI version](https://badge.fury.io/py/plgl.svg)](https://badge.fury.io/py/plgl)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation Status](https://readthedocs.org/projects/plgl/badge/?version=latest)](https://skindeep.ai/documentation)
[![GitHub stars](https://img.shields.io/github/stars/plgl/core.svg)](https://github.com/plgl/core/stargazers)

Transform user preferences into personalized AI-generated content across any domain.

PLGL enables you to build AI applications that learn from user preferences and generate personalized content by navigating the latent spaces of generative models. Originally pioneered in 2018-2019 with SkinDeep.ai, this technology is now open source for the community.

## 🚀 Quick Start

```bash
pip install plgl

# For specific deep learning frameworks
pip install plgl[torch]     # PyTorch support
pip install plgl[tensorflow] # TensorFlow support
pip install plgl[jax]       # JAX support
```

### Basic Example

```python
from plgl import PreferenceLearner
from your_model import load_generator  # Your generative model

# Initialize with any generative model
generator = load_generator('path/to/model')
learner = PreferenceLearner(generator, latent_dim=512)

# Collect user preferences (ratings)
preferences = learner.collect_preferences(n_samples=20)

# Train personalized preference model
learner.train(preferences)

# Generate optimal personalized content
personalized_content = learner.generate_optimal()
```

## 🌟 Key Features

- **Universal**: Works with any generative model that has a latent space (GANs, VAEs, Diffusion Models)
- **Simple**: Users just rate samples - no technical knowledge required
- **Powerful**: Sophisticated optimization to find ideal content in high-dimensional spaces
- **Efficient**: Active learning minimizes the number of ratings needed
- **Flexible**: Supports various rating types (binary, scaled, comparative)

## 📚 How It Works

PLGL implements a three-stage pipeline:

1. **Preference Capture**: Users rate generated samples
2. **Preference Learning**: Build personalized ML models from ratings
3. **Latent Space Navigation**: Optimize through generative model's latent space to create ideal content

```
User Ratings → Preference Model → Latent Space Search → Personalized Content
```

## 🎯 Applications

PLGL has been successfully applied to:

- 🎵 **Music Generation** - Personalized compositions and playlists
- 🎨 **Art & Design** - Custom artwork matching aesthetic preferences  
- 🧬 **Drug Discovery** - Molecules with desired properties
- 🏗️ **Architecture** - Buildings matching lifestyle preferences
- 📚 **Content Creation** - Stories, articles, and narratives
- 👗 **Fashion Design** - Clothing matching personal style
- 🎮 **Game Design** - Levels matching player preferences
- And many more...

## 💻 Installation Options

### From PyPI (Recommended)
```bash
pip install plgl
```

### From Source
```bash
git clone https://github.com/plgl/core.git
cd core
pip install -e .
```

### Development Installation
```bash
pip install plgl[dev]  # Include development dependencies
```

## 🔧 Advanced Usage

### Active Learning for Efficient Preference Capture
```python
from plgl.active import UncertaintySampler

sampler = UncertaintySampler(learner.preference_model)
informative_samples = sampler.select_samples(n=10)
```

### Multi-Modal Preferences
```python
from plgl.multimodal import MultiModalLearner

learner = MultiModalLearner({
    'visual': visual_generator,
    'audio': audio_generator
})
```

### Real-Time Adaptation
```python
from plgl.realtime import AdaptiveLearner

adaptive = AdaptiveLearner(generator)
adaptive.update_from_feedback(content, user_rating)
```

## 📖 Documentation

- [Full Documentation](https://skindeep.ai/documentation)
- [Getting Started Guide](https://skindeep.ai/getting-started)
- [API Reference](https://skindeep.ai/api)
- [Examples Gallery](https://github.com/plgl/examples)

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

```bash
# Fork the repo, then:
git clone https://github.com/YOUR_USERNAME/core.git
cd core
pip install -e .[dev]
pytest  # Run tests
```

## 📊 Benchmarks

| Model Type | Preferences Needed | Optimization Time | Success Rate |
|------------|-------------------|-------------------|--------------|
| StyleGAN2  | 15-20            | < 2 seconds       | 94%          |
| MusicVAE   | 20-30            | < 3 seconds       | 91%          |
| MolecularVAE | 30-40          | < 5 seconds       | 89%          |

## 🏛️ History

- **2018-2019**: Original development with SkinDeep.ai
- **2019**: Provisional patent filed
- **2024**: Open sourced for community benefit

## 📄 Citation

If you use PLGL in your research, please cite:

```bibtex
@software{plgl2024,
  title = {PLGL: Preference Learning in Generative Latent Spaces},
  author = {PLGL Contributors},
  year = {2024},
  url = {https://github.com/plgl/core}
}
```

## 📺 Original Demo Videos

See PLGL in action with the original SkinDeep.ai implementation:
- [Technology Overview (2019)](https://www.youtube.com/watch?v=-6mAyFJ4_ME)
- [Technical Deep Dive (2019)](https://www.youtube.com/watch?v=M4oQLev_Sk8)

## 🌐 Community

- **Discord**: [Join our Discord](https://discord.gg/plgl)
- **Twitter**: [@plgl_ai](https://twitter.com/plgl_ai)
- **Forum**: [community.plgl.ai](https://community.plgl.ai)

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <strong>Transform preferences into personalized AI content</strong><br>
  <a href="https://skindeep.ai">Website</a> • 
  <a href="https://github.com/plgl/core">GitHub</a> • 
  <a href="https://skindeep.ai/documentation">Documentation</a>
</p>