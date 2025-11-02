# T-CRIS: Temporal Cancer Recurrence Intelligence System

**An AI-Powered Platform for Bladder Cancer Recurrence Prediction and Treatment Optimization**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

## 🎯 Project Overview

T-CRIS is a comprehensive, production-ready AI platform that combines classical survival analysis with modern deep learning for bladder cancer recurrence prediction. It features personalized treatment recommendations, interpretable AI explanations, and interactive visualizations.

### Novel Contributions

1. **Hybrid Statistical-DL Framework**: Seamlessly combines Cox PH, Anderson-Gill, Random Survival Forests, LSTM, and Transformer models
2. **Multi-Format Data Fusion**: Automatic unification of WLW, Anderson-Gill, and standard survival data formats
3. **Counterfactual Treatment Analysis**: Personalized "what-if" scenarios for treatment selection
4. **Competing Risks Neural Network**: Multi-task deep learning for recurrence + death events
5. **Attention-Based Temporal Mining**: Discovers which past recurrences predict future risk
6. **Interactive Dashboard + REST API**: Production-ready deployment with FastAPI and Streamlit

---

## 📁 Project Structure

```
project-bcrs/
├── data/                           # Data directory
│   ├── raw/                        # Original CSV files (bladder.csv, bladder1.csv, bladder2.csv)
│   ├── processed/                  # Cleaned, unified data
│   ├── features/                   # Engineered features
│   └── validation/                 # Train/test splits
│
├── src/tcris/                      # Main package
│   ├── config/                     # Configuration management
│   │   ├── settings.py             # Pydantic settings (single source of truth)
│   │   └── logging.yaml            # Logging configuration
│   │
│   ├── data/                       # Data layer
│   │   ├── loaders.py              # Unified CSV loader (DRY principle)
│   │   ├── validators.py           # Data validation (Great Expectations)
│   │   ├── fusion.py               # Multi-format data fusion
│   │   ├── preprocessors.py        # Data preprocessing
│   │   └── augmentation.py         # Data augmentation
│   │
│   ├── features/                   # Feature engineering
│   │   ├── extractors.py           # Feature extraction
│   │   ├── transformers.py         # sklearn-compatible transformers
│   │   └── temporal.py             # Time-dependent features
│   │
│   ├── models/                     # Model layer
│   │   ├── base.py                 # Abstract base classes (DRY)
│   │   ├── statistical/            # Classical survival models
│   │   │   ├── cox.py              # Cox Proportional Hazards
│   │   │   ├── anderson_gill.py    # Anderson-Gill recurrent events
│   │   │   └── competing_risks.py  # Fine-Gray competing risks
│   │   ├── machine_learning/       # ML models
│   │   │   ├── random_survival_forest.py
│   │   │   └── gradient_boosting.py
│   │   ├── deep_learning/          # Deep learning models
│   │   │   ├── lstm_temporal.py    # LSTM for recurrence sequences
│   │   │   ├── transformer.py      # Transformer with attention
│   │   │   ├── competing_risks_nn.py  # Multi-task competing risks
│   │   │   └── bayesian_survival.py   # Bayesian uncertainty quantification
│   │   └── ensemble/               # Ensemble methods
│   │       ├── stacking.py
│   │       └── meta_model.py
│   │
│   ├── prediction/                 # Application layer
│   │   ├── predictor.py            # Main prediction engine
│   │   ├── counterfactual.py       # Treatment comparison
│   │   └── risk_trajectory.py      # Dynamic risk evolution
│   │
│   ├── interpretation/             # Interpretability
│   │   ├── shap_explainer.py       # SHAP explanations
│   │   ├── lime_explainer.py       # LIME local explanations
│   │   └── attention_viz.py        # Attention visualization
│   │
│   ├── similarity/                 # Patient similarity
│   │   ├── distance_metrics.py
│   │   └── clustering.py
│   │
│   ├── evaluation/                 # Model evaluation
│   │   ├── metrics.py              # C-index, Brier score, etc.
│   │   ├── calibration.py          # Calibration plots
│   │   └── validators.py           # Cross-validation
│   │
│   ├── visualization/              # Visualization layer
│   │   ├── survival_plots.py       # Kaplan-Meier, hazard plots
│   │   ├── risk_plots.py           # Risk trajectories
│   │   ├── interpretability_plots.py  # SHAP, attention plots
│   │   └── dashboard_components.py    # Reusable UI components
│   │
│   ├── reports/                    # Report generation
│   │   ├── statistical_report.py
│   │   ├── patient_report.py
│   │   └── templates/              # LaTeX/Jinja templates
│   │
│   ├── api/                        # FastAPI REST API
│   │   ├── main.py                 # FastAPI app
│   │   ├── routes/                 # API routes
│   │   │   ├── prediction.py
│   │   │   ├── analysis.py
│   │   │   └── reports.py
│   │   └── schemas.py              # Pydantic models (DRY)
│   │
│   └── utils/                      # Utilities
│       ├── decorators.py           # Common decorators (DRY)
│       ├── exceptions.py           # Custom exceptions
│       └── helpers.py              # Helper functions
│
├── dashboard/                      # Streamlit dashboard
│   ├── app.py                      # Main Streamlit app
│   ├── pages/                      # Multi-page app
│   │   ├── 01_overview.py
│   │   ├── 02_survival_analysis.py
│   │   ├── 03_predictions.py
│   │   ├── 04_counterfactual.py
│   │   └── 05_interpretability.py
│   └── components/                 # Reusable UI components (DRY)
│
├── notebooks/                      # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_statistical_analysis.ipynb
│   ├── 03_model_development.ipynb
│   ├── 04_model_evaluation.ipynb
│   └── 05_presentation_figures.ipynb
│
├── tests/                          # Tests
│   ├── unit/                       # Unit tests
│   ├── integration/                # Integration tests
│   └── fixtures/                   # Test data (DRY)
│
├── scripts/                        # CLI scripts
│   ├── train_models.py             # Training pipeline
│   ├── generate_report.py          # Report generator
│   └── validate_data.py            # Data validation
│
├── models/                         # Saved model artifacts
│   ├── statistical/
│   ├── deep_learning/
│   └── ensemble/
│
├── outputs/                        # Generated outputs
│   ├── reports/                    # PDF reports
│   ├── figures/                    # Visualizations
│   └── predictions/                # Prediction results
│
├── docs/                           # Documentation
│   ├── api/                        # API documentation
│   ├── user_guide/                 # User guide
│   └── technical/                  # Technical documentation
│
├── pyproject.toml                  # Poetry dependencies
├── Makefile                        # Common commands (KISS)
├── .env.example                    # Environment variables template
├── README.md                       # Dataset information
├── DATA_INFO.md                    # Detailed data documentation
└── PROJECT_README.md               # This file
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- Poetry (for dependency management)

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd project-bcrs
   ```

2. **Install dependencies**:
   ```bash
   make install
   # or manually:
   poetry install
   ```

3. **Set up environment**:
   ```bash
   make setup
   # This copies .env.example to .env
   # Edit .env with your configuration
   ```

4. **Validate data**:
   ```bash
   make validate-data
   ```

### Usage

#### 1. Train Models
```bash
make train
```

#### 2. Run API Server
```bash
make api
# Access Swagger docs at http://localhost:8000/docs
```

#### 3. Launch Dashboard
```bash
make dashboard
# Opens at http://localhost:8501
```

#### 4. Run Jupyter Notebooks
```bash
make notebook
```

---

## 🎨 Key Features

### 1. Data Processing
- **Multi-Format Support**: Handles WLW, Anderson-Gill, and standard formats
- **Automatic Format Detection**: Intelligently detects data format
- **Data Validation**: Comprehensive quality checks with Great Expectations
- **Feature Engineering**: 20+ engineered features (temporal, tumor progression, interactions)

### 2. Modeling

#### Statistical Models
- Cox Proportional Hazards
- Anderson-Gill Frailty Model
- Wei-Lin-Weissfeld Marginal Model
- Fine-Gray Competing Risks Model
- Aalen Additive Model

#### Machine Learning Models
- Random Survival Forest
- Gradient Boosting Survival Analysis

#### Deep Learning Models
- LSTM Temporal Recurrence Model
- Transformer with Attention Mechanism
- Competing Risks Neural Network
- Bayesian Survival Network

#### Ensemble
- Stacking ensemble combining all models
- Optimized meta-model

### 3. Prediction & Analysis
- **Individual Risk Prediction**: Patient-specific recurrence risk scores
- **Survival Curves**: Time-dependent survival probabilities
- **Counterfactual Analysis**: "What-if" treatment scenarios
- **Dynamic Risk Trajectories**: Risk evolution over time
- **Uncertainty Quantification**: Confidence intervals and credible regions

### 4. Interpretability
- **SHAP Values**: Feature importance for each prediction
- **LIME Explanations**: Local model behavior
- **Attention Visualization**: Which past events matter most
- **Feature Importance**: Global model understanding

### 5. Interactive Dashboard

#### Pages:
1. **Overview**: Dataset summary, statistics
2. **Survival Analysis**: Kaplan-Meier curves, log-rank tests
3. **Predictions**: Individual patient risk assessment
4. **Counterfactual**: Treatment comparison
5. **Interpretability**: Model explanations, feature importance

### 6. REST API

#### Endpoints:
- `POST /api/v1/predict` - Get recurrence prediction
- `POST /api/v1/predict/batch` - Batch predictions
- `POST /api/v1/counterfactual` - Treatment comparison
- `GET /api/v1/survival_curve` - Survival curves
- `POST /api/v1/similar_patients` - Find similar patients
- `GET /api/v1/models` - List available models
- `POST /api/v1/reports/generate` - Generate reports

---

## 📊 Datasets

The project uses three bladder cancer recurrence datasets:

1. **bladder.csv**: WLW format, 85 patients, up to 4 recurrences
2. **bladder1.csv**: Extended WLW, 118 patients, up to 9 recurrences
3. **bladder2.csv**: Anderson-Gill format, 85 patients

See [README.md](README.md) and [DATA_INFO.md](DATA_INFO.md) for detailed information.

---

## 🧪 Testing

```bash
# Run all tests with coverage
make test

# Run quick tests (no coverage)
make test-quick

# Run specific test file
poetry run pytest tests/unit/test_data_loaders.py
```

Target coverage: >80%

---

## 📝 Code Quality

### Formatting
```bash
# Format code
make format

# Check formatting
make format-check
```

### Linting
```bash
make lint
```

### All Quality Checks
```bash
make all
```

---

## 🎓 Design Principles

### KISS (Keep It Simple, Stupid)
- Simple, consistent APIs (`.fit()`, `.predict()`)
- Minimal dependencies
- Clear, self-documenting code
- Flat module structure

### DRY (Don't Repeat Yourself)
- Base classes define common interfaces once
- Shared utilities in `utils/` module
- Single data loader handles all formats
- Reusable dashboard components
- Pydantic Settings as single source of truth

---

## 📈 Model Performance

### Target Metrics
- **C-index** (discrimination): >0.70
- **Integrated Brier Score**: <0.20
- **Calibration slope**: ~1.0
- **Time-dependent AUC at 1 year**: >0.75
- **API latency**: <500ms per prediction

---

## 🎤 Presentation Highlights

1. **Live Demo**: Interactive dashboard with real-time predictions
2. **Attention Visualization**: Heatmaps showing temporal patterns learned by transformer
3. **Counterfactual Analysis**: Side-by-side treatment comparison for personalized medicine
4. **Model Performance**: Ensemble outperforms individual models
5. **Interpretability**: SHAP waterfall plots explaining predictions to clinicians

---

## 📚 Documentation

- **User Guide**: `docs/user_guide/`
- **API Documentation**: `docs/api/` (auto-generated from docstrings)
- **Technical Documentation**: `docs/technical/`
- **API Swagger UI**: http://localhost:8000/docs (when API is running)

To build docs:
```bash
make docs
```

To serve docs locally:
```bash
make docs-serve
```

---

## 🛠️ Development

### Project Phases

#### Phase 1: Foundation (Weeks 1-2) ✅
- [x] Project structure
- [x] Configuration management
- [x] Data loading and validation
- [ ] Feature engineering
- [ ] Statistical models
- [ ] Basic dashboard

#### Phase 2: Advanced Analytics (Weeks 3-4)
- [ ] ML models (RSF, GBM)
- [ ] DL models (LSTM, Transformer)
- [ ] REST API
- [ ] Model evaluation framework

#### Phase 3: Novel Features (Weeks 5-6)
- [ ] Counterfactual analysis
- [ ] Interpretability (SHAP, LIME, attention)
- [ ] Patient similarity engine
- [ ] Advanced visualizations

#### Phase 4: Polish & Presentation (Week 7)
- [ ] Report generation
- [ ] Dashboard enhancement
- [ ] Comprehensive documentation
- [ ] Jupyter notebooks

#### Phase 5: Testing & Validation (Week 8)
- [ ] Unit tests (>80% coverage)
- [ ] Integration tests
- [ ] External validation
- [ ] Presentation preparation

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and quality checks (`make all`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- Dataset: Bladder cancer recurrence data from medical research
- Inspiration: Combining classical biostatistics with modern AI for precision medicine
- Tools: Python ecosystem (pandas, scikit-learn, PyTorch, Streamlit, FastAPI)

---

## 📧 Contact

For questions, feedback, or collaboration:
- Open an issue on GitHub
- Email: [your.email@example.com]

---

## 🎯 Project Status

**Current Status**: Phase 1 - Foundation ✅
**Next Milestone**: Feature Engineering & Statistical Models

**Progress**:
- [x] Project structure and configuration
- [x] Data loading infrastructure
- [x] Multi-format data fusion
- [ ] Feature engineering pipeline
- [ ] Statistical models implementation
- [ ] Dashboard MVP

---

**Built with ❤️ for advancing precision medicine through AI**
