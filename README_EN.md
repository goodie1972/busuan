# busuan

A sophisticated machine learning application for lottery number prediction. This project supports two major lottery types: **SSQ (Double Color Ball)** and **DLT (Big Lotto)**, using advanced machine learning and AI techniques to generate predictions based on historical data patterns.

## Latest Updates

### March 25, 2025 Update
- Added dedicated tab for Expected Value Model with enhanced visualization of prediction steps
- Improved model training process with better error handling and progress reporting

### March 23, 2025 Update
- Added support for multiple machine learning models including XGBoost, Random Forest, GBDT, LightGBM and CatBoost
- Enhanced UI with model selection dropdown
- Improved data processing with sliding window features

### March 22, 2025 Update
- Enhanced UI with standardized styling and optimized window dimensions (960x680 pixels)
- Implemented tabbed interface separating prediction and analysis functions
- Performance optimizations including vectorized operations
- Added support for high DPI displays with improved chart rendering

## Installation Guide

### Requirements
- Python 3.9 or higher
- PyTorch (with optional GPU support)
- PyQt5
- pandas, numpy, matplotlib, seaborn
- scikit-learn, xgboost
- torchcrf
- joblib
- Optional: lightgbm, catboost

### Basic Installation
```bash
git clone https://github.com/goodie1972/busuan.git
cd busuan
pip install -r requirements.txt
```

### Manual Installation of TorchCRF (if pip install fails)
```bash
git clone https://github.com/kmkurn/pytorch-crf.git
cd pytorch-crf
python setup.py install
```

## Project Structure

```
busuan/
├── main.py                         # Main entry point, supports command line arguments
├── lottery_predictor_app_new.py    # Main application with complete GUI
├── ui_components.py                # UI components and layout functions
├── thread_utils.py                 # Threading tools for background model training
├── model_utils.py                  # Model loading and prediction utilities
├── prediction_utils.py             # Number generation and prediction tools
├── data_processing.py              # Data processing module with statistical analysis
├── ml_models.py                    # ML model implementations (XGBoost, Random Forest, etc.)
├── expected_value_model.py         # Game theory based expected value model
├── fetch_and_train.py              # Data fetching and model training script
├── train_models.py                 # Command-line model training script
├── theme_manager.py                # UI theme manager with dark/light mode support
├── model.py                        # Neural network model definitions
├── scripts/                        # Analysis and utility scripts
├── model/                          # Trained model storage
├── data/                           # Data storage
├── update/                         # Update documentation
└── requirements.txt                # Project dependencies
```

## Key Features

### Multiple Prediction Models
- **LSTM-CRF (Long Short-Term Memory with Conditional Random Fields)**: Core neural network architecture for sequence modeling of lottery numbers
- **Expected Value Model**: Game theory based model using probability distributions and expected value calculations
- **Traditional Machine Learning Models**:
  - XGBoost
  - Random Forest
  - Gradient Boosting Decision Trees (GBDT)
  - LightGBM and CatBoost (optional dependencies)
- **Ensemble Learning**: Combines predictions from multiple models for improved accuracy

### Advanced Data Processing
- **Time Series Analysis**: Sliding window features to capture temporal patterns
- **Feature Engineering**: Comprehensive statistics including frequency, gap analysis, and combination patterns
- **Data Standardization**: Automatic feature scaling for improved model stability
- **Memoization**: Caching mechanism for faster repeated analysis operations

### Intelligent Prediction Pipeline
- **Normal Distribution Randomization**: Enhanced randomization using normal distribution instead of uniform distribution
- **Temperature Sampling**: Parameter to control diversity vs. accuracy tradeoff
- **Top-K Selection**: Advanced selection method for better prediction diversity
- **GPU Acceleration**: Support for GPU-accelerated training and inference

### Interactive User Interface
- **Multi-tab Design**: Separate tabs for different prediction models and analysis tools
- **Real-time Logging**: Detailed process information displayed during training and prediction
- **Visualization Tools**: Statistical charts and graphs for historical data analysis
- **Theming Support**: Light and dark mode with customizable colors

## Usage Guide

### Quick Start
Simply run the main script:
```bash
python main.py
```

The application will automatically handle data fetching, model training, and resource initialization.

### Model Training
1. Select the lottery type (SSQ or DLT)
2. Choose the desired prediction model from the dropdown menu
3. Click "Train Model" to start the training process
4. View training progress in the log display

### Generating Predictions
1. Select the lottery type and prediction model
2. Input feature values or use default settings
3. Select the number of predictions to generate
4. Click "Generate Predictions" to create lottery number predictions
5. Review results in the prediction display area

### Data Analysis
1. Switch to the "Data Analysis" tab
2. Select analysis type and parameters
3. View statistical information and visualizations
4. Save or export analysis results if needed

## Technical Implementation

### Deep Learning & ML Framework
- **PyTorch**: Core deep learning framework for building and training neural networks
- **TorchCRF**: Conditional Random Fields implementation for sequence labeling tasks
- **Scikit-learn**: For traditional machine learning algorithms and preprocessing
- **XGBoost/LightGBM/CatBoost**: High-performance gradient boosting frameworks

### UI Framework
- **PyQt5**: Modern UI toolkit for desktop application development
- **Matplotlib/Seaborn**: Data visualization libraries for statistical plotting

### Data Processing
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing and array operations
- **Joblib**: Model serialization and parallelism

### Game Theory Applications
The Expected Value Model applies game theory concepts:
- Expected value strategy for optimal decision making under uncertainty
- Risk vs. reward balancing for number selection
- Mixed strategy implementation via temperature parameters
- Advantage principle for number combination valuation

## Screenshots
![Analysis Interface](https://github.com/user-attachments/assets/4e986db9-83fd-4650-92c2-b91d4dbbcb41)
![Prediction Interface](https://github.com/user-attachments/assets/b2ce1770-b6fc-4df2-b2c5-f14b008ef42f)

## Error Handling and Logging
- Data fetching scripts include network request exception handling and detailed logging
- Training scripts provide checkpoints for resuming from interrupted sessions
- The main application captures exceptions and displays error information in the log display
- Detailed diagnostic information helps users troubleshoot problems

## Performance Optimization
- Vectorized operations in pandas and numpy for improved performance
- Efficient data structures (e.g., categorical data types for low-cardinality columns)
- Code profiling to identify and optimize bottlenecks
- GPU acceleration for neural network training and inference

## Contributing
Contributions to busuan are welcome. Please feel free to submit pull requests or open issues for bugs and feature requests.

## License
This project is licensed under the MIT License - see the LICENSE file for details. 