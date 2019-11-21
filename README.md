# DataScience_Python
Python in Data Science 
A curated list of awesome resources for practicing data science using Python, including not only libraries, but also links to tutorials, code snippets, blog posts and talks.

Core
pandas - Data structures built on top of numpy.
scikit-learn - Core ML library.
matplotlib - Plotting library.
seaborn - Data visualization library based on matplotlib.
pandas_summary - Basic statistics using DataFrameSummary(df).summary().
pandas_profiling - Descriptive statistics using ProfileReport.
sklearn_pandas - Helpful DataFrameMapper class.
missingno - Missing data visualization.

Pandas and Jupyter
General tricks: link
Python debugger (pdb) - blog post, video, cheatsheet
cookiecutter-data-science - Project template for data science projects.
nteract - Open Jupyter Notebooks with doubleclick.
swifter - Apply any function to a pandas dataframe faster.
xarray - Extends pandas to n-dimensional arrays.
blackcellmagic - Code formatting for jupyter notebooks.
pivottablejs - Drag n drop Pivot Tables and Charts for jupyter notebooks.
qgrid - Pandas DataFrame sorting.
ipysheet - Jupyter spreadsheet widget.
nbdime - Diff two notebook files, Alternative GitHub App: ReviewNB.
RISE - Turn Jupyter notebooks into presentations.
papermill - Parameterize and execute Jupyter notebooks, tutorial.
pixiedust - Helper library for Jupyter.
pandas_flavor - Write custom accessors like .str and .dt.
pandas-log - Find business logic issues and performance issues in pandas.

Helpful
tqdm - Progress bars for for-loops.
icecream - Simple debugging output.
pyprojroot - Helpful here() command from R.
intake - Loading datasets made easier, talk.

Extraction
textract - Extract text from any document.
camelot - Extract text from PDF.

Big Data
spark - DataFrame for big data, cheatsheet, tutorial.
sparkit-learn, spark-deep-learning - ML frameworks for spark.
koalas - Pandas API on Apache Spark.
dask, dask-ml - Pandas DataFrame for big data and machine learning library, resources, talk1, talk2, notebooks, videos.
dask-gateway - Managing dask clusters.
turicreate - Helpful SFrame class for out-of-memory dataframes.
modin - Parallelization library for faster pandas DataFrame.
h2o - Helpful H2OFrame class for out-of-memory dataframes.
datatable - Data Table for big data support.
cuDF - GPU DataFrame Library.
ray - Flexible, high-performance distributed execution framework.
mars - Tensor-based unified framework for large-scale data computation.
bottleneck - Fast NumPy array functions written in C.
bolz - A columnar data container that can be compressed.
cupy - NumPy-like API accelerated with CUDA.
vaex - Out-of-Core DataFrames.
petastorm - Data access library for parquet files by Uber.
zappy - Distributed numpy arrays.

Command line tools, CSV
ni - Command line tool for big data.
xsv - Command line tool for indexing, slicing, analyzing, splitting and joining CSV files.
csvkit - Another command line tool for CSV files.
csvsort - Sort large csv files.
tsv-utils - Tools for working with CSV files by ebay.
cheat - Make cheatsheets for command line commands.

Classical Statistics
researchpy - Helpful summary_cont() function for summary statistics (Table 1).
scikit-posthocs - Statistical post-hoc tests for pairwise multiple comparisons.
Bland-Altman Plot - Plot for agreement between two methods of measurement.

Tests
Blog post
scipy.stats - Statistical tests. ANOVA, Tutorials: One-way, Two-way, Type 1,2,3 explained.

Visualizations
Null Hypothesis Significance Testing (NHST) and Sample Size Calculation
Correlation
Cohen's d
Confidence Interval
Equivalence, non-inferiority and superiority testing
Bayesian two-sample t test
Distribution of p-values when comparing two groups
Understanding the t-distribution and its normal approximation

Talks
Inverse Propensity Weighting
Dealing with Selection Bias By Propensity Based Feature Selection

Exploration and Cleaning
Checklist.
janitor - Clean messy column names.
impyute - Imputations.
fancyimpute - Matrix completion and imputation algorithms.
imbalanced-learn - Resampling for imbalanced datasets.
tspreprocess - Time series preprocessing: Denoising, Compression, Resampling.
Kaggler - Utility functions (OneHotEncoder(min_obs=100))
pyupset - Visualizing intersecting sets.
pyemd - Earth Mover's Distance, similarity between histograms.

Train / Test Split
iterative-stratification - Stratification of multilabel data.

Feature Engineering
Talk
sklearn - Pipeline, examples.
pdpipe - Pipelines for DataFrames.
scikit-lego - Custom transformers for pipelines.
few - Feature engineering wrapper for sklearn.
skoot - Pipeline helper functions.
categorical-encoding - Categorical encoding of variables, vtreat (R package).
dirty_cat - Encoding dirty categorical variables.
patsy - R-like syntax for statistical models.
mlxtend - LDA.
featuretools - Automated feature engineering, example.
tsfresh - Time series feature engineering.
pypeln - Concurrent data pipelines.
feature_engine - Encoders, transformers, etc.

Feature Selection
Talk
Blog post series - 1, 2, 3, 4
Tutorials - 1, 2
sklearn - Feature selection.
eli5 - Feature selection using permutation importance.
scikit-feature - Feature selection algorithms.
stability-selection - Stability selection.
scikit-rebate - Relief-based feature selection algorithms.
scikit-genetic - Genetic feature selection.
boruta_py - Feature selection, explaination, example.
linselect - Feature selection package.
mlxtend - Exhaustive feature selection.
BoostARoota - Xgboost feature selection algorithm.

Dimensionality Reduction
Talk
prince - Dimensionality reduction, factor analysis (PCA, MCA, CA, FAMD).
sklearn - Multidimensional scaling (MDS).
sklearn - t-distributed Stochastic Neighbor Embedding (t-SNE), intro. Faster implementations: lvdmaaten, MulticoreTSNE.
sklearn - Truncated SVD (aka LSA).
mdr - Dimensionality reduction, multifactor dimensionality reduction (MDR).
umap - Uniform Manifold Approximation and Projection, talk, explorer, explanation.
FIt-SNE - Fast Fourier Transform-accelerated Interpolation-based t-SNE.
scikit-tda - Topological Data Analysis, paper, talk, talk.
ivis - Dimensionality reduction using Siamese Networks.
trimap - Dimensionality reduction using triplets.

Visualization
All charts, Austrian monuments.
cufflinks - Dynamic visualization library, wrapper for plotly, medium, example.
physt - Better histograms, talk, notebook.
matplotlib_venn - Venn diagrams, alternative.
joypy - Draw stacked density plots.
mosaic plots - Categorical variable visualization, example.
scikit-plot - ROC curves and other visualizations for ML models.
yellowbrick - Visualizations for ML models (similar to scikit-plot).
bokeh - Interactive visualization library, Examples, Examples.
animatplot - Animate plots build on matplotlib.
plotnine - ggplot for Python.
altair - Declarative statistical visualization library.
bqplot - Plotting library for IPython/Jupyter Notebooks.
hvplot - High-level plotting library built on top of holoviews.
dtreeviz - Decision tree visualization and model interpretation.
chartify - Generate charts.
VivaGraphJS - Graph visualization (JS package).
pm - Navigatable 3D graph visualization (JS package), example.
python-ternary - Triangle plots.
falcon - Interactive visualizations for big data.

Dashboards
dash - Dashboarding solution by plot.ly. Tutorial: 1, 2, 3, 4, 5, resources
panel - Dashboarding solution.
bokeh - Dashboarding solution.
visdom - Dashboarding library by facebook.
altair example - Video.
voila - Turn Jupyter notebooks into standalone web applications.
steamlit - Dashboards.

Geopraphical Tools
folium - Plot geographical maps using the Leaflet.js library, jupyter plugin.
gmaps - Google Maps for Jupyter notebooks.
stadiamaps - Plot geographical maps.
datashader - Draw millions of points on a map.
sklearn - BallTree, Example.
pynndescent - Nearest neighbor descent for approximate nearest neighbors.
geocoder - Geocoding of addresses, IP addresses.
Conversion of different geo formats: talk, repo
geopandas - Tools for geographic data
Low Level Geospatial Tools (GEOS, GDAL/OGR, PROJ.4)
Vector Data (Shapely, Fiona, Pyproj)
Raster Data (Rasterio)
Plotting (Descartes, Catropy)
Predict economic indicators from Open Street Map ipynb.
PySal - Python Spatial Analysis Library.
geography - Extract countries, regions and cities from a URL or text.

Recommender Systems
Examples: 1, 2, 2-ipynb, 3.
surprise - Recommender, talk.
turicreate - Recommender.
implicit - Fast Collaborative Filtering for Implicit Feedback Datasets.
spotlight - Deep recommender models using PyTorch.
lightfm - Recommendation algorithms for both implicit and explicit feedback.
funk-svd - Fast SVD.
pywFM - Factorization.

Decision Tree Models
Intro to Decision Trees and Random Forests, Intro to Gradient Boosting
lightgbm - Gradient boosting (GBDT, GBRT, GBM or MART) framework based on decision tree algorithms, doc.
xgboost - Gradient boosting (GBDT, GBRT or GBM) library, doc, Methods for CIs: link1, link2.
catboost - Gradient boosting.
thundergbm - GBDTs and Random Forest.
h2o - Gradient boosting.
forestci - Confidence intervals for random forests.
scikit-garden - Quantile Regression.
grf - Generalized random forest.
dtreeviz - Decision tree visualization and model interpretation.
Nuance - Decision tree visualization.
rfpimp - Feature Importance for RandomForests using Permuation Importance.
Why the default feature importance for random forests is wrong: link
treeinterpreter - Interpreting scikit-learn's decision tree and random forest predictions.
bartpy - Bayesian Additive Regression Trees.
infiniteboost - Combination of RFs and GBDTs.
merf - Mixed Effects Random Forest for Clustering, video
rrcf - Robust Random Cut Forest algorithm for anomaly detection on streams.

Natural Language Processing (NLP) / Text Processing
talk-nb, nb2, talk.
Text classification Intro, Preprocessing blog post.
gensim - NLP, doc2vec, word2vec, text processing, topic modelling (LSA, LDA), Example, Coherence Model for evaluation.
Embeddings - GloVe ([1], [2]), StarSpace, wikipedia2vec.
magnitude - Vector embedding utility package.
pyldavis - Visualization for topic modelling.
spaCy - NLP.
NTLK - NLP, helpful KMeansClusterer with cosine_distance.
pytext - NLP from Facebook.
fastText - Efficient text classification and representation learning.
annoy - Approximate nearest neighbor search.
faiss - Approximate nearest neighbor search.
pysparnn - Approximate nearest neighbor search.
infomap - Cluster (word-)vectors to find topics, example.
datasketch - Probabilistic data structures for large data (MinHash, HyperLogLog).
flair - NLP Framework by Zalando.
stanfordnlp - NLP Library.

Papers
Search Engine Correlation

Biology
Sequencing
scanpy - Analyze single-cell gene expression data, tutorial.

Image-related
mahotas - Image processing (Bioinformatics), example.
imagepy - Software package for bioimage analysis.
CellProfiler - Biological image analysis.
imglyb - Viewer for large images, talk, slides.
microscopium - Unsupervised clustering of images + viewer, talk.
cytokit - Analyzing properties of cells in fluorescent microscopy datasets.

Image Processing
Talk
cv2 - OpenCV, classical algorithms: Gaussian Filter, Morphological Transformations.
scikit-image - Image processing.

Neural Networks
Tutorials
Convolutional Neural Networks for Visual Recognition
fast.ai course - Lessons 1-7, Lessons 8-14
Tensorflow without a PhD - Neural Network course by Google.
Feature Visualization: Blog, PPT
Tensorflow Playground
Visualization of optimization algorithms

Image Related
imgaug - More sophisticated image preprocessing.
imgaug_extension - Extension for imgaug.
Augmentor - Image augmentation library.
keras preprocessing - Preprocess images.
albumentations - Wrapper around imgaug and other libraries.
cutouts-explorer - Image Viewer.

Text Related
ktext - Utilities for pre-processing text for deep learning in Keras.
textgenrnn - Ready-to-use LSTM for text generation.
ctrl - Text generation.

Libs
keras - Neural Networks on top of tensorflow, examples.
keras-contrib - Keras community contributions.
keras-tuner - Hyperparameter tuning for Keras.
hyperas - Keras + Hyperopt: Convenient hyperparameter optimization wrapper.
elephas - Distributed Deep learning with Keras & Spark.
tflearn - Neural Networks on top of tensorflow.
tensorlayer - Neural Networks on top of tensorflow, tricks.
tensorforce - Tensorflow for applied reinforcement learning.
fastai - Neural Networks in pytorch.
ignite - Highlevel library for pytorch.
skorch - Scikit-learn compatible neural network library that wraps pytorch, talk, slides.
autokeras - AutoML for deep learning.
PlotNeuralNet - Plot neural networks.
lucid - Neural network interpretability, Activation Maps.
tcav - Interpretability method.
AdaBound - Optimizer that trains as fast as Adam and as good as SGD, alt.
caffe - Deep learning framework, pretrained models.
foolbox - Adversarial examples that fool neural networks.
hiddenlayer - Training metrics.
imgclsmob - Pretrained models.
netron - Visualizer for deep learning and machine learning models.
torchcv - Deep Learning in Computer Vision.

Object detection
detectron2 - Object Detection (Mask R-CNN) by Facebook.
simpledet - Object Detection and Instance Recognition.
CenterNet - Object detection.
FCOS - Fully Convolutional One-Stage Object Detection.

Applications and Snippets
efficientnet - Promising neural network architecture.
CycleGAN and Pix2pix - Various image-to-image tasks.
SPADE - Semantic Image Synthesis.
Entity Embeddings of Categorical Variables, code, kaggle
Image Super-Resolution - Super-scaling using a Residual Dense Network.
Cell Segmentation - Talk, Blog Posts: 1, 2
deeplearning-models - Deep learning models.

GPU
cuML - Run traditional tabular ML tasks on GPUs.
thundergbm - GBDTs and Random Forest.
thundersvm - Support Vector Machines.

Regression
Understanding SVM Regression: slides, forum, paper

pyearth - Multivariate Adaptive Regression Splines (MARS), tutorial.
pygam - Generalized Additive Models (GAMs), Explanation.
GLRM - Generalized Low Rank Models.
tweedie - Specialized distribution for zero inflated targets, Talk.

Classification
Talk, Notebook
Blog post: Probability Scoring
All classification metrics
DESlib - Dynamic classifier and ensemble selection

Clustering
Overview of clustering algorithms applied image data (= Deep Clustering)
pyclustering - All sorts of clustering algorithms.
somoclu - Self-organizing map.
hdbscan - Clustering algorithm, talk.
nmslib - Similarity search library and toolkit for evaluation of k-NN methods.
buckshotpp - Outlier-resistant and scalable clustering algorithm.
merf - Mixed Effects Random Forest for Clustering, video

Interpretable Classifiers and Regressors
skope-rules - Interpretable classifier, IF-THEN rules.
sklearn-expertsys - Interpretable classifiers, Bayesian Rule List classifier.

Multi-label classification
scikit-multilearn - Multi-label classification, talk.

Signal Processing and Filtering
Stanford Lecture Series on Fourier Transformation, Youtube, Lecture Notes.
The Scientist & Engineer's Guide to Digital Signal Processing (1999).
Kalman Filter book - Focuses on intuition using Jupyter Notebooks. Includes Baysian and various Kalman filters.
Interactive Tool for FIR and IIR filters, Examples.
filterpy - Kalman filtering and optimal estimation library.

Time Series
statsmodels - Time series analysis, seasonal decompose example, SARIMA, granger causality.
pyramid, pmdarima - Wrapper for (Auto-) ARIMA.
pyflux - Time series prediction algorithms (ARIMA, GARCH, GAS, Bayesian).
prophet - Time series prediction library.
pm-prophet - Time series prediction and decomposition library.
htsprophet - Hierarchical Time Series Forecasting using Prophet.
nupic - Hierarchical Temporal Memory (HTM) for Time Series Prediction and Anomaly Detection.
tensorflow - LSTM and others, examples: link, link, link, Explain LSTM, seq2seq: 1, 2, 3, 4
tspreprocess - Preprocessing: Denoising, Compression, Resampling.
tsfresh - Time series feature engineering.
thunder - Data structures and algorithms for loading, processing, and analyzing time series data.
gatspy - General tools for Astronomical Time Series, talk.
gendis - shapelets, example.
tslearn - Time series clustering and classification, TimeSeriesKMeans, TimeSeriesKMeans.
pastas - Simulation of time series.
fastdtw - Dynamic Time Warp Distance.
fable - Time Series Forecasting (R package).
CausalImpact - Causal Impact Analysis (R package).
pydlm - Bayesian time series modeling (R package, Blog post)
PyAF - Automatic Time Series Forecasting.
luminol - Anomaly Detection and Correlation library from Linkedin.
matrixprofile-ts - Detecting patterns and anomalies, website, ppt, alternative.
stumpy - Another matrix profile library.
obspy - Seismology package. Useful classic_sta_lta function.
RobustSTL - Robust Seasonal-Trend Decomposition.
seglearn - Time Series library.
pyts - Time series transformation and classification, Imaging time series.
Turn time series into images and use Neural Nets: example, example.
sktime, sktime-dl - Toolbox for (deep) learning with time series.

Time Series Evaluation
TimeSeriesSplit - Sklearn time series split.
tscv - Evaluation with gap.

Financial Data
pyfolio - Portfolio and risk analytics.
zipline - Algorithmic trading.
alphalens - Performance analysis of predictive stock factors.
stockstats - Pandas DataFrame wrapper for working with stock data.
pandas-datareader - Read stock data.

Survival Analysis
Time-dependent Cox Model in R.
lifelines - Survival analysis, Cox PH Regression, talk, talk2.
scikit-survival - Survival analysis.
xgboost - "objective": "survival:cox" NHANES example
survivalstan - Survival analysis, intro.
convoys - Analyze time lagged conversions.
RandomSurvivalForests (R packages: randomForestSRC, ggRandomForests).

Outlier Detection & Anomaly Detection
sklearn - Isolation Forest and others.
pyod - Outlier Detection / Anomaly Detection.
eif - Extended Isolation Forest.
AnomalyDetection - Anomaly detection (R package).
luminol - Anomaly Detection and Correlation library from Linkedin.
Distances for comparing histograms and detecting outliers - Talk: Kolmogorov-Smirnov, Wasserstein, Energy Distance (Cramer), Kullback-Leibler divergence.
banpei - Anomaly detection library based on singular spectrum transformation.
telemanom - Detect anomalies in multivariate time series data using LSTMs.

Ranking
lightning - Large-scale linear classification, regression and ranking.

Scoring
SLIM - Scoring systems for classification, Supersparse linear integer models.

Probabilistic Modeling and Bayes
Intro, Guide
PyMC3 - Baysian modelling, intro
pomegranate - Probabilistic modelling, talk.
pmlearn - Probabilistic machine learning.
arviz - Exploratory analysis of Bayesian models.
zhusuan - Bayesian deep learning, generative models.
dowhy - Estimate causal effects.
edward - Probabilistic modeling, inference, and criticism, Mixture Density Networks (MNDs), MDN Explanation.
Pyro - Deep Universal Probabilistic Programming.
tensorflow probability - Deep learning and probabilistic modelling, talk, example.
bambi - High-level Bayesian model-building interface on top of PyMC3.

Stacking Models and Ensembles
Model Stacking Blog Post
mlxtend - EnsembleVoteClassifier, StackingRegressor, StackingCVRegressor for model stacking.
vecstack - Stacking ML models.
StackNet - Stacking ML models.
mlens - Ensemble learning.

Model Evaluation
pycm - Multi-class confusion matrix.
pandas_ml - Confusion matrix.
Plotting learning curve: link.
yellowbrick - Learning curve.

Model Explanation, Interpretability, Feature Importance
Book, Examples
shap - Explain predictions of machine learning models, talk.
treeinterpreter - Interpreting scikit-learn's decision tree and random forest predictions.
lime - Explaining the predictions of any machine learning classifier, talk, Warning (Myth 7).
lime_xgboost - Create LIMEs for XGBoost.
eli5 - Inspecting machine learning classifiers and explaining their predictions.
lofo-importance - Leave One Feature Out Importance, talk, examples: 1, 2, 3.
pybreakdown - Generate feature contribution plots.
FairML - Model explanation, feature importance.
pycebox - Individual Conditional Expectation Plot Toolbox.
pdpbox - Partial dependence plot toolbox, example.
partial_dependence - Visualize and cluster partial dependence.
skater - Unified framework to enable model interpretation.
anchor - High-Precision Model-Agnostic Explanations for classifiers.
l2x - Instancewise feature selection as methodology for model interpretation.
contrastive_explanation - Contrastive explanations.
DrWhy - Collection of tools for explainable AI.
lucid - Neural network interpretability.
xai - An eXplainability toolbox for machine learning.
innvestigate - A toolbox to investigate neural network predictions.
dalex - Explanations for ML models (R package).
interpret - Fit interpretable models, explain models (Microsoft).

Automated Machine Learning
AdaNet - Automated machine learning based on tensorflow.
tpot - Automated machine learning tool, optimizes machine learning pipelines.
auto_ml - Automated machine learning for analytics & production.
autokeras - AutoML for deep learning.
nni - Toolkit for neural architecture search and hyper-parameter tuning by Microsoft.
automl-gs - Automated machine learning.
mljar - Automated machine learning.

Evolutionary Algorithms & Optimization
deap - Evolutionary computation framework (Genetic Algorithm, Evolution strategies).
evol - DSL for composable evolutionary algorithms, talk.
platypus - Multiobjective optimization.
autograd - Efficiently computes derivatives of numpy code.
nevergrad - Derivation-free optimization.
gplearn - Sklearn-like interface for genetic programming.
blackbox - Optimization of expensive black-box functions.
Optometrist algorithm - paper.
DeepSwarm - Neural architecture search.

Hyperparameter Tuning
sklearn - GridSearchCV, RandomizedSearchCV.
sklearn-deap - Hyperparameter search using genetic algorithms.
hyperopt - Hyperparameter optimization.
hyperopt-sklearn - Hyperopt + sklearn.
optuna - Hyperparamter optimization, Talk.
skopt - BayesSearchCV for Hyperparameter search.
tune - Hyperparameter search with a focus on deep learning and deep reinforcement learning.
hypergraph - Global optimization methods and hyperparameter optimization.
bbopt - Black box hyperparameter optimization.
dragonfly - Scalable Bayesian optimisation.

Incremental Learning, Online Learning
sklearn - PassiveAggressiveClassifier, PassiveAggressiveRegressor.
creme-ml - Incremental learning framework, talk.
Kaggler - Online Learning algorithms.

Active Learning
Talk
modAL - Active learning framework.

Reinforcement Learning
YouTube, YouTube
Intro to Monte Carlo Tree Search (MCTS) - 1, 2, 3
AlphaZero methodology - 1, 2, 3, Cheat Sheet
RLLib - Library for reinforcement learning.
Horizon - Facebook RL framework.

Frameworks
h2o - Scalable machine learning.
turicreate - Apple Machine Learning Toolkit.
astroml - ML for astronomical data.

Deployment and Lifecycle Management
Dependency Management
pipreqs - Generate a requirements.txt from import statements.
pyup - Python dependency management.
pypi-timemachine - Install packages with pip as if you were in the past.
[pypi2nix] - Fix package versions and create reproducible environments, Talk.

Data Science Related
m2cgen - Transpile trained ML models into other languages.
sklearn-porter - Transpile trained scikit-learn estimators to C, Java, JavaScript and others.
mlflow - Manage the machine learning lifecycle, including experimentation, reproducibility and deployment.
modelchimp - Experiment Tracking.
skll - Command-line utilities to make it easier to run machine learning experiments.
BentoML - Package and deploy machine learning models for serving in production.
dvc - Versioning for ML projects.
dagster - Tool with focus on dependency graphs.
knockknock - Be notified when your training ends.

Math and Background
Gilbert Strang - Linear Algebra
Gilbert Strang - Matrix Methods in Data Analysis, Signal Processing, and Machine Learning

Other
daft - Render probabilistic graphical models using matplotlib.
unyt - Working with units.
scrapy - Web scraping library.
VowpalWabbit - ML Toolkit from Microsoft.
metric-learn - Metric learning.

General Python Programming
more_itertools - Extension of itertools.
funcy - Fancy and practical functional tools.
dateparser - A better date parser.
jellyfish - Approximate string matching.
coloredlogs - Colored logging output.

Resources
Distill.pub - Blog. Machine Learning Videos
Data Science Notebooks
Recommender Systems (Microsoft)
The GAN Zoo - List of Generative Adversarial Networks
Datascience Cheatsheets

Other Awesome Lists
Awesome Adversarial Machine Learning
Awesome AI Booksmarks
Awesome AI on Kubernetes
Awesome Big Data
Awesome Business Machine Learning
Awesome Causality
Awesome CSV
Awesome Data Science with Ruby
Awesome Dash
Awesome Deep Learning
Awesome ETL
Awesome Financial Machine Learning
Awesome GAN Applications
Awesome Machine Learning
Awesome Machine Learning Interpretability
Awesome Machine Learning Operations
Awesome Online Machine Learning
Awesome Python
Awesome Python Data Science
Awesome Python Data Science
Awesome Pytorch
Awesome Recommender Systems
Awesome Semantic Segmentation
Awesome Sentence Embedding
Awesome Time Series
Awesome Time Series Anomaly Detection
