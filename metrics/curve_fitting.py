"""
Curve fitting analysis for video-language model metrics.

Implements linear and quadratic model fitting with statistical comparison methods
to analyze how metrics scale with video length (duration).
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
from scipy import stats
from scipy.optimize import minimize, curve_fit
from sklearn.metrics import r2_score
import warnings

@dataclass
class CurveFitResult:
    """Results from curve fitting analysis."""
    model_type: str  # 'linear' or 'quadratic'
    coefficients: np.ndarray  # Model coefficients
    r_squared: float  # Coefficient of determination
    adjusted_r_squared: float  # Adjusted R²
    aic: float  # Akaike Information Criterion
    bic: float  # Bayesian Information Criterion
    mse: float  # Mean squared error
    equation: str  # Human-readable equation
    n_params: int  # Number of parameters
    n_observations: int  # Number of data points

@dataclass
class ComparisonResult:
    """Comparison between linear and quadratic models."""
    better_model: str  # 'linear' or 'quadratic'
    linear_result: CurveFitResult
    quadratic_result: CurveFitResult
    f_test_statistic: float
    f_test_p_value: float
    improvement_significant: bool  # Whether quadratic improvement is significant
    recommendation: str  # Human-readable recommendation


@dataclass
class ExponentialFitResult:
    """Result from fitting an exponential decay model."""
    model_type: str  # fixed to 'exponential_decay'
    coefficients: np.ndarray  # [a, b, c]
    r_squared: float
    adjusted_r_squared: float
    aic: float
    bic: float
    mse: float
    equation: str
    n_params: int
    n_observations: int

def fit_linear_model(x: np.ndarray, y: np.ndarray) -> CurveFitResult:
    """
    Fit linear model: y = ax + b

    Args:
        x: Independent variable (video duration)
        y: Dependent variable (metric values)

    Returns:
        CurveFitResult with linear model statistics
    """
    if len(x) < 2:
        raise ValueError("Need at least 2 data points for linear fitting")

    # Add intercept term
    X = np.column_stack([x, np.ones(len(x))])

    # Constrained least squares: minimize ||Xβ - y||² subject to β[0] ≥ 0
    def objective(coeffs):
        return np.sum((X @ coeffs - y) ** 2)

    # Initial guess from unconstrained solution
    initial_coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

    # Constraint: coefficient on x must be non-negative
    constraints = {'type': 'ineq', 'fun': lambda coeffs: coeffs[0]}

    # Solve constrained optimization
    result = minimize(objective, initial_coeffs, method='SLSQP', constraints=constraints)

    if result.success:
        coeffs = result.x
        # Clean up numerical precision issues
        if abs(coeffs[0]) < 1e-10:
            coeffs[0] = 0.0
    else:
        # Fall back to unconstrained if optimization fails
        coeffs = initial_coeffs
        # Force constraint if violated
        if coeffs[0] < 0:
            coeffs = np.array([0, np.mean(y)])

    # Predictions
    y_pred = X @ coeffs

    # Calculate statistics
    n = len(y)
    k = 2  # Number of parameters (slope + intercept)

    mse = np.mean((y - y_pred) ** 2)
    r_squared = r2_score(y, y_pred)
    # Protect against division by zero when n = k (perfect fit case)
    if n > k:
        adjusted_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - k)
    else:
        adjusted_r_squared = r_squared  # Use regular R² when n = k

    # AIC and BIC (protect against zero MSE)
    if mse <= 0:
        mse = 1e-10  # Small positive value to avoid log(0)

    log_likelihood = -n/2 * np.log(2 * np.pi * mse) - n/2
    aic = 2 * k - 2 * log_likelihood
    bic = k * np.log(n) - 2 * log_likelihood

    # Human-readable equation with regular numbers
    equation = f"y = {coeffs[0]:.6f}x + {coeffs[1]:.6f}"

    return CurveFitResult(
        model_type='linear',
        coefficients=coeffs,
        r_squared=r_squared,
        adjusted_r_squared=adjusted_r_squared,
        aic=aic,
        bic=bic,
        mse=mse,
        equation=equation,
        n_params=k,
        n_observations=n
    )

def fit_quadratic_model(x: np.ndarray, y: np.ndarray) -> CurveFitResult:
    """
    Fit quadratic model: y = cx² + dx + e

    Args:
        x: Independent variable (video duration)
        y: Dependent variable (metric values)

    Returns:
        CurveFitResult with quadratic model statistics
    """
    if len(x) < 3:
        raise ValueError("Need at least 3 data points for quadratic fitting")

    # Add quadratic and intercept terms
    X = np.column_stack([x**2, x, np.ones(len(x))])

    # Constrained least squares: minimize ||Xβ - y||² subject to β[0] > 0
    def objective(coeffs):
        return np.sum((X @ coeffs - y) ** 2)

    # Initial guess from unconstrained solution
    initial_coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

    # Constraint: coefficient on x² must be positive
    constraints = {'type': 'ineq', 'fun': lambda coeffs: coeffs[0] - 1e-8}  # Enforce > 0

    # Solve constrained optimization
    result = minimize(objective, initial_coeffs, method='SLSQP', constraints=constraints)

    if result.success:
        coeffs = result.x
        # Clean up numerical precision issues and enforce minimum positive value
        if coeffs[0] < 1e-8:
            coeffs[0] = 1e-8
    else:
        # Fall back to constrained linear model if optimization fails
        X_linear = np.column_stack([x, np.ones(len(x))])
        linear_result = fit_linear_model(x, y)  # This already has constraints
        coeffs = np.array([1e-8, linear_result.coefficients[0], linear_result.coefficients[1]])

    # Predictions
    y_pred = X @ coeffs

    # Calculate statistics
    n = len(y)
    k = 3  # Number of parameters (quadratic + linear + intercept)

    mse = np.mean((y - y_pred) ** 2)
    r_squared = r2_score(y, y_pred)
    # Protect against division by zero when n = k (perfect fit case)
    if n > k:
        adjusted_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - k)
    else:
        adjusted_r_squared = r_squared  # Use regular R² when n = k

    # AIC and BIC (protect against zero MSE)
    if mse <= 0:
        mse = 1e-10  # Small positive value to avoid log(0)

    log_likelihood = -n/2 * np.log(2 * np.pi * mse) - n/2
    aic = 2 * k - 2 * log_likelihood
    bic = k * np.log(n) - 2 * log_likelihood

    # Human-readable equation with regular numbers
    equation = f"y = {coeffs[0]:.6f}x² + {coeffs[1]:.6f}x + {coeffs[2]:.6f}"

    return CurveFitResult(
        model_type='quadratic',
        coefficients=coeffs,
        r_squared=r_squared,
        adjusted_r_squared=adjusted_r_squared,
        aic=aic,
        bic=bic,
        mse=mse,
        equation=equation,
        n_params=k,
        n_observations=n
    )


def fit_exponential_decay_model(x: np.ndarray, y: np.ndarray) -> ExponentialFitResult:
    """Fit y = a * exp(-b * x) + c to data constrained to [0, 1]."""
    if len(x) < 3:
        raise ValueError("Need at least 3 data points for exponential fitting")

    def decay_fn(x_vals, a, b, c):
        return a * np.exp(-b * x_vals) + c

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # Initial guesses tailored to 0/1 classification style data
    y_min, y_max = float(np.min(y)), float(np.max(y))
    amplitude_guess = y_max - y_min
    if amplitude_guess <= 0:
        amplitude_guess = 0.1
    rate_guess = 1.0 / max(float(np.max(x)), 1.0)
    offset_guess = y_min

    bounds_lower = (-1.0, 0.0, -0.5)
    bounds_upper = (1.5, 5.0, 1.5)

    params, _ = curve_fit(
        decay_fn,
        x,
        y,
        p0=[amplitude_guess, rate_guess, offset_guess],
        bounds=(bounds_lower, bounds_upper),
        maxfev=20000,
    )

    a, b, c = params
    y_pred = decay_fn(x, a, b, c)

    n = len(y)
    k = 3
    residuals = y - y_pred
    mse = float(np.mean(residuals ** 2))

    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    ss_res = float(np.sum(residuals ** 2))
    if ss_tot == 0:
        r_squared = 0.0
    else:
        r_squared = 1 - (ss_res / ss_tot)

    if n > k:
        adjusted_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - k)
    else:
        adjusted_r_squared = r_squared

    if mse <= 0:
        mse = 1e-10

    log_likelihood = -n / 2 * np.log(2 * np.pi * mse) - n / 2
    aic = 2 * k - 2 * log_likelihood
    bic = k * np.log(n) - 2 * log_likelihood

    equation = f"y = {a:.6f}·exp(-{b:.6f}·x) + {c:.6f}"

    return ExponentialFitResult(
        model_type='exponential_decay',
        coefficients=params,
        r_squared=r_squared,
        adjusted_r_squared=adjusted_r_squared,
        aic=aic,
        bic=bic,
        mse=mse,
        equation=equation,
        n_params=k,
        n_observations=n,
    )

def f_test_nested_models(linear_result: CurveFitResult, quadratic_result: CurveFitResult) -> Tuple[float, float]:
    """
    F-test for comparing nested models (linear vs quadratic).

    Tests whether the quadratic term significantly improves the fit.

    Args:
        linear_result: Results from linear model fitting
        quadratic_result: Results from quadratic model fitting

    Returns:
        Tuple of (F-statistic, p-value)
    """
    if linear_result.n_observations != quadratic_result.n_observations:
        raise ValueError("Models must be fit on the same data")

    n = linear_result.n_observations

    # Calculate sum of squared errors for each model
    sse_linear = linear_result.mse * n
    sse_quadratic = quadratic_result.mse * n

    # Degrees of freedom
    df_linear = n - linear_result.n_params
    df_quadratic = n - quadratic_result.n_params
    df_diff = df_linear - df_quadratic  # Should be 1 for linear vs quadratic

    # F-test for additional terms
    if df_diff <= 0 or sse_quadratic >= sse_linear or df_quadratic <= 0:
        # No improvement, worse fit, or insufficient degrees of freedom
        return 0.0, 1.0

    f_statistic = ((sse_linear - sse_quadratic) / df_diff) / (sse_quadratic / df_quadratic)
    p_value = 1 - stats.f.cdf(f_statistic, df_diff, df_quadratic)

    return f_statistic, p_value

def compare_models(x: np.ndarray, y: np.ndarray, alpha: float = 0.05) -> ComparisonResult:
    """
    Compare linear and quadratic models and recommend the best one.

    Args:
        x: Independent variable (video duration)
        y: Dependent variable (metric values)
        alpha: Significance level for F-test (default 0.05)

    Returns:
        ComparisonResult with recommendation
    """
    # Fit both models
    linear_result = fit_linear_model(x, y)
    quadratic_result = fit_quadratic_model(x, y)

    # F-test for nested models
    f_stat, f_p_value = f_test_nested_models(linear_result, quadratic_result)
    improvement_significant = f_p_value < alpha

    # Model selection criteria
    aic_prefers_quadratic = quadratic_result.aic < linear_result.aic
    bic_prefers_quadratic = quadratic_result.bic < linear_result.bic
    adj_r2_prefers_quadratic = quadratic_result.adjusted_r_squared > linear_result.adjusted_r_squared

    # Decision logic
    if improvement_significant:
        better_model = 'quadratic'
        recommendation = f"Quadratic model recommended (F-test p={f_p_value:.4f} < {alpha})"
    elif aic_prefers_quadratic and adj_r2_prefers_quadratic:
        better_model = 'quadratic'
        recommendation = "Quadratic model recommended (better AIC and adjusted R²)"
    elif bic_prefers_quadratic and adj_r2_prefers_quadratic:
        better_model = 'quadratic'
        recommendation = "Quadratic model recommended (better BIC and adjusted R²)"
    else:
        better_model = 'linear'
        recommendation = "Linear model recommended (simpler model preferred)"

    return ComparisonResult(
        better_model=better_model,
        linear_result=linear_result,
        quadratic_result=quadratic_result,
        f_test_statistic=f_stat,
        f_test_p_value=f_p_value,
        improvement_significant=improvement_significant,
        recommendation=recommendation
    )

def analyze_metric_scaling(durations: List[float], metric_values: List[float],
                          metric_name: str = "metric") -> Optional[ComparisonResult]:
    """
    Analyze how a metric scales with video duration.

    Args:
        durations: List of video durations (in seconds)
        metric_values: List of corresponding metric values
        metric_name: Name of the metric for reporting

    Returns:
        ComparisonResult if successful, None if insufficient data
    """
    if len(durations) != len(metric_values):
        raise ValueError("Durations and metric values must have same length")

    if len(durations) < 3:
        print(f"⚠️  {metric_name}: Need at least 3 data points, got {len(durations)}")
        return None

    # Extract numeric values from potentially mixed data types
    def extract_numeric_value(val):
        """Extract numeric value from various data types."""
        if isinstance(val, dict):
            # For FLOPS dictionaries, extract total_flops
            return val.get('total_flops', 0)
        elif isinstance(val, (int, float)):
            return float(val)
        else:
            # Try to convert to float, return 0 if failed
            try:
                return float(val)
            except (ValueError, TypeError):
                return 0.0

    # Remove any NaN or infinite values
    valid_indices = []
    for i, (dur, val) in enumerate(zip(durations, metric_values)):
        numeric_dur = extract_numeric_value(dur)
        numeric_val = extract_numeric_value(val)

        if (np.isfinite(numeric_dur) and np.isfinite(numeric_val) and
            numeric_dur >= 0 and numeric_val >= 0):
            valid_indices.append(i)

    if len(valid_indices) < 3:
        print(f"⚠️  {metric_name}: Insufficient valid data points after filtering")
        return None

    # Filter to valid data and extract numeric values
    x = np.array([extract_numeric_value(durations[i]) for i in valid_indices])
    y = np.array([extract_numeric_value(metric_values[i]) for i in valid_indices])

    # Check for zero variance
    if np.var(y) == 0:
        print(f"⚠️  {metric_name}: No variance in metric values")
        return None

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return compare_models(x, y)
    except Exception as e:
        print(f"⚠️  {metric_name}: Curve fitting failed - {e}")
        return None


def analyze_exponential_metric(
    durations: List[float],
    metric_values: List[float],
    metric_name: str,
    window_seconds: Optional[float] = None,
) -> Optional[ExponentialFitResult]:
    if len(durations) != len(metric_values):
        raise ValueError("Durations and metric values must have same length")

    if len(durations) < 3:
        print(f"⚠️  {metric_name}: Need at least 3 data points, got {len(durations)}")
        return None

    durations_arr = np.asarray(durations, dtype=float)
    values_arr = np.asarray(metric_values, dtype=float)

    if window_seconds is not None:
        mask = durations_arr <= float(window_seconds)
        if mask.sum() < 3:
            print(f"⚠️  {metric_name} (≤{window_seconds}s): Insufficient data points for exponential fit")
            return None
        durations_arr = durations_arr[mask]
        values_arr = values_arr[mask]

    if np.var(values_arr) == 0:
        print(f"⚠️  {metric_name}: No variance in metric values")
        return None

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return fit_exponential_decay_model(durations_arr, values_arr)
    except Exception as exc:  # pragma: no cover - diagnostic path
        scope = f" (≤{window_seconds}s)" if window_seconds is not None else ""
        print(f"⚠️  {metric_name}{scope}: Exponential fit failed - {exc}")
        return None

def extract_video_durations_from_metrics(metrics, operation_type: str) -> List[float]:
    """
    Extract actual video timestamps from metrics data. NO FALLBACKS OR ESTIMATES.

    Args:
        metrics: PerformanceMetrics object
        operation_type: 'add_video', 'add_text', or 'ask_question'

    Returns:
        List of actual video timestamps (in seconds) corresponding to each metric measurement

    Raises:
        ValueError: If real timestamp data is not available
    """
    timestamp_attr = f'video_timestamps_{operation_type}'

    # Must have actual video timestamps for this operation type
    if not hasattr(metrics, timestamp_attr):
        raise ValueError(f"Real timestamp data not available for {operation_type} operations. "
                       f"{timestamp_attr} attribute missing from metrics.")

    timestamps = getattr(metrics, timestamp_attr)
    if not timestamps:
        raise ValueError(f"Real timestamp data not available for {operation_type} operations. "
                       f"{timestamp_attr} is empty.")

    return timestamps

def normalize_metric_values(metric_values: List) -> List[float]:
    """
    Normalize metric values to handle mixed types (int/float or dict with 'total_flops').

    Args:
        metric_values: List that may contain mixed types

    Returns:
        List of float values
    """
    normalized = []
    for value in metric_values:
        if isinstance(value, dict):
            # Extract 'total_flops' or use first numeric value found
            if 'total_flops' in value:
                normalized.append(float(value['total_flops']))
            else:
                # Find first numeric value in dict
                numeric_val = 0
                for v in value.values():
                    if isinstance(v, (int, float)):
                        numeric_val = float(v)
                        break
                normalized.append(numeric_val)
        else:
            normalized.append(float(value))
    return normalized

def analyze_all_metrics(metrics, print_results: bool = True, label: Optional[str] = None) -> Dict[str, ComparisonResult]:
    """
    Analyze curve fitting for all available metrics.

    Args:
        metrics: PerformanceMetrics object
        print_results: Whether to print human-readable results

    Returns:
        Dictionary mapping metric names to ComparisonResult objects
    """
    results = {}

    # Define metrics to analyze with their operation types
    metric_definitions = [
        ('latency_add_video', 'add_video', 'Add Video Latency (seconds)'),
        ('latency_add_text', 'add_text', 'Add Text Latency (seconds)'),
        ('latency_ask_question', 'ask_question', 'Ask Question Latency (seconds)'),
        ('flops_add_video', 'add_video', 'Add Video FLOPs'),
        ('flops_add_text', 'add_text', 'Add Text FLOPs'),
        ('flops_ask_question', 'ask_question', 'Ask Question FLOPs'),
        ('peak_gpu_mem_increase_add_video', 'add_video', 'Add Video Peak GPU Memory (MB)'),
        ('peak_gpu_mem_increase_add_text', 'add_text', 'Add Text Peak GPU Memory (MB)'),
        ('peak_gpu_mem_increase_ask_question', 'ask_question', 'Ask Question Peak GPU Memory (MB)'),
        ('peak_gpu_mem_absolute_add_video', 'add_video', 'Add Video Absolute Peak GPU Memory (MB)'),
        ('peak_gpu_mem_absolute_add_text', 'add_text', 'Add Text Absolute Peak GPU Memory (MB)'),
        ('peak_gpu_mem_absolute_ask_question', 'ask_question', 'Ask Question Absolute Peak GPU Memory (MB)'),
        ('state_memory_delta_add_video', 'add_video', 'Add Video State Δ (floats)'),
        ('state_memory_delta_add_text', 'add_text', 'Add Text State Δ (floats)'),
        ('state_memory_delta_ask_question', 'ask_question', 'Ask Question State Δ (floats)'),
    ]

    for metric_attr, operation_type, display_name in metric_definitions:
        if hasattr(metrics, metric_attr):
            metric_values_raw = getattr(metrics, metric_attr)
            if metric_values_raw:  # Only analyze if we have data
                # Normalize mixed types to handle int/float and dict formats
                metric_values = normalize_metric_values(metric_values_raw)
                try:
                    durations = extract_video_durations_from_metrics(metrics, operation_type)
                except ValueError:
                    continue
                if len(durations) == len(metric_values):
                    result = analyze_metric_scaling(durations, metric_values, display_name)
                    if result:
                        results[metric_attr] = result

    question_metrics = {
        'question_correctness_rate': 'Question Correctness Rate',
        'question_dont_know_rate': "Question Don't-Know Rate",
    }

    for metric_attr, display_name in question_metrics.items():
        metric_values_raw = getattr(metrics, metric_attr, None)
        if not metric_values_raw:
            continue
        try:
            durations = extract_video_durations_from_metrics(metrics, 'question_outcome')
        except ValueError:
            continue
        if len(durations) != len(metric_values_raw):
            continue

        metric_values = normalize_metric_values(metric_values_raw)
        full_result = analyze_exponential_metric(durations, metric_values, display_name)
        window_result = analyze_exponential_metric(
            durations,
            metric_values,
            f"{display_name} (≤300s)",
            window_seconds=300.0,
        )

        if full_result is not None:
            results[f"{metric_attr}_exponential_full"] = full_result
        if window_result is not None:
            results[f"{metric_attr}_exponential_first_300s"] = window_result

    if print_results:
        print_curve_fitting_results(results, label=label)

    return results

def print_curve_fitting_results(results: Dict[str, ComparisonResult], label: Optional[str] = None) -> None:
    """Print human-readable curve fitting results showing only the best model for each metric."""
    if not results:
        print("No curve fitting results to display")
        return

    print("\n" + "="*80)
    heading = "CURVE FITTING ANALYSIS RESULTS"
    if label:
        heading = f"{heading} {label}"
    print(heading)
    print("="*80)
    print("\nModel Selection Criteria:")
    print("1. F-test (p < 0.05): Quadratic significantly better than linear")
    print("2. Information Criteria: Lower AIC and higher Adjusted R² favor more complex model")
    print("3. Simplicity Principle: Linear preferred when performance is similar")
    print("="*80)

    for metric_name, result in results.items():
        # Get the selected model
        print(f"\n📊 {metric_name.replace('_', ' ').title()}")
        print("-" * 50)

        if isinstance(result, ComparisonResult):
            selected_result = result.quadratic_result if result.better_model == 'quadratic' else result.linear_result

            print(f"📈 Best Model ({result.better_model.title()}): {selected_result.equation}")
            print(f"   R² = {selected_result.r_squared:.4f}, Adj R² = {selected_result.adjusted_r_squared:.4f}")
            print(f"   AIC = {selected_result.aic:.2f}, BIC = {selected_result.bic:.2f}")

            if result.improvement_significant:
                print(f"   ✅ Selection: F-test significant (p = {result.f_test_p_value:.4f})")
            elif result.better_model == 'quadratic':
                print(f"   ✅ Selection: Better AIC/Adj R² (F-test p = {result.f_test_p_value:.4f})")
            else:
                print(f"   ✅ Selection: Simpler model preferred (F-test p = {result.f_test_p_value:.4f})")
        elif isinstance(result, ExponentialFitResult):
            print(f"📈 Exponential Decay Fit: {result.equation}")
            print(f"   R² = {result.r_squared:.4f}, Adj R² = {result.adjusted_r_squared:.4f}")
            print(f"   AIC = {result.aic:.2f}, BIC = {result.bic:.2f}")
            print("   ✅ Model: Exponential decay (a·exp(-b·t) + c)")
        else:
            print("   ⚠️  Unknown result type; skipping detailed output")

    print("\n" + "="*80)
