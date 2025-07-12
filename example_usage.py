import numpy as np
import matplotlib.pyplot as plt
from tree_coef_linear import TreeCoefficientLinear
from tree_coef_linear_v2 import TreeCoefficientLinearV2
from vcboost import VCBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.preprocessing import PolynomialFeatures


def generate_synthetic_data(n_samples=1000, random_state=42):
    """
    Generate synthetic data where:
    y = T1(x) * z1 + T2(x) * z2 + noise
    
    where T1 and T2 are interesting non-linear functions
    """
    np.random.seed(random_state)
    
    # Feature variable - 3D
    X = np.random.uniform(-3, 3, (n_samples, 3))
    
    # Coefficient variables
    z1 = np.random.randn(n_samples)
    z2 = np.random.randn(n_samples)
    Z = np.column_stack([z1, z2])
    
    # True non-linear functions - designed to be hard for polynomials
    # T1: High-frequency oscillation using multiple dimensions
    T1_X = np.sin(5 * X[:, 0]) * np.exp(-0.1 * np.abs(X[:, 1])) * np.cos(2 * X[:, 2])
    
    # T2: Step-like function using interaction of dimensions
    T2_X = 0.5 * (np.tanh(10 * (X[:, 0] + X[:, 1] - 1)) + np.tanh(10 * (X[:, 2] - X[:, 0] + 1)))
    
    # Generate target with higher noise
    y = T1_X * z1 + T2_X * z2 + 1.0 * np.random.randn(n_samples)
    
    return X, y, Z, T1_X, T2_X


def main():
    print("TreeCoefficientLinear Example")
    print("=" * 50)
    
    # Generate synthetic data
    X, y, Z, true_T1, true_T2 = generate_synthetic_data(n_samples=2000)
    
    # Split data
    (X_train, X_test, y_train, y_test, 
     Z_train, Z_test) = train_test_split(X, y, Z, test_size=0.3, random_state=42)
    
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    print(f"Features: {X_train.shape[1]}")
    print(f"Coefficient variables: {Z_train.shape[1]}")
    print()
    
    # Example 1: Using XGBoost backend with V2
    print("Example 1: TreeCoefficientLinearV2 with XGBoost")
    print("-" * 40)
    
    model_xgb = TreeCoefficientLinearV2(
        n_components=2,
        base_estimator='xgboost',
        n_estimators=300,
        learning_rate=0.03,
        max_depth=2,
        reg_alpha=1.0,
        reg_lambda=5.0,
        subsample=0.7,
        colsample_bytree=0.7,
        random_state=42
    )
    
    model_xgb.fit(X_train, y_train, Z_train)
    y_pred_xgb = model_xgb.predict(X_test, Z_test)
    
    mse_xgb = mean_squared_error(y_test, y_pred_xgb)
    r2_xgb = r2_score(y_test, y_pred_xgb)
    
    print(f"MSE: {mse_xgb:.4f}")
    print(f"R² Score: {r2_xgb:.4f}")
    print()
    
    # Example 2: Using LightGBM backend with V2
    print("Example 2: TreeCoefficientLinearV2 with LightGBM")
    print("-" * 40)
    
    model_lgb = TreeCoefficientLinearV2(
        n_components=2,
        base_estimator='lightgbm',
        n_estimators=300,
        learning_rate=0.03,
        max_depth=2,
        reg_alpha=1.0,
        reg_lambda=5.0,
        feature_fraction=0.7,
        bagging_fraction=0.7,
        bagging_freq=5,
        random_state=42
    )
    
    model_lgb.fit(X_train, y_train, Z_train)
    y_pred_lgb = model_lgb.predict(X_test, Z_test)
    
    mse_lgb = mean_squared_error(y_test, y_pred_lgb)
    r2_lgb = r2_score(y_test, y_pred_lgb)
    
    print(f"MSE: {mse_lgb:.4f}")
    print(f"R² Score: {r2_lgb:.4f}")
    print()
    
    # Example 3: Using VCBoost with vector-leaf trees
    print("Example 3: VCBoost with vector-leaf trees")
    print("-" * 40)
    
    model_vcb = VCBoostRegressor(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=2,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_lambda=5.0,
        random_state=42
    )
    
    model_vcb.fit(X_train, y_train, Z_train)
    y_pred_vcb = model_vcb.predict(X_test, Z_test)
    
    mse_vcb = mean_squared_error(y_test, y_pred_vcb)
    r2_vcb = r2_score(y_test, y_pred_vcb)
    
    print(f"MSE: {mse_vcb:.4f}")
    print(f"R² Score: {r2_vcb:.4f}")
    print()
    
    # Compare learned functions with true functions
    print("Comparing learned vs true functions")
    print("-" * 40)
    
    # Get learned tree functions
    T_values = model_xgb.get_tree_functions(X_test)
    T_values_vcb = model_vcb.predict_components(X_test)
    
    # For the test set, get the true T1 and T2 values
    # We need to get the true values for the actual test samples
    X_full_train, X_full_test, _, _, _, _, true_T1_train, true_T1_test, true_T2_train, true_T2_test = train_test_split(
        X, y, Z, true_T1, true_T2, test_size=0.3, random_state=42
    )
    
    # Compute correlations
    corr_T1 = np.corrcoef(T_values[:, 0], true_T1_test)[0, 1]
    corr_T2 = np.corrcoef(T_values[:, 1], true_T2_test)[0, 1]
    corr_T1_vcb = np.corrcoef(T_values_vcb[:, 0], true_T1_test)[0, 1]
    corr_T2_vcb = np.corrcoef(T_values_vcb[:, 1], true_T2_test)[0, 1]
    
    print(f"TreeCoef XGB - Correlation T₁: {corr_T1:.4f}, T₂: {corr_T2:.4f}")
    print(f"VCBoost - Correlation T₁: {corr_T1_vcb:.4f}, T₂: {corr_T2_vcb:.4f}")
    print()
    
    # Add linear regression for comparison
    # Simple linear regression with X*Z interactions only
    X_Z_train_simple = np.column_stack([
        X_train,  # X1, X2, X3
        Z_train,  # Z1, Z2
        X_train[:, 0:1] * Z_train[:, 0:1],  # X1*Z1
        X_train[:, 0:1] * Z_train[:, 1:2],  # X1*Z2
        X_train[:, 1:2] * Z_train[:, 0:1],  # X2*Z1
        X_train[:, 1:2] * Z_train[:, 1:2],  # X2*Z2
        X_train[:, 2:3] * Z_train[:, 0:1],  # X3*Z1
        X_train[:, 2:3] * Z_train[:, 1:2]   # X3*Z2
    ])
    X_Z_test_simple = np.column_stack([
        X_test,  # X1, X2, X3
        Z_test,  # Z1, Z2
        X_test[:, 0:1] * Z_test[:, 0:1],  # X1*Z1
        X_test[:, 0:1] * Z_test[:, 1:2],  # X1*Z2
        X_test[:, 1:2] * Z_test[:, 0:1],  # X2*Z1
        X_test[:, 1:2] * Z_test[:, 1:2],  # X2*Z2
        X_test[:, 2:3] * Z_test[:, 0:1],  # X3*Z1
        X_test[:, 2:3] * Z_test[:, 1:2]   # X3*Z2
    ])
    linear_model_simple = LinearRegression()
    linear_model_simple.fit(X_Z_train_simple, y_train)
    y_pred_linear_simple = linear_model_simple.predict(X_Z_test_simple)
    mse_linear_simple = mean_squared_error(y_test, y_pred_linear_simple)
    r2_linear_simple = r2_score(y_test, y_pred_linear_simple)
    
    # Test multiple polynomial degrees
    # Combine X and Z into single feature matrix
    XZ_train = np.column_stack([X_train, Z_train])
    XZ_test = np.column_stack([X_test, Z_test])
    
    # Test degrees 3, 5, and 7
    poly_degrees = [3, 5, 7]
    poly_results = {}
    
    for degree in poly_degrees:
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        XZ_poly_train = poly.fit_transform(XZ_train)
        XZ_poly_test = poly.transform(XZ_test)
        
        # Standard linear regression
        linear_model_poly = LinearRegression()
        linear_model_poly.fit(XZ_poly_train, y_train)
        y_pred_poly = linear_model_poly.predict(XZ_poly_test)
        
        mse_poly = mean_squared_error(y_test, y_pred_poly)
        r2_poly = r2_score(y_test, y_pred_poly)
        
        # Ridge regression with cross-validation
        alphas = np.logspace(-3, 3, 100)
        ridge_model = RidgeCV(alphas=alphas, cv=5)
        ridge_model.fit(XZ_poly_train, y_train)
        y_pred_ridge = ridge_model.predict(XZ_poly_test)
        
        mse_ridge = mean_squared_error(y_test, y_pred_ridge)
        r2_ridge = r2_score(y_test, y_pred_ridge)
        
        poly_results[degree] = {
            'mse': mse_poly,
            'r2': r2_poly,
            'mse_ridge': mse_ridge,
            'r2_ridge': r2_ridge,
            'alpha': ridge_model.alpha_,
            'n_features': XZ_poly_train.shape[1],
            'y_pred': y_pred_poly,
            'y_pred_ridge': y_pred_ridge
        }
    
    # Use degree 7 Ridge for main comparison
    mse_linear_poly = poly_results[7]['mse_ridge']
    r2_linear_poly = poly_results[7]['r2_ridge']
    y_pred_linear_poly = poly_results[7]['y_pred_ridge']
    
    
    # Use polynomial model for visualization
    y_pred_linear = y_pred_linear_poly
    mse_linear = mse_linear_poly
    r2_linear = r2_linear_poly

    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Sort data by first dimension for visualization
    sort_idx = np.argsort(X_test[:, 0])
    X_test_sorted = X_test[sort_idx, 0]
    T_values_sorted = T_values[sort_idx]
    true_T1_test_sorted = true_T1_test[sort_idx]
    true_T2_test_sorted = true_T2_test[sort_idx]
    
    # Plot 1: True vs Learned T1 function (projected on x1)
    axes[0, 0].scatter(X_test_sorted, true_T1_test_sorted, alpha=0.5, s=20, label='True T₁(X)', color='blue')
    axes[0, 0].scatter(X_test_sorted, T_values_sorted[:, 0], alpha=0.5, s=20, label='Learned T₁(X)', color='red')
    axes[0, 0].set_xlabel('x₁')
    axes[0, 0].set_ylabel('T₁(X)')
    axes[0, 0].set_title(f'T₁: Multi-dim Oscillation (ρ = {corr_T1:.3f})')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: True vs Learned T2 function (projected on x1)
    axes[0, 1].scatter(X_test_sorted, true_T2_test_sorted, alpha=0.5, s=20, label='True T₂(X)', color='blue')
    axes[0, 1].scatter(X_test_sorted, T_values_sorted[:, 1], alpha=0.5, s=20, label='Learned T₂(X)', color='red')
    axes[0, 1].set_xlabel('x₁')
    axes[0, 1].set_ylabel('T₂(X)')
    axes[0, 1].set_title(f'T₂: Multi-dim Step Function (ρ = {corr_T2:.3f})')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Predicted vs Actual (TreeCoefficientLinear)
    axes[1, 0].scatter(y_test, y_pred_xgb, alpha=0.5, label='TreeCoefLinear')
    axes[1, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    axes[1, 0].set_xlabel('Actual')
    axes[1, 0].set_ylabel('Predicted')
    axes[1, 0].set_title(f'TreeCoefLinear: Predicted vs Actual (R² = {r2_xgb:.3f})')
    
    # Plot 4: Predicted vs Actual (Linear Regression)
    axes[1, 1].scatter(y_test, y_pred_linear, alpha=0.5, color='orange', label='Linear Regression')
    axes[1, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    axes[1, 1].set_xlabel('Actual')
    axes[1, 1].set_ylabel('Predicted')
    axes[1, 1].set_title(f'Ridge Polynomial (deg 7): Predicted vs Actual (R² = {r2_linear:.3f})')
    
    # Plot 5: Performance Comparison
    ax5 = axes[1, 2]
    
    # Prepare data for bar chart
    model_names = ['Linear\n(X*Z)', 'Poly\n(deg 7)', 'TreeCoef\n(XGB)', 'TreeCoef\n(LGB)', 'VCBoost']
    r2_scores = [r2_linear_simple, r2_linear_poly, r2_xgb, r2_lgb, r2_vcb]
    colors = ['lightcoral', 'orange', 'blue', 'green', 'purple']
    
    # Create bar chart
    bars = ax5.bar(model_names, r2_scores, color=colors, alpha=0.7)
    
    # Add feature counts as text labels
    feature_counts = [
        11,  # X1,X2,X3, Z1, Z2, X1*Z1, X1*Z2, X2*Z1, X2*Z2, X3*Z1, X3*Z2
        poly_results[7]['n_features'],
        2,  # TreeCoef learns 2 functions
        2,  # TreeCoef learns 2 functions
        2   # VCBoost learns 2 functions
    ]
    
    for bar, r2, n_feat in zip(bars, r2_scores, feature_counts):
        # R² score on top of bar
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{r2:.3f}', ha='center', va='bottom', fontsize=10)
        # Number of features at bottom
        ax5.text(bar.get_x() + bar.get_width()/2, 0.05, 
                f'{n_feat} feat', ha='center', va='bottom', fontsize=8, color='white')
    
    ax5.set_ylabel('R² Score')
    ax5.set_title('Model Performance Comparison')
    ax5.set_ylim(0, 1.05)
    ax5.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('tree_coefficient_linear_results.png', dpi=150, bbox_inches='tight')
    print("Results visualization saved to 'tree_coefficient_linear_results.png'")
    plt.show()
    
    # Example 4: Using more components
    print("\nExample 4: Using more components than needed")
    print("-" * 40)
    
    model_extra = TreeCoefficientLinearV2(
        n_components=4,  # Using 4 components when data has only 2
        base_estimator='xgboost',
        n_estimators=300,
        learning_rate=0.03,
        max_depth=2,
        reg_alpha=1.0,
        reg_lambda=5.0,
        subsample=0.7,
        colsample_bytree=0.7,
        random_state=42
    )
    
    # Create Z with extra zero columns
    Z_train_extra = np.column_stack([Z_train, np.zeros((len(Z_train), 2))])
    Z_test_extra = np.column_stack([Z_test, np.zeros((len(Z_test), 2))])
    
    model_extra.fit(X_train, y_train, Z_train_extra)
    y_pred_extra = model_extra.predict(X_test, Z_test_extra)
    
    mse_extra = mean_squared_error(y_test, y_pred_extra)
    r2_extra = r2_score(y_test, y_pred_extra)
    
    print(f"MSE with 4 components: {mse_extra:.4f}")
    print(f"R² Score with 4 components: {r2_extra:.4f}")
    print("(Should perform similarly since extra components have zero coefficients)")
    print()
    
    # Baseline comparison results (already computed above)
    print("Baseline: Linear Regression Comparisons")
    print("-" * 40)
    print(f"Linear Regression (X*Z only) MSE: {mse_linear_simple:.4f}")
    print(f"Linear Regression (X*Z only) R² Score: {r2_linear_simple:.4f}")
    print()
    print("Polynomial Regression Results:")
    for degree in poly_degrees:
        print(f"  Degree {degree} (OLS):   R² = {poly_results[degree]['r2']:.4f}, "
              f"MSE = {poly_results[degree]['mse']:.4f}")
        print(f"  Degree {degree} (Ridge): R² = {poly_results[degree]['r2_ridge']:.4f}, "
              f"MSE = {poly_results[degree]['mse_ridge']:.4f}, "
              f"α = {poly_results[degree]['alpha']:.2e}, "
              f"Features = {poly_results[degree]['n_features']}")
    print()
    
    # Comparison summary
    print("Model Comparison Summary")
    print("=" * 40)
    print(f"Linear Regression (X*Z):      R² = {r2_linear_simple:.4f}, MSE = {mse_linear_simple:.4f}")
    print(f"Ridge Polynomial (deg 7):      R² = {r2_linear_poly:.4f}, MSE = {mse_linear_poly:.4f}")
    print(f"TreeCoefficientLinear (XGB):   R² = {r2_xgb:.4f}, MSE = {mse_xgb:.4f}")
    print(f"TreeCoefficientLinear (LGB):   R² = {r2_lgb:.4f}, MSE = {mse_lgb:.4f}")
    print(f"VCBoost (vector-leaf):         R² = {r2_vcb:.4f}, MSE = {mse_vcb:.4f}")
    
    improvement_xgb_poly = ((mse_linear_poly - mse_xgb) / mse_linear_poly) * 100
    improvement_lgb_poly = ((mse_linear_poly - mse_lgb) / mse_linear_poly) * 100
    improvement_vcb_poly = ((mse_linear_poly - mse_vcb) / mse_linear_poly) * 100
    
    print(f"\nMSE Improvement vs Polynomial:")
    print(f"TreeCoef XGB: {improvement_xgb_poly:.1f}% better")
    print(f"TreeCoef LGB: {improvement_lgb_poly:.1f}% better") 
    print(f"VCBoost: {improvement_vcb_poly:.1f}% better")


if __name__ == "__main__":
    main()
