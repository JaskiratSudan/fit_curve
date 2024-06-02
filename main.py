from flask import Flask, render_template, jsonify, request
import random
import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend for Matplotlib
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
import time

from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

def generate_random_polynomial(degree):
    coefficients = [random.randint(-10, 10) for _ in range(degree + 1)]
    terms = []
    for i, coef in enumerate(coefficients):
        if coef != 0:
            term = f"{coef:.2f}"
            if i == 1:
                term += "x"
            elif i > 1:
                term += f"x^{i}"
            terms.append(term)
    polynomial_function = "f(x) = " + " + ".join(terms)
    return coefficients, polynomial_function

def evaluate_polynomial(coefficients, x):
    result = 0
    for i, coef in enumerate(coefficients):
        result += coef * (x ** i)
    return result

def train_model_animated(model_name, x, y, x_min, x_max, **kwargs):
    model = None
    predictions = []
    if model_name == 'Linear Regression':
        model = LinearRegression(**kwargs)
        model.fit(x.reshape(-1, 1), y)
        predictions.append(model.predict(np.linspace(x_min, x_max, 100).reshape(-1, 1)))
    elif model_name == 'Neural Network':
        model = MLPRegressor(warm_start=True, random_state=0, **kwargs)
        for _ in range(kwargs['max_iter']):
            model.fit(x.reshape(-1, 1), y)
            predictions.append(model.predict(np.linspace(x_min, x_max, 100).reshape(-1, 1)))
    elif model_name == 'Decision Tree':
        model = DecisionTreeRegressor(**kwargs)
        model.fit(x.reshape(-1, 1), y)
        predictions.append(model.predict(np.linspace(x_min, x_max, 100).reshape(-1, 1)))
    else:
        raise ValueError("Invalid model name")
    return model, predictions

@app.route('/random_data', methods=['POST'])
def random_data():
    num_points = int(request.form.get('num_points', 30))
    degree = int(request.form.get('degree', 2))
    x_min = float(request.form.get('x_min', -10))
    x_max = float(request.form.get('x_max', 10))
    
    coefficients, polynomial_function = generate_random_polynomial(degree)

    data = []
    for _ in range(num_points):
        x = random.uniform(x_min, x_max)
        y = evaluate_polynomial(coefficients, x)
        data.append((x, y))

    img_base64 = generate_plot(data, polynomial_function)

    return jsonify({'img_base64': img_base64, 'polynomial_function': polynomial_function, 'data': data})

def generate_plot(data, polynomial_function, model=None, x_min=None, x_max=None):
    x_values = np.array(data)[:, 0]
    y_values = np.array(data)[:, 1]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(x_values, y_values, color='blue', label='Original Data', alpha=0.4)
    
    if model and x_min is not None and x_max is not None:
        x_model = np.linspace(x_min, x_max, 100)
        y_model = model.predict(x_model.reshape(-1, 1))
        ax.plot(x_model, y_model, color='red', label='Model Output')
    
    ax.set_title(polynomial_function)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()
    ax.grid(True)

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close(fig)

    img_base64 = base64.b64encode(img.getvalue()).decode('utf8')
    return img_base64

def generate_animation(data, predictions, polynomial_function):
    x_values = np.array(data)[:, 0]
    y_values = np.array(data)[:, 1]
    x_model = np.linspace(min(x_values), max(x_values), 100)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(x_values, y_values, color='blue', label='Original Data', alpha=0.4)
    ax.set_title(polynomial_function)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    
    ax.grid(True)
    plt.tight_layout()

    img_list = []

    line, = ax.plot([], [], color='red', label='Model Output')
    ax.legend()
    plt.tight_layout()

    for y_model in predictions:
        line.set_data(x_model, y_model)
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        img_list.append(base64.b64encode(img.getvalue()).decode('utf8'))

    plt.close(fig)
    return img_list

@app.route('/train_model', methods=['POST'])
def train_model_endpoint():
    data = request.form.get('data')
    degree = int(request.form.get('degree', 2))
    x_min = float(request.form.get('x_min', -10))
    x_max = float(request.form.get('x_max', 10))
    model_name = request.form.get('model_name', 'Linear Regression')

    data = eval(data)
    x_values = np.array(data)[:, 0]
    y_values = np.array(data)[:, 1]

    model_params = {
        'Linear Regression': {},
        'Neural Network': {'hidden_layer_sizes': (30, 30), 'max_iter': 30},
        'Decision Tree': {'max_depth': 20}
    }

    start_time = time.time()
    model, predictions = train_model_animated(model_name, x_values, y_values, x_min, x_max, **model_params[model_name])
    end_time = time.time()

    training_time = end_time - start_time
    y_pred = model.predict(x_values.reshape(-1, 1))
    mse = mean_squared_error(y_values, y_pred)
    r2 = r2_score(y_values, y_pred)

    status = f'Model trained successfully! Training Time: {training_time:.4f} seconds, MSE: {mse:.4f}, R2: {r2:.4f}'
    img_list = generate_animation(data, predictions, "")

    return jsonify({'img_list': img_list, 'status': status})

if __name__ == '__main__':
    app.run(debug=True)
