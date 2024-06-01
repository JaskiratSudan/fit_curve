from flask import Flask, render_template, jsonify, request
import random
import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend for Matplotlib
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

def generate_random_polynomial(degree):
    coefficients = [random.uniform(-10, 10) for _ in range(degree + 1)]
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

@app.route('/random_data', methods=['POST'])
def random_data():
    num_points = int(request.form.get('num_points', 30))
    degree = int(request.form.get('degree', 2))
    x_range = request.form.get('x_range', '-1 to 1')  # Get the range of x values
    x_min, x_max = map(float, x_range.split(' to '))  # Parse the range string
    
    coefficients, polynomial_function = generate_random_polynomial(degree)

    data = []
    for _ in range(num_points):
        x = random.uniform(x_min, x_max)
        y = evaluate_polynomial(coefficients, x)
        data.append((x, y))

    fig, ax = plt.subplots(figsize=(11, 6.5))  # Increase figure size
    x_vals, y_vals = zip(*data)
    ax.scatter(x_vals, y_vals, color='blue', alpha=0.75)  # Simple scatter plot without custom marker style
    ax.set_xlabel('x')  # Remove fontsize and fontweight
    ax.set_ylabel('y')  # Remove fontsize and fontweight
    ax.axhline(y=0, color='k', linewidth=1.5)  # Remove fontweight
    ax.axvline(x=0, color='k', linewidth=1.5)  # Remove fontweight
    ax.grid()
    plt.tight_layout()

    
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plt.close(fig)

    img_base64 = base64.b64encode(img.getvalue()).decode('utf8')
    return jsonify({'img_base64': img_base64, 'polynomial_function': polynomial_function})

if __name__ == '__main__':
    app.run(debug=True)
