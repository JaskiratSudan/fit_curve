from flask import Flask, render_template, jsonify, request
import random

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

# Function to generate a random polynomial of degree n
def generate_random_polynomial(degree):
    # Generate random coefficients for the polynomial
    coefficients = [random.uniform(0, 1) for _ in range(degree + 1)]
    # Construct the polynomial function string
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

# Function to evaluate a polynomial at a given x
def evaluate_polynomial(coefficients, x):
    return sum(coef * (x ** i) for i, coef in enumerate(coefficients))

@app.route('/random_data', methods=['POST'])
def random_data():
    num_points = int(request.form.get('num_points', 10))
    degree = int(request.form.get('degree', 2))  # Degree of the polynomial
    coefficients, polynomial_function = generate_random_polynomial(degree)

    data = []
    for _ in range(num_points):
        x = random.uniform(-10, 10)  # Random x value between -10 and 10
        y = evaluate_polynomial(coefficients, x)
        data.append({'x': x, 'y': y})

    print("Polynomial Function:", polynomial_function)  # Print polynomial function to server logs

    return jsonify({'data': data, 'polynomial_function': polynomial_function})

if __name__ == '__main__':
    app.run(debug=True)
