#Hecho por Alexandro Gutierrez Serna

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Carga el conjunto de datos
data = pd.read_csv("Employee.csv")

# Preprocesamiento de datos
# Codificación de variables categóricas con one-hot encoding
data = pd.get_dummies(data, columns=["Education", "City", "Gender", "EverBenched"], drop_first=True)

# Convertir columnas booleanas a enteros (True a 1, False a 0)
boolean_columns = ["Education_Masters", "Education_PHD", "City_New Delhi", "City_Pune", "Gender_Male", "EverBenched_Yes"]
data[boolean_columns] = data[boolean_columns].astype(int)

# Normaliza variables numéricas
data["Age"] = (data["Age"] - data["Age"].mean()) / data["Age"].std()

#Para ver todas las columnas de datos
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
print(data)

# Divide los datos en conjuntos de entrenamiento y prueba
train_size = int(0.8 * len(data))
train_data = data[:train_size]
test_data = data[train_size:]

# Define la función sigmoide
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Inicializa los parámetros (pesos y sesgo)
num_features = len(data.columns) - 1  # Excluye la columna de LeaveOrNot
weights = np.random.randn(num_features)
bias = np.random.rand()

print("Número de características:", num_features)
print("Pesos iniciales:", weights)
print("Sesgo inicial:", bias)

# Define la función de regresión logística
def logistic_regression(features, weights, bias):
    z = np.dot(features, weights) + bias
    return sigmoid(z)

# Define la función de entrenamiento usando descenso del gradiente
def train_logistic_regression(features, labels, weights, bias, learning_rate, num_iterations):
    m = len(labels)
    for _ in range(num_iterations):
        predictions = logistic_regression(features, weights, bias)
        dw = (1/m) * np.dot(features.T, (predictions - labels))
        db = (1/m) * np.sum(predictions - labels)
        weights -= learning_rate * dw
        bias -= learning_rate * db
    return weights, bias

# Entrena el modelo
learning_rate = 0.0001
num_iterations = 1000
train_features = data[['JoiningYear', 'PaymentTier', 'Age', 'ExperienceInCurrentDomain', 'Education_Masters', 'Education_PHD', 'City_New Delhi', 'City_Pune', 'Gender_Male', 'EverBenched_Yes']].values
train_labels = data["LeaveOrNot"].values

weights, bias = train_logistic_regression(train_features, train_labels, weights, bias, learning_rate, num_iterations)

print("Pesos finales:", weights)
print("Sesgo final:", bias)

"""
print("#0 - Probando a mano con 2017 3 0.954543 -1.864701 0 0 0 0 1 0")
res1 = 1/(1+np.exp(-(bias + weights[0]*2017 + weights[1]*3 + weights[2]*0.954543 + weights[3]*-1.864701 + weights[4]*0 + weights[5]*0 + weights[6]*0 + weights[7]*0 + weights[8]*1 + weights[9]*0)))
print("Resultado:", res1)

print("#1 - Probando a mano con 2013 1 -0.288701 0.060548 0 0 0 1 0 0")
res2 = 1/(1+np.exp(-(bias + weights[0]*2013 + weights[1]*1 + weights[2]*-0.288701 + weights[3]*0.060548 + weights[4]*0 + weights[5]*0 + weights[6]*0 + weights[7]*1 + weights[8]*0 + weights[9]*0)))
print("Resultado:", res2)

print("#3719 - Probando a mano con 2015 3 1.161750 -0.581202 0 0 0 1 0 0")
res3 = 1/(1+np.exp(-(bias + weights[0]*2015 + weights[1]*3 + weights[2]*1.161750 + weights[3]*-0.581202 + weights[4]*0 + weights[5]*0 + weights[6]*0 + weights[7]*1 + weights[8]*0 + weights[9]*0)))
print("Resultado:", res3)

print("#3721 - Probando a mano con 2014 3 0.540128 1.344047 1 0 1 0 0 0")
res4 = 1/(1+np.exp(-(bias + weights[0]*2014 + weights[1]*3 + weights[2]*0.540128 + weights[3]*1.344047 + weights[4]*1 + weights[5]*0 + weights[6]*1 + weights[7]*0 + weights[8]*0 + weights[9]*0)))
print("Resultado:", res4)
"""

# Define una función para hacer predicciones
def predict(features, weights, bias):
    return (logistic_regression(features, weights, bias) >= 0.5).astype(int)

# Evalúa el modelo en el conjunto de prueba
test_features = test_data.drop("LeaveOrNot", axis=1).values
test_labels = test_data["LeaveOrNot"].values
predictions = predict(test_features, weights, bias)

print("Predicciones:", predictions)
print("Etiquetas reales:", test_labels)

# Calcular la precisión del modelo
accuracy = np.mean(predictions == test_labels)
print(f"Precisión del modelo: {accuracy}")


#Grafica los valores reales vs los valores predichos
plt.figure(figsize=(8, 6))
plt.scatter(range(len(test_labels)), test_labels, label="Valores reales", color="blue", marker="o")
plt.scatter(range(len(predictions)), predictions, label="Valores predichos", color="red", marker="x")
plt.title("Valores reales vs. Valores predichos")
plt.xlabel("Muestras")
plt.ylabel("Valores")
plt.legend()
plt.show()
