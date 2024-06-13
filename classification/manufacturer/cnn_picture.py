import matplotlib.pyplot as plt
import networkx as nx

# Definieren der Neuronen pro Schicht
neurons_per_layer = {
    'Input': 3,
    'Conv1': 4,
    'Pool1': 4,
    'Conv2': 8,
    'Pool2': 8,
    'FC1': 8,
    'Output': 2
}

# Definieren der Verbindungen zwischen den Schichten
connections = [
    ('Input', 'Conv1'),
    ('Conv1', 'Pool1'),
    ('Pool1', 'Conv2'),
    ('Conv2', 'Pool2'),
    ('Pool2', 'FC1'),
    ('FC1', 'Output')
]

# Erstellen eines leeren gerichteten Graphen
G = nx.DiGraph()

# Hinzufügen der Schichten und Neuronen als Knoten zum Graphen
for layer, num_neurons in neurons_per_layer.items():
    layer_position = list(neurons_per_layer.keys()).index(layer)
    y_positions = list(range(num_neurons))
    if len(y_positions) > 1:
        y_positions = [y - (len(y_positions) - 1) / 2 for y in y_positions]
    else:
        y_positions = [0]
    for idx, neuron in enumerate(range(num_neurons)):
        G.add_node(f'{layer}_{neuron+1}', pos=(layer_position, y_positions[idx]))

# Hinzufügen der Verbindungen zwischen den Schichten als Kanten zum Graphen
for connection in connections:
    layer_from, layer_to = connection
    if layer_to.startswith('Conv'):  # Wenn die Ziel-Schicht eine Convolutional Schicht ist
        num_neurons_from = neurons_per_layer[layer_from]
        num_neurons_to = neurons_per_layer[layer_to]
        # Verbinden nur benachbarte Neuronen entsprechend der Kernelgröße
        kernel_size = 3  # Beispiel: Kernelgröße
        stride = 1  # Beispiel: Stride
        for neuron_from in range(num_neurons_from):
            for neuron_to in range(num_neurons_to):
                # Nur verbinden, wenn neuron_to in der Reichweite des Kernels liegt
                if abs(neuron_from - neuron_to) < kernel_size * stride:
                    G.add_edge(f'{layer_from}_{neuron_from+1}', f'{layer_to}_{neuron_to+1}')

    else:  # Für alle anderen Schichten
        for neuron_from in range(neurons_per_layer[layer_from]):
            for neuron_to in range(neurons_per_layer[layer_to]):
                G.add_edge(f'{layer_from}_{neuron_from+1}', f'{layer_to}_{neuron_to+1}')

# Zeichnen des Graphen
pos = nx.get_node_attributes(G, 'pos')
nx.draw(G, pos, with_labels=True, node_size=1500, node_color='lightblue', font_size=10, font_color='black', font_weight='bold', edge_color='gray', arrows=True, arrowstyle='-|>')
plt.title('Schematic Representation of Neural Network with Conv, Pooling, and FC Layers (Localized Connections)')
plt.show()
