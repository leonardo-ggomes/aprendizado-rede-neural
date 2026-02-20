import Matrix from "./Matrix.js";

class NeuralNetwork {
    constructor(i_nodes, h_nodes, o_nodes) {
        this.i_nodes = i_nodes;
        this.h_nodes = h_nodes;
        this.o_nodes = o_nodes;
        this.learning_rate = 0.1; 

        // Pesos independentes para cada camada
        this.weights_ih = new Matrix(h_nodes, i_nodes);
        this.weights_ho = new Matrix(o_nodes, h_nodes);
        this.weights_ih.randomize();
        this.weights_ho.randomize();

        // Bias independentes para cada camada
        this.bias_h = new Matrix(h_nodes, 1);
        this.bias_o = new Matrix(o_nodes, 1);
        this.bias_h.randomize();
        this.bias_o.randomize();
    }

    static sigmoid(x) {
        return 1 / (1 + Math.exp(-x)); 
    }

    static dsigmoid(x) {
        return x * (1 - x); 
    }

    train(inputs, target) {
        // --- FeedFoward ---
        let input = Matrix.arrayToMatrix(inputs);
        
        // Input -> Hidden
        let hidden = Matrix.multiply(this.weights_ih, input);
        hidden = Matrix.add(hidden, this.bias_h);
        hidden.map(NeuralNetwork.sigmoid);

        // Hidden -> Output
        let output = Matrix.multiply(this.weights_ho, hidden);
        output = Matrix.add(output, this.bias_o);
        output.map(NeuralNetwork.sigmoid);

        // --- BACKPROPAGATION ---
        let targets = Matrix.arrayToMatrix(target);
        let output_errors = Matrix.subtract(targets, output);

        // 1. Calcular Gradiente da Saída
        // Gradiente = dsigmoid(output) * erro * learning_rate
        let gradients = Matrix.map(output, NeuralNetwork.dsigmoid);
        gradients = Matrix.hadamard(gradients, output_errors);
        gradients = Matrix.multiply_scalar(gradients, this.learning_rate);

        // 2. Calcular Deltas de Hidden -> Output
        let hidden_T = Matrix.transpose(hidden);
        let weights_ho_deltas = Matrix.multiply(gradients, hidden_T);

        // Atualizar Pesos e Bias da Saída (SOMA, não substituição)
        this.weights_ho = Matrix.add(this.weights_ho, weights_ho_deltas);
        this.bias_o = Matrix.add(this.bias_o, gradients);

        // 3. Calcular Erro da Camada Oculta (Retropropagação)
        let who_T = Matrix.transpose(this.weights_ho);
        let hidden_errors = Matrix.multiply(who_T, output_errors);

        // 4. Calcular Gradiente da Camada Oculta
        let hidden_gradient = Matrix.map(hidden, NeuralNetwork.dsigmoid);
        hidden_gradient = Matrix.hadamard(hidden_gradient, hidden_errors);
        hidden_gradient = Matrix.multiply_scalar(hidden_gradient, this.learning_rate);

        // 5. Calcular Deltas de Input -> Hidden
        let input_T = Matrix.transpose(input);
        let weights_ih_deltas = Matrix.multiply(hidden_gradient, input_T);

        // Atualizar Pesos e Bias da Entrada
        this.weights_ih = Matrix.add(this.weights_ih, weights_ih_deltas);
        this.bias_h = Matrix.add(this.bias_h, hidden_gradient);
    }

    predict(inputs) {
        let input = Matrix.arrayToMatrix(inputs);
        let hidden = Matrix.multiply(this.weights_ih, input);
        hidden = Matrix.add(hidden, this.bias_h);
        hidden.map(NeuralNetwork.sigmoid);

        let output = Matrix.multiply(this.weights_ho, hidden);
        output = Matrix.add(output, this.bias_o);
        output.map(NeuralNetwork.sigmoid);

        return output.data; // Retorna o array de resultados
    }
}
export default NeuralNetwork;