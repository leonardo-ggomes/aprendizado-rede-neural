import NeuralNetwork from "./NeuralNetwork.js";

// Configurações da Rede e Visualização
const nn = new NeuralNetwork(2, 4, 1); 
const canvasWidth = 800;
const canvasHeight = 500;

// Problema XOR
const dataset = {
    inputs: [[1, 1], [1, 0], [0, 1], [0, 0]],
    outputs: [[0], [1], [1], [0]]
};

let trainingCount = 0;
let currentError = 1;
let predictions = [];

function setup() {
    const canvas = createCanvas(canvasWidth, canvasHeight);
    canvas.parent('canvas-container');
}

function draw() {
    background(245);
    
    // 1. Executa o Treinamento (10000 ciclos por frame para animação fluida)
    if (currentError > 0.01) {
        for (let i = 0; i < 10000; i++) {
            const r = Math.floor(Math.random() * dataset.inputs.length);
            nn.train(dataset.inputs[r], dataset.outputs[r]);
            trainingCount++;
        }
    }

    // 2. Calcula indicadores e predições atuais
    updateMetrics();

    // 3. Desenha a Interface e Indicadores
    drawUI();

    // 4. Desenha a Rede Neural (Nós e Conexões)
    drawNetwork();
}

function updateMetrics() {
    let sumError = 0;
    predictions = dataset.inputs.map((input, i) => {
        let p = nn.predict(input)[0][0];
        sumError += Math.abs(dataset.outputs[i][0] - p);
        return p.toFixed(3);
    });
    currentError = sumError / 4;
}

function drawUI() {
    fill(50);
    noStroke();
    textSize(18);
    textStyle(BOLD);
    text("Monitor de Treinamento Neural", 200, 30);
    
    textStyle(NORMAL);
    textSize(14);
    text(`Ciclos de Treino: ${trainingCount}`, 100, 60);
    text(`Erro Médio: ${(currentError * 100).toFixed(2)}%`, 100, 80);
    
    // Tabela de Predições
    text("Predições XOR:", 100, 120);
    dataset.inputs.forEach((inp, i) => {
        text(`[${inp}] -> Alvo: ${dataset.outputs[i]} | Atual: ${predictions[i]}`, 100, 145 + (i * 20));
    });
}

function drawNetwork() {
    const nodes = {
        input: { x: 350, y: [180, 320], label: "Entrada" },
        hidden: { x: 550, y: [120, 200, 280, 360], label: "Oculta" },
        output: { x: 750, y: [240], label: "Saída" }
    };

    // --- DESENHAR CONEXÕES (PESOS) ---
    
    // Pesos Entrada -> Oculta (weights_ih)
    for (let i = 0; i < nn.i_nodes; i++) {
        for (let j = 0; j < nn.h_nodes; j++) {
            let weight = nn.weights_ih.data[j][i];
            drawConnection(nodes.input.x, nodes.input.y[i], nodes.hidden.x, nodes.hidden.y[j], weight);
        }
    }

    // Pesos Oculta -> Saída (weights_ho)
    for (let i = 0; i < nn.h_nodes; i++) {
        for (let j = 0; j < nn.o_nodes; j++) {
            let weight = nn.weights_ho.data[j][i];
            drawConnection(nodes.hidden.x, nodes.hidden.y[i], nodes.output.x, nodes.output.y[j], weight);
        }
    }

    // --- DESENHAR NÓS ---
    drawLayer(nodes.input.x, nodes.input.y, "#3498db", "In");
    drawLayer(nodes.hidden.x, nodes.hidden.y, "#9b59b6", "Hid");
    drawLayer(nodes.output.x, nodes.output.y, "#2ecc71", "Out");
}

function drawConnection(x1, y1, x2, y2, weight) {
    // Cor: Azul para positivo, Vermelho para negativo
    stroke(weight > 0 ? color(52, 152, 219, 150) : color(231, 76, 60, 150));
    // Espessura baseada no valor absoluto do peso
    strokeWeight(map(Math.abs(weight), 0, 2, 0.5, 5));
    line(x1, y1, x2, y2);
}

function drawLayer(x, yArray, col, label) {
    yArray.forEach((y, i) => {
        fill(col);
        stroke(255);
        strokeWeight(2);
        ellipse(x, y, 35, 35);
        fill(255);
        noStroke();
        textSize(10);
        textAlign(CENTER, CENTER);
        text(`${label} ${i}`, x, y);
    });
}

// Global para o p5.js
window.setup = setup;
window.draw = draw;