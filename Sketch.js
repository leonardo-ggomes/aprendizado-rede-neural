import NeuralNetwork from "./NeuralNetwork.js";

const nn = new NeuralNetwork(2, 4, 1);
const canvasWidth = 800;
const canvasHeight = 550;

const dataset = {
    inputs: [[1, 1], [1, 0], [0, 1], [0, 0]],
    outputs: [[0], [1], [1], [0]]
};

const COLORS = {
    bg: 255,
    text: "#2D3436",
    accent: "#6C5CE7",
    positive: "#00B894",
    negative: "#FF7675",
    neutral: "#F0F2F5", // Cinza bem clarinho para fundos
    graph: "rgba(108, 92, 231, 0.15)"
};

let trainingCount = 0;
let currentError = 1;
let predictions = [];
let errorHistory = [];

function setup() {
    const canvas = createCanvas(canvasWidth, canvasHeight);
    canvas.parent('canvas-container');
}

function draw() {
    background(COLORS.bg);
    
    if (currentError > 0.01) {
        for (let i = 0; i < 10000; i++) {
            const r = Math.floor(Math.random() * dataset.inputs.length);
            nn.train(dataset.inputs[r], dataset.outputs[r]);
            trainingCount++;
        }
    }

    updateMetrics();
    
    // 1. Gráfico (ajustado para a parte inferior com margem)
    drawErrorGraph();
    
    // 2. Interface Lateral (Painel de controle)
    drawSidebar();

    // 3. Rede Neural (Centralizada no espaço restante)
    drawNetwork();
}

function updateMetrics() {
    let sumError = 0;
    predictions = dataset.inputs.map((input, i) => {
        let p = nn.predict(input)[0][0];
        sumError += Math.abs(dataset.outputs[i][0] - p);
        return p;
    });
    currentError = sumError / 4;
    
    if (frameCount % 5 === 0) {
        errorHistory.push(currentError);
        if (errorHistory.length > 200) errorHistory.shift();
    }
}

function drawSidebar() {
    push();
    const margin = 120;
    translate(margin, margin + 10);
    
    // Título Principal
    fill(COLORS.text);
    noStroke();
    textSize(20);
    textStyle(BOLD);
    text("Neural Engine", 0, 0);
    
    // Status condensado
    textStyle(NORMAL);
    textSize(11);
    fill(140);
    text(`ITERATIONS: ${trainingCount.toLocaleString()}`, 0, 22);
    text(`AVG ERROR: ${(currentError * 100).toFixed(2)}%`, 0, 36);

    // Tabela XOR (Mais compacta)
    translate(0, 75);
    textSize(10);
    textStyle(BOLD);
    text("PREDICTIONS", 0, 0);
    
    textStyle(NORMAL);
    dataset.inputs.forEach((inp, i) => {
        const val = predictions[i];
        const y = 20 + (i * 18);
        fill(160);
        text(`[${inp}]`, 0, y);
        fill(COLORS.text);
        text(`→  ${val.toFixed(3)}`, 45, y);
    });
    pop();
}

function drawErrorGraph() {
    push();
    const gW = 220; // Largura do gráfico
    const gH = 60;  // Altura do gráfico
    const gX = 30;  // Margem X
    const gY = canvasHeight - 90; // Margem Y (fundo)

    // Fundo do gráfico (sutil)
    fill(COLORS.neutral);
    noStroke();
    rect(gX - 5, gY - 15, gW + 10, gH + 20, 4);

    // Linha do erro
    noFill();
    stroke(COLORS.accent);
    strokeWeight(1.5);
    beginShape();
    errorHistory.forEach((err, i) => {
        let x = map(i, 0, 200, gX, gX + gW);
        let y = map(err, 0, 1, gY + gH, gY);
        vertex(x, y);
    });
    endShape();

    fill(120);
    noStroke();
    textSize(9);
    text("CONVERGENCE HISTORY", gX, gY - 22);
    pop();
}

function drawNetwork() {
    // Coordenadas calculadas para evitar sair da margem direita
    const netX = 320; 
    const nodes = {
        input: { x: netX, y: [200, 350], label: "IN" },
        hidden: { x: netX + 180, y: [130, 225, 320, 415], label: "H" },
        output: { x: netX + 360, y: [275], label: "OUT" }
    };

    drawWeightLines(nodes.input, nodes.hidden, nn.weights_ih.data);
    drawWeightLines(nodes.hidden, nodes.output, nn.weights_ho.data);

    drawLayerNodes(nodes.input.x, nodes.input.y, nodes.input.label, COLORS.accent);
    drawLayerNodes(nodes.hidden.x, nodes.hidden.y, nodes.hidden.label, COLORS.accent);
    drawLayerNodes(nodes.output.x, nodes.output.y, nodes.output.label, COLORS.positive);
}

function drawWeightLines(layerA, layerB, weights) {
    for (let i = 0; i < layerA.y.length; i++) {
        for (let j = 0; j < layerB.y.length; j++) {
            let weight = weights[j][i];
            let alpha = map(Math.abs(weight), 0, 2, 40, 200);
            stroke(weight > 0 ? COLORS.positive : COLORS.negative);
            strokeWeight(map(Math.abs(weight), 0, 2, 0.5, 3));
            drawingContext.globalAlpha = alpha / 255;
            line(layerA.x, layerA.y[i], layerB.x, layerB.y[j]);
            drawingContext.globalAlpha = 1;
        }
    }
}

function drawLayerNodes(x, yArray, labelPrefix, colorHex) {
    yArray.forEach((y, i) => {
        fill(colorHex);
        noStroke();
        ellipse(x, y, 30, 30);
        
        // Borda de contraste
        noFill();
        stroke(COLORS.bg);
        strokeWeight(2);
        ellipse(x, y, 32, 32);

        fill(255);
        noStroke();
        textAlign(CENTER, CENTER);
        textSize(9);
        textStyle(BOLD);
        text(`${labelPrefix}${i}`, x, y);
    });
}

window.setup = setup;

window.draw = draw;
